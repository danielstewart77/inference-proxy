"""`python -m app.cli` — the rows an install needs, without the console.

A fresh database has no provider, model or client, and the console that makes
them is a cookie-authenticated HTML form. An installer has neither a browser
nor a reason to log in, so these commands make the same rows through the same
functions the console's forms call (`insert_*`, `issue_key`) — validation and
all — and print what they did as JSON on stdout.

Every command is safe to repeat. A name already taken is reused and reported
with `"created": false`, never changed: a key rotated or a URL corrected in the
console since the first run stays as the console left it.

Exit status: 0 on success, 2 when the input is refused (the JSON carries the
reason), 1 when the database cannot be reached or holds no proxy schema.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Optional, Sequence

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.clients import VALID_KINDS as CLIENT_KINDS, insert_client, issue_key
from app.admin.credentials import insert_credential
from app.admin.models import insert_model
from app.admin.providers import insert_provider
from app.db import SessionLocal, dispose_engine
from app.deployments import HARNESS_WIRES, listing_for
from app.orm import ApiKey, Client, Credential, Model, Provider

#: What a provider kind means as a row: where it lives by default, the paths
#: of the request shapes it serves, and how it takes its key. Ollama has no
#: default address and ignores the key, so one is optional for it.
PROVIDER_KINDS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "auth_scheme": "x_api_key",
        "messages_path": "/v1/messages",
        "label": "Anthropic",
        "key_required": True,
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "auth_scheme": "bearer",
        "responses_path": "/v1/responses",
        "chat_completions_path": "/v1/chat/completions",
        "label": "OpenAI",
        "key_required": True,
    },
    "ollama": {
        "base_url": None,
        "auth_scheme": "bearer",
        "messages_path": "/v1/messages",
        "responses_path": "/v1/responses",
        "chat_completions_path": "/v1/chat/completions",
        "label": "Ollama",
        "key_required": False,
    },
}

#: Sent to an Ollama that was given no key. Every provider authenticates
#: through a credential and Ollama ignores the header, so this is a value
#: that has to exist, not a secret.
OLLAMA_PLACEHOLDER_KEY = "ollama"

#: A model's rotation threshold when none is given: the value migration 0008
#: gave every existing row.
DEFAULT_ROTATION_THRESHOLD_PERCENT = 30


class Refused(Exception):
    """Input the command will not act on."""


def _provider_json(row: Provider) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "label": row.label,
        "base_url": row.base_url,
        "messages_path": row.messages_path,
        "responses_path": row.responses_path,
        "chat_completions_path": row.chat_completions_path,
        "auth_scheme": row.auth_scheme,
        "enabled": row.enabled,
    }


def _model_json(row: Model, provider_name: str) -> dict:
    return {
        "id": row.id,
        "model": row.deployment_name,
        "provider": provider_name,
        "context_window": row.context_window,
        "rotation_threshold_percent": row.rotation_threshold_percent,
        "harnesses": row.harnesses,
        "effort_levels": row.effort_levels,
        "accepts_images": row.accepts_images,
        "enabled": row.enabled,
        "admin_only": row.admin_only,
    }


async def _by_name(session: AsyncSession, column, name: str):
    return (
        await session.execute(select(column.class_).where(column == name.strip()))
    ).scalar_one_or_none()


async def add_provider(
    session: AsyncSession,
    *,
    name: str,
    kind: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """A provider of `kind` named `name`, with a static credential of the same
    name holding `api_key`. An existing provider is returned untouched."""
    if kind not in PROVIDER_KINDS:
        raise Refused(f"kind must be one of {', '.join(PROVIDER_KINDS)}, got {kind!r}")
    existing = await _by_name(session, Provider.name, name)
    if existing is not None:
        return {"created": False, "provider": _provider_json(existing)}

    preset = PROVIDER_KINDS[kind]
    url = (base_url or "").strip() or preset["base_url"]
    if not url:
        raise Refused(f"a {kind} provider needs --base-url")
    key = (api_key or "").strip()
    if not key:
        if preset["key_required"]:
            raise Refused(f"a {kind} provider needs --api-key")
        key = OLLAMA_PLACEHOLDER_KEY

    # A credential left by a run that stopped before its provider was written
    # is that run's, and is reused rather than refused as a name clash.
    credential = await _by_name(session, Credential.name, name)
    if credential is None:
        credential = await insert_credential(session, name=name, kind="static", secret=key)
    row = await insert_provider(
        session,
        name=name,
        label=preset["label"],
        base_url=url,
        messages_path=preset.get("messages_path") or "",
        responses_path=preset.get("responses_path") or "",
        chat_completions_path=preset.get("chat_completions_path") or "",
        credential_id=str(credential.id),
        auth_scheme=preset["auth_scheme"],
    )
    return {"created": True, "provider": _provider_json(row)}


async def add_model(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    context_window: int,
    rotation_threshold_percent: int = DEFAULT_ROTATION_THRESHOLD_PERCENT,
    label: str = "",
    effort_levels: str = "",
    accepts_images: str = "",
    harnesses: str = "",
) -> dict:
    """Model `model` hosted by provider `provider`. An existing model of that
    name is returned untouched, whichever provider it is on."""
    existing = await _by_name(session, Model.deployment_name, model)
    if existing is not None:
        host = (
            await session.execute(select(Provider.name).where(Provider.id == existing.provider_id))
        ).scalar_one_or_none()
        return {"created": False, "model": _model_json(existing, host)}
    host = await _by_name(session, Provider.name, provider)
    if host is None:
        raise Refused(f"no provider named {provider!r}; add it with add-provider first")
    row = await insert_model(
        session,
        deployment_name=model,
        provider_id=str(host.id),
        context_window=str(context_window),
        rotation_threshold_percent=str(rotation_threshold_percent),
        label=label,
        effort_levels=effort_levels,
        accepts_images=accepts_images,
        harnesses=harnesses,
    )
    return {"created": True, "model": _model_json(row, host.name)}


async def mint_key(
    session: AsyncSession,
    *,
    client: str,
    kind: str = "app",
    privileged: bool = False,
    label: str = "default",
) -> dict:
    """A key for client `client`, made if absent, printed once.

    A client that already holds a live key gets nothing new: the key it holds
    is the one in use, and only its hash exists here. A client left without
    one, by a run that stopped between the two writes, is given its first.
    """
    row = await _by_name(session, Client.name, client)
    created = row is None
    if created:
        row = await insert_client(session, name=client, kind=kind, privileged=privileged)
    else:
        live = (
            await session.execute(
                select(ApiKey.id).where(ApiKey.client_id == row.id, ApiKey.revoked.is_(False))
            )
        ).first()
        if live is not None:
            return {
                "client": row.name,
                "created": False,
                "minted": False,
                "key": None,
                "message": f"client {row.name!r} already holds a key; none minted",
            }
    raw_key = await issue_key(session, row, label)
    return {"client": row.name, "created": created, "minted": True, "key": raw_key}


async def list_models(
    session: AsyncSession, *, harness: Optional[str] = None, admin: bool = False
) -> dict:
    """Exactly what `GET /v1/models` answers: an ordinary key unless `admin`,
    which is a privileged one and sees admin-only models too."""
    return await listing_for(session, harness=harness, is_admin=admin)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-provider", help="add an upstream provider and its key")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", required=True, choices=sorted(PROVIDER_KINDS))
    p.add_argument("--base-url", default=None,
                   help="required for ollama; anthropic and openai default to their public APIs")
    p.add_argument("--api-key", default=None, help="required for anthropic and openai")

    m = sub.add_parser("add-model", help="add a model hosted by a provider")
    m.add_argument("--provider", required=True)
    m.add_argument("--model", required=True, help="the provider's own model name")
    m.add_argument("--context-window", required=True, type=int)
    m.add_argument("--rotation-threshold-percent", type=int,
                   default=DEFAULT_ROTATION_THRESHOLD_PERCENT)
    m.add_argument("--label", default="")
    m.add_argument("--effort-levels", default="", help="comma-separated, e.g. low,medium,high")
    m.add_argument("--accepts-images", default="", choices=["", "yes", "no", "unknown"])
    m.add_argument("--harnesses", default="",
                   help="comma-separated harnesses to restrict to; blank is every harness")

    k = sub.add_parser("mint-key", help="create a client and print its first key once")
    k.add_argument("--client", required=True)
    k.add_argument("--kind", default="app", choices=CLIENT_KINDS)
    k.add_argument("--privileged", action="store_true")
    k.add_argument("--label", default="default")

    lm = sub.add_parser("list-models", help="what GET /v1/models answers")
    lm.add_argument("--harness", default=None, choices=sorted(HARNESS_WIRES))
    lm.add_argument("--admin", action="store_true", help="as a privileged key")
    return parser


async def _dispatch(args: argparse.Namespace) -> dict:
    async with SessionLocal() as session:
        if args.command == "add-provider":
            return await add_provider(
                session, name=args.name, kind=args.kind,
                base_url=args.base_url, api_key=args.api_key)
        if args.command == "add-model":
            return await add_model(
                session, provider=args.provider, model=args.model,
                context_window=args.context_window,
                rotation_threshold_percent=args.rotation_threshold_percent,
                label=args.label, effort_levels=args.effort_levels,
                accepts_images=args.accepts_images, harnesses=args.harnesses)
        if args.command == "mint-key":
            return await mint_key(
                session, client=args.client, kind=args.kind,
                privileged=args.privileged, label=args.label)
        return await list_models(session, harness=args.harness, admin=args.admin)


def _detail(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        return (detail.get("error") or {}).get("message") or json.dumps(detail)
    return str(detail)


async def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = await _dispatch(args)
        code = 0
    except Refused as exc:
        result, code = {"error": str(exc)}, 2
    except HTTPException as exc:
        result, code = {"error": _detail(exc), "status": exc.status_code}, 2
    except Exception as exc:  # the database, unreachable or not migrated
        result, code = {"error": f"{type(exc).__name__}: {exc}"}, 1
    finally:
        await dispose_engine()
    print(json.dumps(result, indent=2, default=str))
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
