"""`python -m app.cli` — the rows an install needs, without the console.

A fresh database has no provider, model or client, and the console that makes
them is a cookie-authenticated HTML form. An installer has neither a browser
nor a reason to log in, so these commands make the same rows through the same
functions the console's forms call (`insert_*`, `issue_key`) — validation and
all — and print what they did as JSON on stdout.

Every command is safe to repeat. A name already taken is reused and reported
with `"created": false`, never changed: a key rotated or a URL corrected in the
console since the first run stays as the console left it. `add-provider
--update` and `add-model --update` are the exception, for a rerun correcting
its own mistake: the existing row is rewritten through the same functions the
console's edit forms call (`apply_*_update`), and reported `"updated": true`.

A provider's key never has to be on the command line: `--api-key -` reads it
from stdin and `--api-key-env NAME` from the environment.

`check-model` sends one small completion through the proxy's own forwarding
code (`forward_chat_completion`) and reports the upstream's status, so a wrong
key or address is found before the proxy is called ready.

Exit status: 0 on success, 2 when the input is refused or a checked model did
not answer (the JSON carries the reason, and the status where there is one), 1
when the database cannot be reached or holds no proxy schema.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from typing import Any, Optional, Sequence

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.admin.clients import VALID_KINDS as CLIENT_KINDS, insert_client, issue_key
from app.admin.credentials import apply_credential_update, insert_credential
from app.admin.models import apply_model_update, insert_model
from app.admin.providers import apply_provider_update, insert_provider
from app.azure import close_shared_client
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


#: What `check-model` asks: as small a turn as any model answers. No
#: `max_tokens`, which OpenAI's reasoning models refuse on the chat wire; the
#: Anthropic translation supplies its own.
CHECK_PROMPT = "Reply with the single word: ok"

#: The harnesses whose requests arrive on the chat wire `check-model` uses.
CHECK_HARNESSES = sorted(h for h, wires in HARNESS_WIRES.items() if "chat_completions" in wires)


class Refused(Exception):
    """Input the command will not act on. `extra` rides along in the JSON."""

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.extra = extra


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
    update: bool = False,
) -> dict:
    """A provider of `kind` named `name`, with a static credential of the same
    name holding `api_key`. An existing provider is returned untouched, unless
    `update`: then its address, auth scheme and paths become what `kind` and
    `base_url` say, and its credential takes `api_key` when one is given."""
    if kind not in PROVIDER_KINDS:
        raise Refused(f"kind must be one of {', '.join(PROVIDER_KINDS)}, got {kind!r}")
    preset = PROVIDER_KINDS[kind]
    url = (base_url or "").strip() or preset["base_url"]
    key = (api_key or "").strip()
    existing = await _by_name(session, Provider.name, name)
    if existing is not None:
        if not update:
            return {"created": False, "updated": False, "provider": _provider_json(existing)}
        if not url:
            raise Refused(f"a {kind} provider needs --base-url")
        return await _update_provider(session, existing, preset=preset, url=url, key=key)

    if not url:
        raise Refused(f"a {kind} provider needs --base-url")
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
    return {"created": True, "updated": False, "provider": _provider_json(row)}


async def _update_provider(
    session: AsyncSession, row: Provider, *, preset: dict, url: str, key: str
) -> dict:
    """Rewrite an existing provider as `preset` and `url` describe it. A given
    key replaces its credential's secret; without one the stored secret stays,
    so a rerun that was not given the key again keeps the working one."""
    credential_id = row.credential_id
    if key:
        credential = None
        if credential_id is not None:
            credential = (
                await session.execute(select(Credential).where(Credential.id == credential_id))
            ).scalar_one_or_none()
        if credential is None:
            credential = await _by_name(session, Credential.name, row.name)
        if credential is None:
            credential = await insert_credential(
                session, name=row.name, kind="static", secret=key)
        else:
            await apply_credential_update(
                session, credential, name=credential.name, kind="static", secret=key)
        credential_id = credential.id
    elif credential_id is None:
        if preset["key_required"]:
            raise Refused("this provider holds no key; give one with --api-key")
        credential = await _by_name(session, Credential.name, row.name) or await insert_credential(
            session, name=row.name, kind="static", secret=OLLAMA_PLACEHOLDER_KEY)
        credential_id = credential.id
    await apply_provider_update(
        session,
        row,
        label=row.label or "",
        base_url=url,
        messages_path=preset.get("messages_path") or "",
        responses_path=preset.get("responses_path") or "",
        chat_completions_path=preset.get("chat_completions_path") or "",
        credential_id=str(credential_id),
        auth_scheme=preset["auth_scheme"],
        api_version=row.api_version or "",
        enabled=row.enabled,
    )
    return {"created": False, "updated": True, "provider": _provider_json(row)}


async def add_model(
    session: AsyncSession,
    *,
    provider: str,
    model: str,
    context_window: int,
    rotation_threshold_percent: Optional[int] = None,
    label: Optional[str] = None,
    effort_levels: Optional[str] = None,
    accepts_images: Optional[str] = None,
    harnesses: Optional[str] = None,
    update: bool = False,
) -> dict:
    """Model `model` hosted by provider `provider`. An existing model of that
    name is returned untouched, whichever provider it is on, unless `update`:
    then it moves to `provider` and takes `context_window` and whichever of
    the other values were given; those not given keep their stored values."""
    existing = await _by_name(session, Model.deployment_name, model)
    if existing is not None and not update:
        held_by = (
            await session.execute(select(Provider.name).where(Provider.id == existing.provider_id))
        ).scalar_one_or_none()
        return {"created": False, "updated": False, "model": _model_json(existing, held_by)}
    host = await _by_name(session, Provider.name, provider)
    if host is None:
        raise Refused(f"no provider named {provider!r}; add it with add-provider first")
    if existing is not None:
        await apply_model_update(
            session,
            existing,
            provider_id=str(host.id),
            context_window=str(context_window),
            rotation_threshold_percent=_given(
                rotation_threshold_percent, existing.rotation_threshold_percent),
            label=_given(label, existing.label),
            effort_levels=_given(effort_levels, existing.effort_levels),
            accepts_images=_given(
                accepts_images,
                {True: "yes", False: "no"}.get(existing.accepts_images, "")),
            harnesses=_given(harnesses, existing.harnesses),
            api_version=_given(None, existing.api_version),
            description=_given(None, existing.description),
            cost_per_million_input=_given(None, existing.cost_per_million_input),
            cost_per_million_output=_given(None, existing.cost_per_million_output),
            cost_per_million_cache_write=_given(None, existing.cost_per_million_cache_write),
            cost_per_million_cache_read=_given(None, existing.cost_per_million_cache_read),
            enabled=existing.enabled,
            admin_only=existing.admin_only,
        )
        return {"created": False, "updated": True, "model": _model_json(existing, host.name)}
    row = await insert_model(
        session,
        deployment_name=model,
        provider_id=str(host.id),
        context_window=str(context_window),
        rotation_threshold_percent=str(
            DEFAULT_ROTATION_THRESHOLD_PERCENT
            if rotation_threshold_percent is None else rotation_threshold_percent),
        label=label or "",
        effort_levels=effort_levels or "",
        accepts_images=accepts_images or "",
        harnesses=harnesses or "",
    )
    return {"created": True, "updated": False, "model": _model_json(row, host.name)}


def _given(value: Any, stored: Any) -> str:
    """`value` if the command was given one, else the stored value, both as
    the edit form would submit them."""
    chosen = stored if value is None else value
    return "" if chosen is None else str(chosen)


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


async def check_model(session: AsyncSession, *, model: str, harness: str = "dsh") -> dict:
    """One small completion for `model`, sent the way a `harness` request on
    the chat wire is: through `forward_chat_completion`, so it resolves,
    authenticates and translates exactly as served traffic does. Unmetered and
    as a privileged caller, so an admin-only model can be checked too.

    Answers the upstream's 200; anything else is refused with its status and
    the upstream's own words."""
    from app.proxy.chat_completions import forward_chat_completion

    body = {
        "model": model,
        "messages": [{"role": "user", "content": CHECK_PROMPT}],
        "stream": False,
    }
    answer = await forward_chat_completion(
        session, body, model=model, harness=harness, is_admin=True)
    payload: Any = None
    try:
        payload = json.loads(answer.body)
    except (ValueError, AttributeError):
        pass
    if answer.status_code != 200:
        message = None
        if isinstance(payload, dict):
            error = payload.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            message = message or payload.get("detail")
        raise Refused(
            f"model {model!r}: the upstream answered {answer.status_code}"
            + (f": {message}" if message else ""),
            model=model, status=answer.status_code, upstream=payload,
        )
    reply = None
    if isinstance(payload, dict):
        choices = payload.get("choices") or [{}]
        reply = ((choices[0] or {}).get("message") or {}).get("content")
    return {"model": model, "ok": True, "status": 200, "reply": reply}


def _api_key(args: argparse.Namespace) -> Optional[str]:
    """The provider key, from wherever the command was told to take it.
    `--api-key -` is stdin; `--api-key-env NAME` is that variable."""
    if getattr(args, "api_key_env", None):
        name = args.api_key_env
        if name not in os.environ:
            raise Refused(f"--api-key-env {name}: no such environment variable")
        return os.environ[name]
    if args.api_key == "-":
        return sys.stdin.read()
    return args.api_key


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add-provider", help="add an upstream provider and its key")
    p.add_argument("--name", required=True)
    p.add_argument("--kind", required=True, choices=sorted(PROVIDER_KINDS))
    p.add_argument("--base-url", default=None,
                   help="required for ollama; anthropic and openai default to their public APIs")
    key = p.add_mutually_exclusive_group()
    key.add_argument("--api-key", default=None,
                     help="required for anthropic and openai; '-' reads it from stdin")
    key.add_argument("--api-key-env", default=None, metavar="NAME",
                     help="read the key from environment variable NAME")
    p.add_argument("--update", action="store_true",
                   help="rewrite an existing provider's address and, if given, its key")

    m = sub.add_parser("add-model", help="add a model hosted by a provider")
    m.add_argument("--provider", required=True)
    m.add_argument("--model", required=True, help="the provider's own model name")
    m.add_argument("--context-window", required=True, type=int)
    m.add_argument("--rotation-threshold-percent", type=int, default=None,
                   help=f"default {DEFAULT_ROTATION_THRESHOLD_PERCENT} for a new model")
    m.add_argument("--label", default=None)
    m.add_argument("--effort-levels", default=None, help="comma-separated, e.g. low,medium,high")
    m.add_argument("--accepts-images", default=None, choices=["", "yes", "no", "unknown"])
    m.add_argument("--harnesses", default=None,
                   help="comma-separated harnesses to restrict to; blank is every harness")
    m.add_argument("--update", action="store_true",
                   help="rewrite an existing model: its provider, context window and "
                        "whichever other values are given")

    k = sub.add_parser("mint-key", help="create a client and print its first key once")
    k.add_argument("--client", required=True)
    k.add_argument("--kind", default="app", choices=CLIENT_KINDS)
    k.add_argument("--privileged", action="store_true")
    k.add_argument("--label", default="default")

    lm = sub.add_parser("list-models", help="what GET /v1/models answers")
    lm.add_argument("--harness", default=None, choices=sorted(HARNESS_WIRES))
    lm.add_argument("--admin", action="store_true", help="as a privileged key")

    c = sub.add_parser("check-model", help="send one small completion and report the upstream status")
    c.add_argument("--model", required=True)
    c.add_argument("--harness", default="dsh", choices=CHECK_HARNESSES,
                   help="judge the request as this harness's (default dsh)")
    return parser


async def _dispatch(args: argparse.Namespace) -> dict:
    async with SessionLocal() as session:
        if args.command == "add-provider":
            return await add_provider(
                session, name=args.name, kind=args.kind,
                base_url=args.base_url, api_key=_api_key(args), update=args.update)
        if args.command == "add-model":
            return await add_model(
                session, provider=args.provider, model=args.model,
                context_window=args.context_window,
                rotation_threshold_percent=args.rotation_threshold_percent,
                label=args.label, effort_levels=args.effort_levels,
                accepts_images=args.accepts_images, harnesses=args.harnesses,
                update=args.update)
        if args.command == "mint-key":
            return await mint_key(
                session, client=args.client, kind=args.kind,
                privileged=args.privileged, label=args.label)
        if args.command == "check-model":
            return await check_model(session, model=args.model, harness=args.harness)
        return await list_models(session, harness=args.harness, admin=args.admin)


def _detail(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict):
        return (detail.get("error") or {}).get("message") or json.dumps(detail)
    return str(detail)


async def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # stdout is the one JSON object. The forwarding code `check-model`
        # runs prints its request log there, so for the command it goes to
        # stderr.
        with contextlib.redirect_stdout(sys.stderr):
            result = await _dispatch(args)
        code = 0
    except Refused as exc:
        result, code = {"error": str(exc), **exc.extra}, 2
    except HTTPException as exc:
        result, code = {"error": _detail(exc), "status": exc.status_code}, 2
    except Exception as exc:  # the database, unreachable or not migrated
        result, code = {"error": f"{type(exc).__name__}: {exc}"}, 1
    finally:
        await close_shared_client()
        await dispose_engine()
    print(json.dumps(result, indent=2, default=str))
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())
