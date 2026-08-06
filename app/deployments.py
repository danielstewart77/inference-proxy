"""Per-deployment routing.

Each row in the `models` table is self-contained — its own target URI, upstream
credential, optional api_version, and auth_scheme. There is no global endpoint:
the proxy looks up a `DeploymentTarget` by name on every request and forwards
there. The secret itself comes from the attached `credentials` row via
`app.credentials.resolve_secret`, which handles OAuth refresh transparently.

The path (or host) of `target_uri` determines the upstream protocol:
  - `/anthropic/v1/messages`, or host `api.anthropic.com` + `/messages`
        → Anthropic Messages
  - `/openai/responses`, or host `api.openai.com` + `/responses`
        → Responses API (Azure or real OpenAI)
  - `/chat/completions`  → OpenAI-shaped Chat Completions
        (Foundry Models-as-a-Service, or real OpenAI)

`auth_scheme` determines how `api_key` is sent upstream — see
`build_upstream_headers`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.credentials import CredentialError, codex_account_id, resolve_secret
from app.orm import Model, Provider

REAL_ANTHROPIC_HOST = "api.anthropic.com"
REAL_OPENAI_HOST = "api.openai.com"
# The ChatGPT Codex backend — where subscription OAuth tokens are honoured.
# Speaks the Responses API shape, but requires streaming and the identity
# headers `build_upstream_headers` adds.
CHATGPT_CODEX_HOST = "chatgpt.com"


@dataclass(frozen=True)
class DeploymentTarget:
    name: str
    target_uri: str
    api_key: str
    api_version: Optional[str]
    auth_scheme: str = "bearer"
    # Kind of the `credentials` row the secret came from. Routes key off this
    # when a credential type needs more than an auth header — an Anthropic
    # subscription OAuth token, for instance, is only accepted alongside the
    # OAuth beta flag and the CLI identity block.
    credential_kind: str = "static"

    @property
    def is_anthropic_oauth(self) -> bool:
        return self.credential_kind == "anthropic_oauth"

    @property
    def upstream_url(self) -> str:
        """Final URL to POST to: target_uri with api-version appended if set."""
        if not self.api_version:
            return self.target_uri
        sep = "&" if "?" in self.target_uri else "?"
        return f"{self.target_uri}{sep}api-version={self.api_version}"

    @property
    def is_anthropic_messages(self) -> bool:
        if REAL_ANTHROPIC_HOST in self.target_uri:
            return "/messages" in self.target_uri
        if "/anthropic/" in self.target_uri and "/messages" in self.target_uri:
            return True
        # Anthropic-compatible servers on other hosts (e.g. Ollama's
        # /v1/messages endpoint) — the path shape is the protocol signal.
        return self.target_uri.rstrip("/").endswith("/v1/messages")

    @property
    def is_codex_backend(self) -> bool:
        return CHATGPT_CODEX_HOST in self.target_uri

    @property
    def is_openai_responses(self) -> bool:
        if REAL_OPENAI_HOST in self.target_uri or self.is_codex_backend:
            return "/responses" in self.target_uri
        if "/openai/responses" in self.target_uri:
            return True
        # Responses-compatible servers on other hosts (e.g. Ollama's
        # /v1/responses endpoint) — the path shape is the protocol signal.
        return self.target_uri.rstrip("/").endswith("/v1/responses")

    @property
    def is_native_chat_completions(self) -> bool:
        # Foundry Models-as-a-Service surface (DeepSeek, Kimi, gpt-oss, etc.)
        # — `/models/chat/completions` — or real OpenAI's `/v1/chat/completions`.
        # Distinguished from `/openai/responses` / real OpenAI's `/v1/responses`,
        # which route through the Responses API instead.
        return "/chat/completions" in self.target_uri and "/openai/responses" not in self.target_uri


def build_upstream_headers(target: DeploymentTarget) -> dict[str, str]:
    """Build the base auth + content headers for an upstream call.

    Callers may still layer route-specific headers on top (e.g. the Anthropic
    Messages route also sets `anthropic-version` and forwards client
    `anthropic-*` headers) — see `app/proxy/anthropic.py`.
    """
    if target.auth_scheme == "bearer":
        headers = {
            "Authorization": f"Bearer {target.api_key}",
            "Content-Type": "application/json",
        }
        if target.is_codex_backend:
            # The Codex backend wants the caller's ChatGPT identity alongside
            # the bearer token — same idea as Anthropic's OAuth beta flag +
            # identity block. The account id lives inside the JWT itself.
            account = codex_account_id(target.api_key)
            if account:
                headers["chatgpt-account-id"] = account
            headers["OpenAI-Beta"] = "responses=experimental"
            headers["originator"] = "codex_cli_rs"
        return headers
    if target.auth_scheme == "x_api_key":
        return {
            "x-api-key": target.api_key,
            "Content-Type": "application/json",
        }
    if target.auth_scheme == "api_key_header":
        return {"api-key": target.api_key, "Content-Type": "application/json"}
    raise ValueError(f"Unknown auth_scheme: {target.auth_scheme!r}")


def _raise(error_kind: str, status: int, message: str) -> None:
    if error_kind == "anthropic":
        raise HTTPException(
            status_code=status,
            detail={
                "type": "error",
                "error": {
                    "type": "invalid_request_error" if status < 500 else "api_error",
                    "message": message,
                },
            },
        )
    raise HTTPException(status_code=status, detail=message)


#: Which harness speaks each request shape. The endpoint a request arrives on
#: is therefore enough to know who is asking — no caller has to declare it.
HARNESS_FOR_WIRE = {
    "anthropic_messages": "claude",
    "openai_responses": "codex",
    "chat_completions": "codex",
}


def uri_serves(target_uri: Optional[str], wire: str) -> bool:
    """Whether a stored URI answers requests of shape `wire`.

    Only needed for a model registered before providers existed, whose single
    baked-in path is the only shape it can serve.
    """
    if not target_uri:
        return False
    probe = DeploymentTarget(name="", target_uri=target_uri, api_key="", api_version=None)
    if wire == "anthropic_messages":
        return probe.is_anthropic_messages
    if wire == "openai_responses":
        return probe.is_openai_responses
    if wire == "chat_completions":
        return probe.is_openai_responses or probe.is_native_chat_completions
    return False


def resolve_target_uri(row: Model, wire: Optional[str]) -> Optional[str]:
    """Where a request in shape `wire` for this model should be sent.

    A model attached to a provider has no path of its own: the provider holds
    the host and one path per shape it serves, and the shape is whichever
    endpoint the request came in on. That is what lets one locally-hosted model
    answer a Claude harness and a Codex harness both. A model with no provider
    falls back to its own stored URI, which is the pre-provider arrangement and
    serves exactly one shape.

    Disabling a provider takes its models with it. Falling back to the URI a
    migrated row still carries would send traffic to the upstream that was
    just switched off, and — because that stale URI serves one shape while a
    provider-attached row is listed for every shape the provider serves — would
    offer the model to a harness whose requests it cannot answer.
    """
    provider = row.provider
    if provider is None:
        return row.target_uri
    if not provider.enabled:
        return None

    path = provider.path_for(wire) if wire else None
    if path is None and wire is not None:
        # `chat_completions` is the one shape with a stand-in: the Responses
        # API answers it after translation, which is what the chat route
        # already does for the Codex backend.
        if wire == "chat_completions":
            path = provider.responses_path
        if path is None:
            return None
    if path is None:
        # No shape asked for (a legacy caller): first path the provider serves.
        path = (
            provider.messages_path
            or provider.responses_path
            or provider.chat_completions_path
        )
    if path is None:
        return None
    return f"{provider.base_url.rstrip('/')}/{path.lstrip('/')}"


async def resolve_deployment(
    session: AsyncSession,
    name: Optional[str],
    *,
    error_kind: str = "openai",
    is_admin: bool = False,
    wire: Optional[str] = None,
) -> DeploymentTarget:
    """Look up a deployment by name; return a fully-populated target.

    Raises HTTPException on missing/unknown/disabled/incomplete rows. The
    `error_kind` selects the response shape — `'openai'` (raw detail string)
    or `'anthropic'` (Anthropic-shaped error envelope).

    `is_admin` gates admin-only models. It defaults to False so any caller that
    forgets to pass it fails closed (over-restricts) rather than leaking. An
    admin-only row hit by a non-admin is reported as 404 — indistinguishable
    from a model that does not exist, so restricted models aren't enumerable.
    """
    if not name:
        _raise(error_kind, 400, "Missing required 'model' field")
    row = (
        await session.execute(
            select(Model)
            .options(
                selectinload(Model.credential),
                selectinload(Model.provider).selectinload(Provider.credential),
            )
            .where(
                Model.deployment_name == name,
                Model.enabled.is_(True),
            )
        )
    ).scalar_one_or_none()
    if row is None:
        _raise(error_kind, 404, f"Model {name!r} is not registered")
    if row.admin_only and not is_admin:
        # Same shape/status as "not registered" — don't reveal it exists.
        _raise(error_kind, 404, f"Model {name!r} is not registered")
    harness = HARNESS_FOR_WIRE.get(wire or "")
    if harness and not row.visible_to(harness):
        # Withheld from this harness — reported exactly as absent, for the
        # same reason an admin-only row is: a refusal that differs from a
        # miss is an enumeration oracle.
        _raise(error_kind, 404, f"Model {name!r} is not registered")
    target_uri = resolve_target_uri(row, wire)
    if not target_uri:
        _raise(error_kind, 503, f"Model {name!r} has no upstream URI configured")
    provider = row.provider if row.provider is not None and row.provider.enabled else None
    credential = (
        row.credential
        if row.credential is not None or provider is None
        else provider.credential
    )
    try:
        secret = await resolve_secret(session, credential)
    except CredentialError as exc:
        _raise(error_kind, 503, f"Model {name!r} credential unavailable: {exc}")
    return DeploymentTarget(
        name=row.deployment_name,
        target_uri=target_uri,
        api_key=secret,
        api_version=row.api_version or (provider.api_version if provider else None),
        auth_scheme=row.auth_scheme,
        credential_kind=credential.kind if credential else "static",
    )


async def listing_for(
    session: AsyncSession, *, wire: str, is_admin: bool
) -> dict:
    """Every model a harness speaking `wire` may address, with its provider.

    The endpoint a listing is asked on identifies the harness — a Claude CLI
    speaks Anthropic Messages, a Codex CLI the Responses shape — so no caller
    has to declare who it is. A model is listed when its provider serves that
    shape and its own harness list does not withhold it, which is what lets a
    locally-hosted model appear to both harnesses without being registered
    twice.
    """
    harness = HARNESS_FOR_WIRE[wire]
    now = int(time.time())
    stmt = (
        select(Model)
        .options(selectinload(Model.provider))
        .where(Model.enabled.is_(True))
        .order_by(Model.deployment_name)
    )
    if not is_admin:
        stmt = stmt.where(Model.admin_only.is_(False))
    data = []
    for row in (await session.execute(stmt)).scalars().all():
        if not row.visible_to(harness):
            continue
        uri = resolve_target_uri(row, wire)
        if not uri:
            continue
        provider = row.provider
        if provider is None and not uri_serves(uri, wire):
            # Registered before providers existed: its one baked-in path is the
            # only shape it answers.
            continue
        data.append(
            {
                "id": row.deployment_name,
                "object": "model",
                "created": now,
                "owned_by": provider.name if provider else "proxy",
                "provider": provider.name if provider else None,
                "provider_label": (provider.label or provider.name) if provider else None,
                "label": row.label,
                "description": row.description,
            }
        )
    return {"object": "list", "data": data}


def reject_wrong_protocol(
    target: DeploymentTarget,
    *,
    expected: str,
    error_kind: str = "openai",
) -> None:
    """Guard against using a deployment on the wrong client route.

    `expected` is one of: 'anthropic_messages', 'openai_responses',
    'chat_completions_any' (chat route accepts both responses and native chat).
    """
    if expected == "anthropic_messages":
        if not target.is_anthropic_messages:
            _raise(
                error_kind, 400,
                f"Deployment {target.name!r} is not an Anthropic Messages endpoint",
            )
    elif expected == "openai_responses":
        if not target.is_openai_responses:
            _raise(
                error_kind, 400,
                f"Deployment {target.name!r} is not an Azure OpenAI Responses endpoint",
            )
    elif expected == "chat_completions_any":
        if not (target.is_openai_responses or target.is_native_chat_completions):
            _raise(
                error_kind, 400,
                f"Deployment {target.name!r} is not a Chat Completions-compatible endpoint",
            )
    else:
        raise ValueError(f"Unknown expected protocol: {expected}")
