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

from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.credentials import CredentialError, codex_account_id, resolve_secret
from app.orm import Model

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


async def resolve_deployment(
    session: AsyncSession,
    name: Optional[str],
    *,
    error_kind: str = "openai",
    is_admin: bool = False,
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
            .options(selectinload(Model.credential))
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
    if not row.target_uri:
        _raise(error_kind, 503, f"Model {name!r} has no upstream URI configured")
    try:
        secret = await resolve_secret(session, row.credential)
    except CredentialError as exc:
        _raise(error_kind, 503, f"Model {name!r} credential unavailable: {exc}")
    return DeploymentTarget(
        name=row.deployment_name,
        target_uri=row.target_uri,
        api_key=secret,
        api_version=row.api_version,
        auth_scheme=row.auth_scheme,
        credential_kind=row.credential.kind if row.credential else "static",
    )


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
