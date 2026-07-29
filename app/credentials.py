"""Upstream credential resolution.

A `models` row no longer carries its own API key. It points at a `credentials`
row, and this module turns that row into the secret string that
`app.deployments.build_upstream_headers` sends upstream.

Three kinds are supported:

- `static` — the literal key stored in `Credential.secret`.
- `anthropic_oauth` — either a literal long-lived OAuth token stored in
  `Credential.secret` (e.g. a year-long `claude setup-token` grant — returned
  as-is, no refresh machinery), or an OAuth credential file written by the
  Claude CLI (`~/.claude/.credentials.json`) at `source_path`. The file's
  access token is short-lived, so it is refreshed against Anthropic's token
  endpoint when within `REFRESH_SKEW_SECONDS` of expiry and the fresh token
  is cached on the row. A literal `secret` takes precedence over
  `source_path`.
- `openai_oauth` — an OAuth credential file written by the Codex CLI
  (`~/.codex/auth.json`). When the file carries a plain `OPENAI_API_KEY`
  (API-key auth mode) that key is used directly. Otherwise the subscription
  OAuth access token is used — it authenticates the ChatGPT Codex backend
  (`chatgpt.com/backend-api/codex/responses`) alongside the identity headers
  `app.deployments.build_upstream_headers` adds. The token is a short-lived
  JWT; when stale it is refreshed against OpenAI's token endpoint and the
  fresh tokens are written back to the file so the CLI and the proxy keep
  sharing one grant.

Resolution is async and touches the database only when a refresh actually
happens, so the hot path is a dict lookup plus an expiry comparison.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.orm import Credential

# Public client id used by the Claude CLI's OAuth flow. Overridable so a
# different registered client can be used without a code change.
ANTHROPIC_CLIENT_ID = os.getenv(
    "ANTHROPIC_OAUTH_CLIENT_ID", "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
)
ANTHROPIC_TOKEN_URL = os.getenv(
    "ANTHROPIC_OAUTH_TOKEN_URL", "https://console.anthropic.com/v1/oauth/token"
)

# Public client id of the Codex CLI's OAuth app — it is the `client_id` claim
# inside every Codex access token. Overridable like the Anthropic pair.
OPENAI_CLIENT_ID = os.getenv("OPENAI_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
OPENAI_TOKEN_URL = os.getenv("OPENAI_OAUTH_TOKEN_URL", "https://auth.openai.com/oauth/token")

# Refresh this far ahead of the recorded expiry, so a token never goes stale
# mid-flight on a long streaming response.
REFRESH_SKEW_SECONDS = 300

# Serialises refreshes so concurrent requests against the same credential do
# not each burn a round trip (and possibly rotate the refresh token twice).
_refresh_locks: dict[int, asyncio.Lock] = {}


class CredentialError(RuntimeError):
    """Raised when a credential cannot be resolved to a usable secret."""


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _read_json(path: str) -> dict:
    p = Path(path).expanduser()
    if not p.is_file():
        raise CredentialError(f"credential file not found: {p}")
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialError(f"credential file unreadable: {p} ({exc})") from exc


def read_anthropic_oauth_file(path: str) -> tuple[str, str, Optional[datetime]]:
    """Return (access_token, refresh_token, expires_at) from a Claude CLI file."""
    blob = _read_json(path)
    oauth = blob.get("claudeAiOauth") or blob
    access = oauth.get("accessToken")
    refresh = oauth.get("refreshToken")
    if not access or not refresh:
        raise CredentialError(f"no OAuth tokens in {path}")
    expires_at = None
    raw_expiry = oauth.get("expiresAt")
    if raw_expiry:
        # The CLI stores epoch milliseconds.
        expires_at = datetime.fromtimestamp(int(raw_expiry) / 1000, tz=timezone.utc)
        expires_at = expires_at.replace(tzinfo=None)
    return access, refresh, expires_at


def _jwt_claims(token: str) -> dict:
    """Decode a JWT's payload without verifying — we only read our own tokens."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return {}


def _jwt_expiry(token: str) -> Optional[datetime]:
    exp = _jwt_claims(token).get("exp")
    if exp is None:
        return None
    return datetime.fromtimestamp(int(exp), tz=timezone.utc).replace(tzinfo=None)


def codex_account_id(token: str) -> Optional[str]:
    """ChatGPT account id from a Codex access token's auth claim.

    The Codex backend requires it as the `chatgpt-account-id` header; riding
    it inside the token means no extra plumbing between credential resolution
    and header building.
    """
    return (_jwt_claims(token).get("https://api.openai.com/auth") or {}).get(
        "chatgpt_account_id"
    )


def read_codex_oauth_tokens(path: str) -> tuple[str, str, Optional[datetime]]:
    """Return (access_token, refresh_token, expires_at) from a Codex auth file.

    Expiry comes from the access token's own `exp` claim — the file doesn't
    store one.
    """
    blob = _read_json(path)
    tokens = blob.get("tokens") or {}
    access = tokens.get("access_token")
    refresh = tokens.get("refresh_token")
    if not access or not refresh:
        raise CredentialError(f"no OAuth tokens in {path}")
    return access, refresh, _jwt_expiry(access)


def _write_codex_tokens(path: str, access: str, refresh: Optional[str]) -> None:
    """Write refreshed tokens back to the auth file.

    Once minds route through the proxy nothing else refreshes this file, and
    OpenAI may rotate the refresh token on use — writing back keeps the CLI
    login and the proxy on the same grant.
    """
    p = Path(path).expanduser()
    blob = json.loads(p.read_text())
    tokens = blob.setdefault("tokens", {})
    tokens["access_token"] = access
    if refresh:
        tokens["refresh_token"] = refresh
    blob["last_refresh"] = (
        datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    tmp.replace(p)


async def refresh_openai_token(refresh_token: str) -> tuple[str, Optional[str], datetime]:
    """Exchange a Codex refresh token; return (access, new_refresh, expires_at).

    `new_refresh` is None when the endpoint didn't rotate the refresh token.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            OPENAI_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OPENAI_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        raise CredentialError(
            f"OpenAI token refresh failed ({resp.status_code}): {resp.text[:200]}"
        )
    body = resp.json()
    access = body.get("access_token")
    if not access:
        raise CredentialError("OpenAI token refresh returned no access_token")
    expires_at = _jwt_expiry(access) or (
        _now() + timedelta(seconds=int(body.get("expires_in", 3600)))
    )
    return access, body.get("refresh_token"), expires_at


async def refresh_anthropic_token(refresh_token: str) -> tuple[str, datetime]:
    """Exchange a refresh token for a fresh access token and its expiry."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ANTHROPIC_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": ANTHROPIC_CLIENT_ID,
            },
            headers={"Content-Type": "application/json"},
        )
    if resp.status_code != 200:
        raise CredentialError(
            f"Anthropic token refresh failed ({resp.status_code}): {resp.text[:200]}"
        )
    body = resp.json()
    access = body.get("access_token")
    if not access:
        raise CredentialError("Anthropic token refresh returned no access_token")
    expires_in = int(body.get("expires_in", 3600))
    return access, _now() + timedelta(seconds=expires_in)


def _is_fresh(cred: Credential) -> bool:
    if not cred.access_token or cred.expires_at is None:
        return False
    return cred.expires_at - timedelta(seconds=REFRESH_SKEW_SECONDS) > _now()


async def resolve_secret(session: AsyncSession, cred: Optional[Credential]) -> str:
    """Return the secret to send upstream for `cred`.

    Refreshes and persists an OAuth access token when the cached one is missing
    or close to expiry. Raises `CredentialError` when no usable secret exists.
    """
    if cred is None:
        raise CredentialError("no credential attached")

    if cred.kind == "static":
        if not cred.secret:
            raise CredentialError(f"credential {cred.name!r} has no secret set")
        return cred.secret

    if cred.kind == "openai_oauth":
        return await _resolve_openai_oauth(session, cred)

    if cred.kind != "anthropic_oauth":
        raise CredentialError(f"unknown credential kind: {cred.kind!r}")

    if cred.secret:
        # A literal long-lived token stored on the row (e.g. a year-long
        # `claude setup-token` grant). Sent as-is; no refresh, no expiry
        # bookkeeping. Takes precedence over `source_path`.
        return cred.secret

    if _is_fresh(cred):
        return cred.access_token

    if not cred.source_path:
        raise CredentialError(f"credential {cred.name!r} has no secret and no source_path")

    lock = _refresh_locks.setdefault(cred.id, asyncio.Lock())
    async with lock:
        # Another waiter may have refreshed while we queued.
        await session.refresh(cred)
        if _is_fresh(cred):
            return cred.access_token

        file_access, refresh_token, file_expiry = read_anthropic_oauth_file(cred.source_path)
        if file_expiry and file_expiry - timedelta(seconds=REFRESH_SKEW_SECONDS) > _now():
            # The CLI has already refreshed on disk; adopt its token rather
            # than spending a refresh of our own.
            access, expires_at = file_access, file_expiry
        else:
            access, expires_at = await refresh_anthropic_token(refresh_token)

        cred.access_token = access
        cred.expires_at = expires_at
        cred.refreshed_at = _now()
        await session.commit()
        return access


async def _resolve_openai_oauth(session: AsyncSession, cred: Credential) -> str:
    """Resolve a Codex auth file to a secret.

    A plain `OPENAI_API_KEY` in the file wins (API-key auth mode). Otherwise
    the subscription OAuth access token is served, refreshed when stale —
    mirroring the Anthropic flow, plus a write-back of rotated tokens to the
    file (see `_write_codex_tokens`).
    """
    if not cred.source_path:
        raise CredentialError(f"credential {cred.name!r} has no source_path")

    # Cached-token check first, mirroring the Anthropic flow: a fresh cached
    # token must survive the file being briefly unreadable. API-key-mode files
    # never populate the cache columns, so this can't shadow a plain key.
    if _is_fresh(cred):
        return cred.access_token

    api_key = _read_json(cred.source_path).get("OPENAI_API_KEY")
    if api_key:
        return api_key

    lock = _refresh_locks.setdefault(cred.id, asyncio.Lock())
    async with lock:
        # Another waiter may have refreshed while we queued.
        await session.refresh(cred)
        if _is_fresh(cred):
            return cred.access_token

        file_access, refresh_token, file_expiry = read_codex_oauth_tokens(cred.source_path)
        if file_expiry and file_expiry - timedelta(seconds=REFRESH_SKEW_SECONDS) > _now():
            # The CLI has already refreshed on disk; adopt its token.
            access, expires_at = file_access, file_expiry
        else:
            access, new_refresh, expires_at = await refresh_openai_token(refresh_token)
            _write_codex_tokens(cred.source_path, access, new_refresh)

        cred.access_token = access
        cred.expires_at = expires_at
        cred.refreshed_at = _now()
        await session.commit()
        return access
