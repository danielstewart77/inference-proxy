"""Authentication: proxy API key validation + admin console session auth.

Two entirely separate principals:

- **Clients** (apps and agents) present an issued proxy key on every request.
  Validation is a sync, lock-free hash lookup against an in-memory cache.
- **The admin** signs into the HTML console with the username and password held
  in environment config. There is no user table and no self-service portal.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.orm import ApiKey, Client

KEY_PREFIX = "hmp-"


def generate_proxy_key() -> str:
    """Mint a new proxy key. Only the hash is ever stored."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


# ---------- Proxy API key validation (DB-backed with in-memory cache) ----------
#
# Hot path is sync and lock-free. Two module-level structures rebuilt
# atomically by refresh_key_cache():
#   _DB_KEY_HASHES: frozenset[str] for fast membership checks (validate_api_key)
#   _DB_KEY_LOOKUP: dict[str, (client_id, key_id, privileged)] for principal
#     resolution (usage_log attribution and admin-only model gating)
# Both are replaced atomically by reference swap (safe under the GIL).

_DB_KEY_HASHES: frozenset[str] = frozenset()
_DB_KEY_LOOKUP: dict[str, tuple[int, int, bool]] = {}


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def refresh_key_cache(session: AsyncSession) -> int:
    """Rebuild the in-memory key cache from the keys table.

    Excludes revoked keys and keys whose owning client is disabled. Returns
    cache size.
    """
    global _DB_KEY_HASHES, _DB_KEY_LOOKUP
    rows = (
        await session.execute(
            select(ApiKey.key_hash, ApiKey.client_id, ApiKey.id, Client.privileged)
            .join(Client, ApiKey.client_id == Client.id)
            .where(Client.disabled.is_(False))
            .where(ApiKey.revoked.is_(False))
        )
    ).all()
    new_lookup = {row[0]: (row[1], row[2], bool(row[3])) for row in rows}
    _DB_KEY_LOOKUP = new_lookup
    _DB_KEY_HASHES = frozenset(new_lookup.keys())
    return len(_DB_KEY_HASHES)


def proxy_key_count() -> int:
    return len(_DB_KEY_HASHES)


def _raw_key_from_headers(
    authorization: str | None,
    api_key_header: str | None,
) -> Optional[str]:
    """Extract the presented credential from either header.

    `api-key: <key>` takes precedence; otherwise an `Authorization: Bearer
    <key>` value. Returns None when neither yields a usable token.
    """
    if api_key_header:
        return api_key_header
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def validate_api_key(
    authorization: str | None = None,
    api_key_header: str | None = None,
) -> bool:
    """Accept either Authorization: Bearer <key> or api-key: <key> header.

    The presented credential must hash to an issued proxy key. There is no
    prefix shortcut — an upstream provider key or OAuth token presented by a
    caller is not recognized and gets a 401 like any other unknown token.
    """
    raw_key = _raw_key_from_headers(authorization, api_key_header)
    if not raw_key:
        return False
    return _hash_key(raw_key) in _DB_KEY_HASHES


def resolve_principal(
    authorization: str | None = None,
    api_key_header: str | None = None,
) -> Optional[tuple[int, int]]:
    """Look up (client_id, key_id) for the authenticated request.

    Returns None when the credential does not match an issued proxy key.
    """
    raw_key = _raw_key_from_headers(authorization, api_key_header)
    if not raw_key:
        return None
    entry = _DB_KEY_LOOKUP.get(_hash_key(raw_key))
    return (entry[0], entry[1]) if entry else None


def resolve_requester_role(
    authorization: str | None = None,
    api_key_header: str | None = None,
) -> Optional[str]:
    """Return 'admin' for a privileged client, 'client' otherwise, or None.

    None means the credential matched no issued key — callers treat that as
    unprivileged (fail safe).
    """
    raw_key = _raw_key_from_headers(authorization, api_key_header)
    if not raw_key:
        return None
    entry = _DB_KEY_LOOKUP.get(_hash_key(raw_key))
    if entry is None:
        return None
    return "admin" if entry[2] else "client"


# ---------- Admin console session ----------

SESSION_ADMIN_KEY = "admin"


def verify_admin_credentials(username: str, password: str) -> bool:
    """Constant-time check against the configured admin identity.

    Returns False when either config value is unset, so an unconfigured deploy
    cannot be signed into at all.
    """
    if not config.admin_username or not config.admin_password:
        return False
    ok_user = hmac.compare_digest(username, config.admin_username)
    ok_pass = hmac.compare_digest(password, config.admin_password)
    return ok_user and ok_pass


def login_session(request: Request) -> None:
    request.session[SESSION_ADMIN_KEY] = True


def logout_session(request: Request) -> None:
    request.session.pop(SESSION_ADMIN_KEY, None)


def is_admin_session(request: Request) -> bool:
    return bool(request.session.get(SESSION_ADMIN_KEY))


async def require_html_admin(request: Request) -> None:
    """Guard for admin console HTML routes. Raises a 303 redirect to /login."""
    if not is_admin_session(request):
        raise _redirect("/login")


def _redirect(location: str) -> HTTPException:
    """HTTPException that FastAPI's exception handler turns into a 303 redirect.

    We register a handler in app.main that converts this exception shape into
    an actual RedirectResponse.
    """
    return HTTPException(
        status_code=status.HTTP_303_SEE_OTHER,
        detail={"redirect": location},
        headers={"Location": location},
    )


def is_redirect_exception(exc: HTTPException) -> Optional[str]:
    if exc.status_code == status.HTTP_303_SEE_OTHER and exc.headers:
        return exc.headers.get("Location")
    return None
