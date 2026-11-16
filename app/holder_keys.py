"""`/credentials` -- one proxy key per holder, minted by the hive console.

The hive's comms gateway and memory store each key credentials per holder and
let the console mint, rotate and revoke them over one small contract:
`PUT /credentials/{holder}` with the value, `DELETE /credentials/{holder}`,
and `GET /credentials` for who holds one. This is the same contract here, so
creating a mind on the console gives it a proxy key of its own the same way
it gives it the other two, and every call it makes is metered to it rather
than to whoever's key it was handed (story 2.1.9).

A holder is a client, matched by name without case: the existing clients
carry display names (`Stash`, `Bob`) and the console speaks lowercase. The
value is chosen by the console, which writes it where the mind reads it; only
its digest is stored, and no route returns one.

The routes answer only to a key whose client is named in
`CREDENTIAL_ADMIN_CLIENTS`. A route that issues keys must not answer to the
keys it issues, so a mind's own key -- privileged or not -- is refused.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import client_id_for_key, _hash_key, refresh_key_cache
from app.config import config
from app.db import get_session
from app.orm import ApiKey, Client

router = APIRouter()

#: What the console names a holder: a mind, a bot, a service.
HOLDER_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

#: The console's role for a holder, as this proxy's client kind.
KIND_BY_ROLE = {"mind": "mind", "service": "app", "person": "human"}
ROLE_BY_KIND = {"mind": "mind", "app": "service", "agent": "service", "human": "person"}

#: A value shorter than this is not a credential anybody should be handed.
MIN_TOKEN_LENGTH = 32


async def require_credential_admin(
    authorization: Optional[str] = Header(default=None),
    api_key: Optional[str] = Header(default=None, alias="api-key"),
    session: AsyncSession = Depends(get_session),
) -> str:
    """The calling client's name, if it is a credential admin; else refused."""
    client_id = client_id_for_key(authorization, api_key)
    if client_id is None:
        raise HTTPException(status_code=401, detail="a proxy key is required")
    name = (
        await session.execute(select(Client.name).where(Client.id == client_id))
    ).scalar_one_or_none()
    if not name or name.strip().lower() not in config.credential_admin_clients:
        raise HTTPException(
            status_code=403, detail="this key may not manage other holders' keys"
        )
    return name


def _holder(raw: str) -> str:
    holder = (raw or "").strip().lower()
    if not HOLDER_RE.fullmatch(holder):
        raise HTTPException(status_code=400, detail=f"not a holder name: {raw!r}")
    return holder


async def _client_for(session: AsyncSession, holder: str) -> Optional[Client]:
    rows = (
        await session.execute(select(Client).where(func.lower(Client.name) == holder))
    ).scalars().all()
    if len(rows) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"more than one client is named {holder!r} without case",
        )
    return rows[0] if rows else None


class HolderKey(BaseModel):
    #: The user this holder answers for. Not stored: the hive holds a mind's
    #: owner, and a per-person total is its sum over the minds they own.
    username: str = ""
    token: str
    role: Optional[str] = None


@router.put("/credentials/{holder}")
async def put_holder_key(
    holder: str,
    body: HolderKey,
    caller: str = Depends(require_credential_admin),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Make `body.token` the holder's one live key, creating its client.

    Every key it held before is revoked in the same transaction: a rotation
    that left the old value working would be a rotation in name only.
    """
    del caller
    name = _holder(holder)
    role = (body.role or "mind").strip().lower()
    if role not in KIND_BY_ROLE:
        raise HTTPException(status_code=400, detail=f"role must be one of {sorted(KIND_BY_ROLE)}")
    token = body.token or ""
    if len(token) < MIN_TOKEN_LENGTH or token != token.strip():
        raise HTTPException(
            status_code=400, detail=f"a key is at least {MIN_TOKEN_LENGTH} characters, no spaces"
        )
    digest = _hash_key(token)
    taken = (
        await session.execute(select(ApiKey.client_id).where(ApiKey.key_hash == digest))
    ).first()
    if taken is not None:
        raise HTTPException(status_code=409, detail="that value is already a key")

    client = await _client_for(session, name)
    created = client is None
    if created:
        client = Client(name=name, kind=KIND_BY_ROLE[role], disabled=False, privileged=False)
        session.add(client)
        await session.flush()
    elif client.disabled:
        raise HTTPException(status_code=409, detail=f"client {client.name!r} is disabled")

    live = (
        await session.execute(
            select(ApiKey).where(ApiKey.client_id == client.id, ApiKey.revoked.is_(False))
        )
    ).scalars().all()
    for key in live:
        key.revoked = True
    session.add(ApiKey(client_id=client.id, label="console", key_hash=digest, revoked=False))
    await session.commit()
    # Now, not on the next refresh tick: the console probes the value the
    # moment it is written, and a mind may start a turn seconds later.
    await refresh_key_cache(session)
    return {"holder": name, "client": client.name, "created": created, "replaced": len(live)}


@router.delete("/credentials/{holder}")
async def delete_holder_key(
    holder: str,
    caller: str = Depends(require_credential_admin),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Revoke this holder's keys and nobody else's. The client and its usage
    history stay."""
    del caller
    name = _holder(holder)
    client = await _client_for(session, name)
    revoked = 0
    if client is not None:
        live = (
            await session.execute(
                select(ApiKey).where(ApiKey.client_id == client.id, ApiKey.revoked.is_(False))
            )
        ).scalars().all()
        for key in live:
            key.revoked = True
        revoked = len(live)
        await session.commit()
        await refresh_key_cache(session)
    return {"holder": name, "revoked": bool(revoked)}


@router.get("/credentials")
async def list_holder_keys(
    caller: str = Depends(require_credential_admin),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """Every client, whether it holds a live key, and never a value."""
    del caller
    clients = (await session.execute(select(Client).order_by(Client.name))).scalars().all()
    live_by_client = dict(
        (
            await session.execute(
                select(ApiKey.client_id, func.count())
                .where(ApiKey.revoked.is_(False))
                .group_by(ApiKey.client_id)
            )
        ).all()
    )
    rows = []
    for client in clients:
        live = bool(live_by_client.get(client.id)) and not client.disabled
        rows.append(
            {
                "holder": client.name.strip().lower(),
                "client": client.name,
                "username": None,
                "role": ROLE_BY_KIND.get(client.kind, "service"),
                "created_at": client.created_at.isoformat() if client.created_at else None,
                "revoked_at": None if live else "revoked",
                "live": live,
            }
        )
    return rows
