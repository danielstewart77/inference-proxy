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

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
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


async def _client_for(session: AsyncSession, holder: str, *, lock: bool = False) -> Optional[Client]:
    query = select(Client).where(func.lower(Client.name) == holder)
    if lock:
        # Two writes for one holder in turn, never interleaved: both would
        # read the same live keys and each leave its own live.
        query = query.with_for_update()
    rows = (await session.execute(query)).scalars().all()
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
    if len(token) < MIN_TOKEN_LENGTH or any(ch.isspace() or not ch.isprintable() for ch in token):
        raise HTTPException(
            status_code=400,
            detail=f"a key is at least {MIN_TOKEN_LENGTH} printable characters, no spaces",
        )
    digest = _hash_key(token)
    kind = KIND_BY_ROLE[role]

    client = await _client_for(session, name, lock=True)
    created = client is None
    if created:
        client = Client(name=name, kind=kind, disabled=False, privileged=False)
        session.add(client)
        try:
            await session.flush()
        except IntegrityError:
            await session.rollback()
            raise HTTPException(
                status_code=409, detail=f"{name!r} was created by another request; try again"
            )
    else:
        _refuse_to_take_over(client, kind)

    same = None
    owner = (
        await session.execute(select(ApiKey).where(ApiKey.key_hash == digest))
    ).scalar_one_or_none()
    if owner is not None:
        if owner.client_id != client.id:
            raise HTTPException(status_code=409, detail="that value is already another holder's key")
        # The console putting back the value it just replaced, after the
        # write of the new one failed: that key is reinstated, not refused.
        same = owner

    live = (
        await session.execute(
            select(ApiKey).where(ApiKey.client_id == client.id, ApiKey.revoked.is_(False))
        )
    ).scalars().all()
    replaced = 0
    for key in live:
        if key is not same:
            key.revoked = True
            replaced += 1
    if same is not None:
        same.revoked = False
    else:
        session.add(ApiKey(client_id=client.id, label="console", key_hash=digest, revoked=False))
    await session.commit()
    await _refresh_after_commit(session)
    return {"holder": name, "client": client.name, "created": created, "replaced": replaced}


def _refuse_to_take_over(client: Client, kind: str) -> None:
    """A holder's PUT reaches only a client that is already that holder.

    Without this a mind created under the name of an app (`frame-main`) or
    of the console itself would revoke that app's key and inherit its
    client, its usage and, for a privileged one, its admin-only models.
    """
    if client.disabled:
        raise HTTPException(status_code=409, detail=f"client {client.name!r} is disabled")
    if client.name.strip().lower() in config.credential_admin_clients:
        raise HTTPException(
            status_code=409, detail=f"{client.name!r} manages keys and is not managed here"
        )
    if client.privileged:
        raise HTTPException(
            status_code=409,
            detail=f"{client.name!r} is privileged; its key is changed on the proxy's own page",
        )
    if client.kind != kind:
        raise HTTPException(
            status_code=409,
            detail=f"{client.name!r} is a {client.kind} client, not a {kind}",
        )


async def _refresh_after_commit(session: AsyncSession) -> None:
    """Load the change into the key cache now rather than on the next tick:
    the console probes a value the moment it is written. A refresh that
    fails is not a failed change - the commit stands and the refresh loop
    picks it up within seconds - so it must not answer as a refusal, which
    the console reads as "nothing changed" and does not write the file."""
    try:
        await refresh_key_cache(session)
    except Exception as exc:  # noqa: BLE001
        print(f"[holder_keys] key cache refresh after commit failed: {type(exc).__name__}", flush=True)


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
    client = await _client_for(session, name, lock=True)
    revoked = 0
    if client is not None and client.name.strip().lower() in config.credential_admin_clients:
        # The key every one of these routes is called with: revoking it
        # locks the console out of the proxy until someone edits it by hand.
        raise HTTPException(
            status_code=409, detail=f"{client.name!r} manages keys and is not revoked here"
        )
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
        await _refresh_after_commit(session)
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


@router.get("/credentials/usage")
async def holder_usage(
    days: int = Query(30, ge=1, le=365),
    caller: str = Depends(require_credential_admin),
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """What each holder used over the last `days` days: requests, tokens and
    estimated cost, priced as `/admin/usage` prices them.

    The console totals these per person - each holder is a mind, a bot or a
    service, and the hive knows whose mind is whose (story 2.1.17) - so this
    answers per holder and knows nothing of people. Every client is listed,
    a quiet one at zero, so a mind that used nothing is not mistaken for one
    the proxy has never heard of.
    """
    del caller
    from app.admin.usage import _cost_expr, _since, _sum_tokens
    from app.orm import Model, UsageLog

    used = {
        row.client_id: row
        for row in (
            await session.execute(
                select(
                    UsageLog.client_id.label("client_id"),
                    func.count(UsageLog.id).label("requests"),
                    _sum_tokens(UsageLog.total_tokens).label("total_tokens"),
                    _cost_expr().label("cost"),
                )
                .outerjoin(Model, UsageLog.model_name == Model.deployment_name)
                .where(UsageLog.created_at >= _since(days))
                .group_by(UsageLog.client_id)
            )
        ).all()
    }
    clients = (await session.execute(select(Client).order_by(Client.name))).scalars().all()
    rows = []
    for client in clients:
        row = used.get(client.id)
        rows.append(
            {
                "holder": client.name.strip().lower(),
                "role": ROLE_BY_KIND.get(client.kind, "service"),
                "requests": int(row.requests) if row else 0,
                "total_tokens": int(row.total_tokens) if row else 0,
                "cost": round(float(row.cost or 0), 4) if row else 0.0,
            }
        )
    return rows
