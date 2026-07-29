"""/admin/clients — list, create, edit, enable/disable, delete, and key management.

A client is an app or agent, never a human. It holds any number of proxy keys
so credentials can be rotated without downtime: mint the replacement, move the
caller over, revoke the old one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import (
    _hash_key,
    generate_proxy_key,
    refresh_key_cache,
    require_html_admin,
)
from app.db import get_session
from app.orm import ApiKey, Client
from app.templating import templates

router = APIRouter()

VALID_KINDS = ("app", "agent", "mind", "human")


def _validate_kind(raw: str) -> str:
    val = (raw or "").strip()
    if val not in VALID_KINDS:
        raise HTTPException(
            status_code=400, detail=f"kind must be one of {VALID_KINDS}, got {val!r}"
        )
    return val


def _required_str(raw: str | None, field: str, *, max_len: int) -> str:
    val = (raw or "").strip()
    if not val:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"{field} must be {max_len} characters or fewer"
        )
    return val


def _optional_str(raw: str | None, *, max_len: int) -> str | None:
    val = (raw or "").strip()
    if not val:
        return None
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"value must be {max_len} characters or fewer"
        )
    return val


async def _get_client(session: AsyncSession, client_id: int) -> Client:
    target = (
        await session.execute(select(Client).where(Client.id == client_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Client not found")
    return target


@router.get("/admin/clients", response_class=HTMLResponse)
async def list_clients(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    rows = (
        await session.execute(
            select(Client)
            .options(selectinload(Client.keys))
            .order_by(Client.name)
        )
    ).scalars().all()
    return templates.TemplateResponse(
        request,
        "admin/clients.html",
        {
            "clients": rows,
            "kinds": VALID_KINDS,
            "new_key": request.query_params.get("new_key"),
            "for_client": request.query_params.get("for_client"),
        },
    )


@router.post("/admin/clients")
async def create_client(
    request: Request,
    name: str = Form(...),
    kind: str = Form("app"),
    description: str = Form(""),
    privileged: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    client_name = _required_str(name, "name", max_len=64)
    client_kind = _validate_kind(kind)

    existing = (
        await session.execute(select(Client).where(Client.name == client_name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Client name already exists")

    session.add(
        Client(
            name=client_name,
            kind=client_kind,
            description=_optional_str(description, max_len=4000),
            disabled=False,
            privileged=(privileged == "on"),
        )
    )
    await session.commit()
    # A new client holds no keys yet, so the cache is unchanged.
    return RedirectResponse("/admin/clients", status_code=303)


@router.post("/admin/clients/{client_id}")
async def update_client(
    client_id: int,
    request: Request,
    name: str = Form(...),
    kind: str = Form("app"),
    description: str = Form(""),
    privileged: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_client(session, client_id)

    client_name = _required_str(name, "name", max_len=64)
    clash = (
        await session.execute(
            select(Client).where(Client.name == client_name, Client.id != client_id)
        )
    ).scalar_one_or_none()
    if clash is not None:
        raise HTTPException(status_code=409, detail="Client name already exists")

    target.name = client_name
    target.kind = _validate_kind(kind)
    target.description = _optional_str(description, max_len=4000)
    target.privileged = (privileged == "on")
    await session.commit()

    # `privileged` rides the key cache; without a refresh the change waits for
    # a restart.
    await refresh_key_cache(session)
    return RedirectResponse("/admin/clients", status_code=303)


@router.post("/admin/clients/{client_id}/disable")
async def disable_client(
    client_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_client(session, client_id)
    target.disabled = True
    await session.commit()

    # The client's keys must stop authenticating immediately.
    await refresh_key_cache(session)
    return RedirectResponse("/admin/clients", status_code=303)


@router.post("/admin/clients/{client_id}/enable")
async def enable_client(
    client_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_client(session, client_id)
    target.disabled = False
    await session.commit()

    # Re-admit the client's un-revoked keys.
    await refresh_key_cache(session)
    return RedirectResponse("/admin/clients", status_code=303)


@router.post("/admin/clients/{client_id}/delete")
async def delete_client(
    client_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_client(session, client_id)
    await session.delete(target)
    await session.commit()

    # Cascade removed the client's keys along with it.
    await refresh_key_cache(session)
    return RedirectResponse("/admin/clients", status_code=303)


@router.post("/admin/clients/{client_id}/keys")
async def mint_key(
    client_id: int,
    request: Request,
    label: str = Form("default"),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_client(session, client_id)

    raw_key = generate_proxy_key()
    session.add(
        ApiKey(
            client_id=target.id,
            label=_optional_str(label, max_len=64) or "default",
            key_hash=_hash_key(raw_key),
            revoked=False,
        )
    )
    await session.commit()

    await refresh_key_cache(session)
    # Plaintext is shown exactly once, on the redirect target — only the hash
    # was stored.
    return RedirectResponse(
        f"/admin/clients?new_key={raw_key}&for_client={target.name}",
        status_code=303,
    )


@router.post("/admin/clients/{client_id}/keys/{key_id}/revoke")
async def revoke_key(
    client_id: int,
    key_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.client_id == client_id)
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Key not found")
    target.revoked = True
    await session.commit()

    await refresh_key_cache(session)
    return RedirectResponse("/admin/clients", status_code=303)
