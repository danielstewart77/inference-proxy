"""/admin/providers — manage the upstreams that host models.

A provider is a host, a credential, and one path per request shape it serves.
Models attach to it and carry only their own name and pricing, which is what
lets one locally-hosted model answer a Claude harness on `/v1/messages` and a
Codex harness on `/v1/responses` without being registered twice — and what
makes adding a new vendor a row here rather than a code change.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_html_admin
from app.db import get_session
from app.orm import Credential, Model, Provider
from app.templating import templates

router = APIRouter()

VALID_AUTH_SCHEMES = ("api_key_header", "bearer", "x_api_key")


def _required(raw: Optional[str], field: str, *, max_len: int) -> str:
    val = (raw or "").strip()
    if not val:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"{field} must be {max_len} characters or fewer"
        )
    return val


def _optional(raw: Optional[str], *, max_len: int) -> Optional[str]:
    val = (raw or "").strip()
    if not val:
        return None
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"value must be {max_len} characters or fewer"
        )
    return val


def _auth_scheme(raw: str) -> str:
    val = (raw or "").strip()
    if val not in VALID_AUTH_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"auth_scheme must be one of {VALID_AUTH_SCHEMES}, got {val!r}",
        )
    return val


async def _credentials(session: AsyncSession) -> list[Credential]:
    return list(
        (await session.execute(select(Credential).order_by(Credential.name)))
        .scalars()
        .all()
    )


async def _resolve_credential_id(
    session: AsyncSession, raw: Optional[str]
) -> Optional[int]:
    val = (raw or "").strip()
    if not val:
        return None
    try:
        credential_id = int(val)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid credential_id: {val!r}")
    exists = (
        await session.execute(select(Credential.id).where(Credential.id == credential_id))
    ).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return credential_id


def _paths(messages: str, responses: str, chat: str) -> dict[str, Optional[str]]:
    """At least one shape, or the provider can serve nobody."""
    values = {
        "messages_path": _optional(messages, max_len=256),
        "responses_path": _optional(responses, max_len=256),
        "chat_completions_path": _optional(chat, max_len=256),
    }
    if not any(values.values()):
        raise HTTPException(
            status_code=400,
            detail="A provider must serve at least one request shape",
        )
    return values


@router.get("/admin/providers", response_class=HTMLResponse)
async def list_providers(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    rows = (
        await session.execute(select(Provider).order_by(Provider.name))
    ).scalars().all()
    counts = {}
    for provider_id, in (
        await session.execute(select(Model.provider_id).where(Model.provider_id.is_not(None)))
    ).all():
        counts[provider_id] = counts.get(provider_id, 0) + 1
    return templates.TemplateResponse(
        request,
        "admin/providers.html",
        {
            "providers": rows,
            "counts": counts,
            "credentials": await _credentials(session),
        },
    )


@router.post("/admin/providers")
async def add_provider(
    request: Request,
    name: str = Form(...),
    label: str = Form(""),
    base_url: str = Form(...),
    messages_path: str = Form(""),
    responses_path: str = Form(""),
    chat_completions_path: str = Form(""),
    credential_id: str = Form(""),
    auth_scheme: str = Form("bearer"),
    api_version: str = Form(""),
    enabled: str = Form("on"),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    provider_name = _required(name, "name", max_len=64)
    existing = (
        await session.execute(select(Provider).where(Provider.name == provider_name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Provider already exists")
    session.add(
        Provider(
            name=provider_name,
            label=_optional(label, max_len=128),
            base_url=_required(base_url, "base_url", max_len=512).rstrip("/"),
            credential_id=await _resolve_credential_id(session, credential_id),
            auth_scheme=_auth_scheme(auth_scheme),
            api_version=_optional(api_version, max_len=64),
            enabled=(enabled == "on"),
            **_paths(messages_path, responses_path, chat_completions_path),
        )
    )
    await session.commit()
    return RedirectResponse("/admin/providers", status_code=303)


@router.post("/admin/providers/{provider_id}")
async def update_provider(
    provider_id: int,
    request: Request,
    label: str = Form(""),
    base_url: str = Form(...),
    messages_path: str = Form(""),
    responses_path: str = Form(""),
    chat_completions_path: str = Form(""),
    credential_id: str = Form(""),
    auth_scheme: str = Form("bearer"),
    api_version: str = Form(""),
    enabled: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Provider).where(Provider.id == provider_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    target.label = _optional(label, max_len=128)
    target.base_url = _required(base_url, "base_url", max_len=512).rstrip("/")
    target.credential_id = await _resolve_credential_id(session, credential_id)
    target.auth_scheme = _auth_scheme(auth_scheme)
    target.api_version = _optional(api_version, max_len=64)
    target.enabled = enabled == "on"
    for column, value in _paths(
        messages_path, responses_path, chat_completions_path
    ).items():
        setattr(target, column, value)
    await session.commit()
    return RedirectResponse("/admin/providers", status_code=303)


@router.post("/admin/providers/{provider_id}/delete")
async def delete_provider(
    provider_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Provider).where(Provider.id == provider_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    attached = (
        await session.execute(
            select(Model.deployment_name).where(Model.provider_id == provider_id)
        )
    ).scalars().all()
    if attached:
        # Deleting would leave those models with no upstream at all, which
        # reads in the console as models that simply stopped working.
        raise HTTPException(
            status_code=409,
            detail=f"Still hosting {len(attached)} model(s): {', '.join(attached)}",
        )
    await session.delete(target)
    await session.commit()
    return RedirectResponse("/admin/providers", status_code=303)
