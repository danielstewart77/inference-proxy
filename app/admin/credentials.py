"""/admin/credentials — manage the upstream credentials models point at.

Stored secrets and cached access tokens are write-only from the console: the
list shows whether a value is set, never the value itself, and a blank field on
save leaves the stored column untouched.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_html_admin
from app.db import get_session
from app.orm import Credential, Model
from app.templating import templates

router = APIRouter()

VALID_KINDS = ("static", "anthropic_oauth", "openai_oauth")

# Kinds whose secret is read from a file on disk rather than stored inline.
FILE_KINDS = ("anthropic_oauth", "openai_oauth")


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


async def _get_credential(session: AsyncSession, credential_id: int) -> Credential:
    target = (
        await session.execute(select(Credential).where(Credential.id == credential_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Credential not found")
    return target


@router.get("/admin/credentials", response_class=HTMLResponse)
async def list_credentials(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    rows = (
        await session.execute(select(Credential).order_by(Credential.name))
    ).scalars().all()
    usage_rows = (
        await session.execute(
            select(Model.credential_id, Model.deployment_name)
            .where(Model.credential_id.is_not(None))
            .order_by(Model.deployment_name)
        )
    ).all()
    used_by: dict[int, list[str]] = {}
    for credential_id, deployment_name in usage_rows:
        used_by.setdefault(credential_id, []).append(deployment_name)
    return templates.TemplateResponse(
        request,
        "admin/credentials.html",
        {"credentials": rows, "kinds": VALID_KINDS, "used_by": used_by},
    )


@router.post("/admin/credentials")
async def create_credential(
    request: Request,
    name: str = Form(...),
    kind: str = Form("static"),
    secret: str = Form(""),
    source_path: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    cred_name = _required_str(name, "name", max_len=64)
    cred_kind = _validate_kind(kind)

    existing = (
        await session.execute(select(Credential).where(Credential.name == cred_name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Credential name already exists")

    if cred_kind == "static":
        cred_secret = _required_str(secret, "secret", max_len=2048)
        cred_path = None
    elif cred_kind == "anthropic_oauth":
        # Either a literal long-lived token or a CLI credential file; a
        # stored secret takes precedence at resolve time.
        cred_secret = _optional_str(secret, max_len=2048)
        cred_path = _optional_str(source_path, max_len=512)
        if cred_secret is None and cred_path is None:
            raise HTTPException(
                status_code=400,
                detail="anthropic_oauth needs a secret (long-lived token) or a source_path",
            )
    else:
        cred_secret = None
        cred_path = _required_str(source_path, "source_path", max_len=512)

    session.add(
        Credential(
            name=cred_name,
            kind=cred_kind,
            secret=cred_secret,
            source_path=cred_path,
        )
    )
    await session.commit()
    return RedirectResponse("/admin/credentials", status_code=303)


@router.post("/admin/credentials/{credential_id}")
async def update_credential(
    credential_id: int,
    request: Request,
    name: str = Form(...),
    kind: str = Form("static"),
    secret: str = Form(""),
    source_path: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_credential(session, credential_id)

    cred_name = _required_str(name, "name", max_len=64)
    cred_kind = _validate_kind(kind)
    clash = (
        await session.execute(
            select(Credential).where(
                Credential.name == cred_name, Credential.id != credential_id
            )
        )
    ).scalar_one_or_none()
    if clash is not None:
        raise HTTPException(status_code=409, detail="Credential name already exists")

    kind_changed = cred_kind != target.kind
    target.name = cred_name
    target.kind = cred_kind

    # A blank field means "leave the stored value alone" — the console never
    # renders the current one back, so it cannot be resubmitted.
    new_secret = _optional_str(secret, max_len=2048)
    if new_secret is not None:
        target.secret = new_secret
    new_path = _optional_str(source_path, max_len=512)
    if new_path is not None:
        target.source_path = new_path

    if cred_kind == "static":
        if not target.secret:
            raise HTTPException(status_code=400, detail="secret is required")
        target.source_path = None
    elif cred_kind == "anthropic_oauth":
        if not target.secret and not target.source_path:
            raise HTTPException(
                status_code=400,
                detail="anthropic_oauth needs a secret (long-lived token) or a source_path",
            )
    else:
        if not target.source_path:
            raise HTTPException(status_code=400, detail="source_path is required")

    if kind_changed or new_path is not None or new_secret is not None:
        # The cached token belonged to the previous kind or file; drop it so the
        # next request resolves afresh.
        target.access_token = None
        target.expires_at = None
        target.refreshed_at = None

    await session.commit()
    return RedirectResponse("/admin/credentials", status_code=303)


@router.post("/admin/credentials/{credential_id}/delete")
async def delete_credential(
    credential_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = await _get_credential(session, credential_id)
    # Detach any models pointing here before the row goes, so they don't keep a
    # dangling id on backends that don't enforce ON DELETE SET NULL.
    await session.execute(
        update(Model)
        .where(Model.credential_id == credential_id)
        .values(credential_id=None)
    )
    await session.delete(target)
    await session.commit()
    return RedirectResponse("/admin/credentials", status_code=303)
