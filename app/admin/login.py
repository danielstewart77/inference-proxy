"""/login, /logout, /admin/ landing.

The admin is a single operator identity held in environment config — there is
no user row to look up, so login is a constant-time comparison and the session
carries nothing but an "is admin" flag.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    is_admin_session,
    login_session,
    logout_session,
    proxy_key_count,
    require_html_admin,
    verify_admin_credentials,
)
from app.db import get_session
from app.orm import Client, Credential, Model
from app.templating import templates

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    if is_admin_session(request):
        return RedirectResponse("/admin/", status_code=303)
    return templates.TemplateResponse(request, "admin/login.html", {"error": None})


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if not verify_admin_credentials(username, password):
        return templates.TemplateResponse(
            request,
            "admin/login.html",
            {"error": "Invalid username or password"},
            status_code=401,
        )
    login_session(request)
    return RedirectResponse("/admin/", status_code=303)


@router.post("/logout")
async def logout(request: Request):
    logout_session(request)
    return RedirectResponse("/login", status_code=303)


@router.get("/admin/", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    client_count = (
        await session.execute(select(func.count()).select_from(Client))
    ).scalar_one()
    credential_count = (
        await session.execute(select(func.count()).select_from(Credential))
    ).scalar_one()
    model_count = (
        await session.execute(select(func.count()).select_from(Model))
    ).scalar_one()
    return templates.TemplateResponse(
        request,
        "admin/dashboard.html",
        {
            "client_count": client_count,
            "credential_count": credential_count,
            "model_count": model_count,
            "active_key_count": proxy_key_count(),
        },
    )
