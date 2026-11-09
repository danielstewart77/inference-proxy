"""/admin/models — manage advertised deployments + per-million costs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_html_admin
from app.db import get_session
from app.deployments import WITHHOLDABLE_HARNESSES
from app.orm import Model, Provider
from app.templating import templates

router = APIRouter()


def _required_str(raw: Optional[str], field: str, *, max_len: int) -> str:
    val = (raw or "").strip()
    if not val:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"{field} must be {max_len} characters or fewer"
        )
    return val


def _optional_str(raw: Optional[str], *, max_len: int) -> Optional[str]:
    val = (raw or "").strip()
    if not val:
        return None
    if len(val) > max_len:
        raise HTTPException(
            status_code=400, detail=f"value must be {max_len} characters or fewer"
        )
    return val


async def _provider_choices(session: AsyncSession) -> list[Provider]:
    """Upstreams offered in the model form's `provider_id` select."""
    return list(
        (await session.execute(select(Provider).order_by(Provider.name))).scalars().all()
    )


async def _resolve_provider_id(session: AsyncSession, raw: Optional[str]) -> Optional[int]:
    val = (raw or "").strip()
    if not val:
        return None
    try:
        provider_id = int(val)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider_id: {val!r}")
    exists = (
        await session.execute(select(Provider.id).where(Provider.id == provider_id))
    ).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider_id


def _parse_harnesses(raw: Optional[str]) -> Optional[str]:
    """Normalize the harness withholding list.

    Blank is the default and means every harness the provider can serve. The
    column exists to withhold a model from a harness that would otherwise
    reach it, so an empty value must round-trip to null rather than to a list
    matching nobody.
    """
    parts = [part.strip().lower() for part in (raw or "").split(",")]
    parts = [part for part in parts if part]
    unknown = [part for part in parts if part not in WITHHOLDABLE_HARNESSES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown harness(es): {', '.join(unknown)}. "
                   f"Use {', '.join(WITHHOLDABLE_HARNESSES)}.",
        )
    return ",".join(sorted(set(parts))) or None


#: Every effort level any harness accepts. Claude takes low through max, Codex
#: minimal through xhigh; a model's row names the subset it supports.
EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def _parse_effort_levels(raw: Optional[str]) -> Optional[str]:
    """Normalize a model's effort levels, keeping the order typed.

    Blank means the model takes no effort setting. An unknown word is refused
    rather than stored, since a stored typo is a button that spawns a harness
    on a flag value it rejects.
    """
    parts: list[str] = []
    for part in (raw or "").split(","):
        part = part.strip().lower()
        if part and part not in parts:
            parts.append(part)
    unknown = [part for part in parts if part not in EFFORT_LEVELS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown effort level(s): {', '.join(unknown)}. "
                   f"Use {', '.join(EFFORT_LEVELS)}.",
        )
    return ",".join(parts) or None


def _provider_required(provider_id: Optional[int]) -> int:
    """Every model is hosted by a provider, which routes and authenticates it.

    The model row's own target URI, credential and auth scheme are no longer
    read, so the form neither offers nor writes them.
    """
    if provider_id is None:
        raise HTTPException(status_code=400, detail="Choose a provider")
    return provider_id


def _parse_context_window(raw: Optional[str]) -> int:
    """The model's context window, which every model must declare.

    Required rather than optional: every conversation rotates at a percentage
    of the window of the model it is on, and a model saved without one is a
    model nobody can pick.
    """
    if raw is None or raw.strip() == "":
        raise HTTPException(status_code=400, detail="context_window is required")
    try:
        value = int(raw.strip())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid context window: {raw!r}")
    if value <= 0:
        raise HTTPException(status_code=400, detail="context_window must be positive")
    return value


def _parse_rotation_threshold(raw: Optional[str]) -> int:
    """The percentage of the context window a conversation on this model
    rotates at. Required like the window itself, and held to 1..99: zero
    rotates every turn and a hundred never rotates before the overflow."""
    if raw is None or raw.strip() == "":
        raise HTTPException(status_code=400, detail="rotation_threshold_percent is required")
    try:
        value = int(raw.strip())
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid rotation threshold: {raw!r}"
        )
    if not 1 <= value <= 99:
        raise HTTPException(
            status_code=400, detail="rotation_threshold_percent must be between 1 and 99"
        )
    return value


#: The three answers to "does this model read images", as the form spells them.
_ACCEPTS_IMAGES = {"yes": True, "no": False, "": None, "unknown": None}


def _parse_accepts_images(raw: Optional[str]) -> Optional[bool]:
    """True, False, or None for unknown. An unrecognised answer is refused
    rather than read as unknown, since a typo would otherwise erase a
    declared capability."""
    key = (raw or "").strip().lower()
    if key not in _ACCEPTS_IMAGES:
        raise HTTPException(status_code=400, detail=f"Invalid accepts_images: {raw!r}")
    return _ACCEPTS_IMAGES[key]


def _parse_decimal(raw: Optional[str]) -> Optional[Decimal]:
    if raw is None or raw.strip() == "":
        return None
    try:
        return Decimal(raw.strip())
    except InvalidOperation:
        raise HTTPException(status_code=400, detail=f"Invalid decimal: {raw!r}")


@router.get("/admin/models", response_class=HTMLResponse)
async def list_models(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    rows = (
        await session.execute(select(Model).order_by(Model.deployment_name))
    ).scalars().all()
    return templates.TemplateResponse(
        request,
        "admin/models.html",
        {"models": rows},
    )


# ---- Fragment endpoints used by htmx for in-page swaps -----------------------

@router.get("/admin/models/_add-button", response_class=HTMLResponse)
async def add_button_fragment(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    return templates.TemplateResponse(request, "admin/_models_add_button.html", {})


@router.get("/admin/models/_add-form", response_class=HTMLResponse)
async def add_form_fragment(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    return templates.TemplateResponse(
        request,
        "admin/_models_add_form.html",
        {
            "providers": await _provider_choices(session),
        },
    )


@router.get("/admin/models/{model_id}/_row", response_class=HTMLResponse)
async def row_fragment(
    model_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Model).where(Model.id == model_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(request, "admin/_models_row.html", {"m": target})


@router.get("/admin/models/{model_id}/_edit-form", response_class=HTMLResponse)
async def edit_form_fragment(
    model_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Model).where(Model.id == model_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(
        request,
        "admin/_models_edit_form.html",
        {
            "m": target,
            "providers": await _provider_choices(session),
        },
    )


async def insert_model(
    session: AsyncSession,
    *,
    deployment_name: str,
    provider_id: str,
    context_window: str,
    rotation_threshold_percent: str,
    harnesses: str = "",
    api_version: str = "",
    label: str = "",
    description: str = "",
    cost_per_million_input: str = "",
    cost_per_million_output: str = "",
    cost_per_million_cache_write: str = "",
    cost_per_million_cache_read: str = "",
    effort_levels: str = "",
    accepts_images: str = "",
    enabled: bool = True,
    admin_only: bool = False,
) -> Model:
    """Validate and store a new model; the console form and `app.cli` both
    land here. Values arrive as the form submits them. A taken name is 409."""
    name = _required_str(deployment_name, "deployment_name", max_len=128)
    prov_id = _provider_required(await _resolve_provider_id(session, provider_id))
    harness_list = _parse_harnesses(harnesses)

    existing = (
        await session.execute(
            select(Model).where(Model.deployment_name == name)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Deployment already exists")

    row = Model(
        deployment_name=name,
        label=_optional_str(label, max_len=128),
        description=_optional_str(description, max_len=4000),
        provider_id=prov_id,
        harnesses=harness_list,
        api_version=_optional_str(api_version, max_len=64),
        enabled=enabled,
        admin_only=admin_only,
        cost_per_million_input=_parse_decimal(cost_per_million_input),
        cost_per_million_output=_parse_decimal(cost_per_million_output),
        cost_per_million_cache_write=_parse_decimal(cost_per_million_cache_write),
        cost_per_million_cache_read=_parse_decimal(cost_per_million_cache_read),
        context_window=_parse_context_window(context_window),
        rotation_threshold_percent=_parse_rotation_threshold(rotation_threshold_percent),
        effort_levels=_parse_effort_levels(effort_levels),
        accepts_images=_parse_accepts_images(accepts_images),
    )
    session.add(row)
    await session.commit()
    return row


@router.post("/admin/models")
async def add_model(
    request: Request,
    deployment_name: str = Form(...),
    provider_id: str = Form(""),
    harnesses: str = Form(""),
    api_version: str = Form(""),
    label: str = Form(""),
    description: str = Form(""),
    cost_per_million_input: str = Form(""),
    cost_per_million_output: str = Form(""),
    cost_per_million_cache_write: str = Form(""),
    cost_per_million_cache_read: str = Form(""),
    context_window: str = Form(""),
    rotation_threshold_percent: str = Form(""),
    effort_levels: str = Form(""),
    accepts_images: str = Form(""),
    enabled: str = Form("on"),
    admin_only: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    await insert_model(
        session,
        deployment_name=deployment_name,
        provider_id=provider_id,
        harnesses=harnesses,
        api_version=api_version,
        label=label,
        description=description,
        cost_per_million_input=cost_per_million_input,
        cost_per_million_output=cost_per_million_output,
        cost_per_million_cache_write=cost_per_million_cache_write,
        cost_per_million_cache_read=cost_per_million_cache_read,
        context_window=context_window,
        rotation_threshold_percent=rotation_threshold_percent,
        effort_levels=effort_levels,
        accepts_images=accepts_images,
        enabled=(enabled == "on"),
        admin_only=(admin_only == "on"),
    )
    return RedirectResponse("/admin/models", status_code=303)


@router.post("/admin/models/{model_id}")
async def update_model(
    model_id: int,
    request: Request,
    provider_id: str = Form(""),
    harnesses: str = Form(""),
    api_version: str = Form(""),
    label: str = Form(""),
    description: str = Form(""),
    cost_per_million_input: str = Form(""),
    cost_per_million_output: str = Form(""),
    cost_per_million_cache_write: str = Form(""),
    cost_per_million_cache_read: str = Form(""),
    context_window: str = Form(""),
    rotation_threshold_percent: str = Form(""),
    effort_levels: str = Form(""),
    accepts_images: str = Form(""),
    enabled: str = Form(""),  # checkbox: "on" if checked, "" if not
    admin_only: str = Form(""),  # checkbox: "on" if checked, "" if not
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Model).where(Model.id == model_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")

    target.provider_id = _provider_required(
        await _resolve_provider_id(session, provider_id)
    )
    target.harnesses = _parse_harnesses(harnesses)
    target.api_version = _optional_str(api_version, max_len=64)
    target.label = _optional_str(label, max_len=128)
    target.description = _optional_str(description, max_len=4000)
    target.enabled = (enabled == "on")
    target.admin_only = (admin_only == "on")
    target.context_window = _parse_context_window(context_window)
    target.rotation_threshold_percent = _parse_rotation_threshold(rotation_threshold_percent)
    target.effort_levels = _parse_effort_levels(effort_levels)
    target.accepts_images = _parse_accepts_images(accepts_images)
    target.cost_per_million_input = _parse_decimal(cost_per_million_input)
    target.cost_per_million_output = _parse_decimal(cost_per_million_output)
    target.cost_per_million_cache_write = _parse_decimal(cost_per_million_cache_write)
    target.cost_per_million_cache_read = _parse_decimal(cost_per_million_cache_read)
    await session.commit()
    return RedirectResponse("/admin/models", status_code=303)


@router.post("/admin/models/{model_id}/delete")
async def delete_model(
    model_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    target = (
        await session.execute(select(Model).where(Model.id == model_id))
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Model not found")
    await session.delete(target)
    await session.commit()
    return RedirectResponse("/admin/models", status_code=303)
