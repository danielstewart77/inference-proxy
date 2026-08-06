"""/admin/models — manage advertised deployments + per-million costs."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


def _validate_auth_scheme(raw: str) -> str:
    val = (raw or "").strip()
    if val not in VALID_AUTH_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=f"auth_scheme must be one of {VALID_AUTH_SCHEMES}, got {val!r}",
        )
    return val


def _split_api_version(uri: str) -> tuple[str, Optional[str]]:
    """Lift any `api-version=...` query param out of `uri`.

    Returns (cleaned_uri, api_version_or_None). All other query params and
    the URL fragment are preserved. If `api-version` appears multiple times
    we keep the last value (matches Azure's own URL parsing).
    """
    parts = urlsplit(uri)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    extracted: Optional[str] = None
    remaining: list[tuple[str, str]] = []
    for k, v in pairs:
        if k.lower() == "api-version":
            extracted = v or None
        else:
            remaining.append((k, v))
    cleaned_query = urlencode(remaining)
    cleaned = urlunsplit((parts.scheme, parts.netloc, parts.path, cleaned_query, parts.fragment))
    return cleaned, extracted


def _reconcile_api_version(
    target_uri: str, form_api_version: Optional[str]
) -> tuple[str, Optional[str]]:
    """Normalize target_uri / api_version coming from the admin form.

    If the URI has `?api-version=...`, lift it into the api_version column.
    If the form also supplied a value, the two must match — otherwise we
    400 rather than silently picking one.
    """
    cleaned_uri, from_uri = _split_api_version(target_uri)
    if from_uri and form_api_version and from_uri != form_api_version:
        raise HTTPException(
            status_code=400,
            detail=(
                f"api-version mismatch: target_uri has {from_uri!r} but the "
                f"api_version field has {form_api_version!r}. Remove one."
            ),
        )
    final = form_api_version or from_uri
    return cleaned_uri, final


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


async def _credential_choices(session: AsyncSession) -> list[Credential]:
    """Credentials offered in the model form's `credential_id` select."""
    return list(
        (
            await session.execute(select(Credential).order_by(Credential.name))
        ).scalars().all()
    )


async def _resolve_credential_id(session: AsyncSession, raw: Optional[str]) -> Optional[int]:
    """Turn the form's `credential_id` value into a validated id or None.

    Blank means "no credential attached" — a model in that state cannot resolve
    a secret, so the proxy will reject requests to it until one is chosen.
    """
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
    unknown = [part for part in parts if part not in ("claude", "codex")]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown harness(es): {', '.join(unknown)}. Use claude and/or codex.",
        )
    return ",".join(sorted(set(parts))) or None


def _upstream_required(target_uri: Optional[str], provider_id: Optional[int]) -> str:
    """A model needs somewhere to go: a provider, or its own full URI."""
    uri = (target_uri or "").strip()
    if not uri and provider_id is None:
        raise HTTPException(
            status_code=400,
            detail="Choose a provider, or give this model its own target URI",
        )
    return uri


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
            "credentials": await _credential_choices(session),
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
            "credentials": await _credential_choices(session),
            "providers": await _provider_choices(session),
        },
    )


@router.post("/admin/models")
async def add_model(
    request: Request,
    deployment_name: str = Form(...),
    target_uri: str = Form(""),
    provider_id: str = Form(""),
    harnesses: str = Form(""),
    credential_id: str = Form(""),
    api_version: str = Form(""),
    auth_scheme: str = Form("bearer"),
    label: str = Form(""),
    description: str = Form(""),
    cost_per_million_input: str = Form(""),
    cost_per_million_output: str = Form(""),
    cost_per_million_cache_write: str = Form(""),
    cost_per_million_cache_read: str = Form(""),
    enabled: str = Form("on"),
    admin_only: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    await require_html_admin(request)
    name = _required_str(deployment_name, "deployment_name", max_len=128)
    prov_id = await _resolve_provider_id(session, provider_id)
    raw_uri = _upstream_required(target_uri, prov_id)
    if len(raw_uri) > 512:
        raise HTTPException(status_code=400, detail="target_uri must be 512 characters or fewer")
    harness_list = _parse_harnesses(harnesses)
    cred_id = await _resolve_credential_id(session, credential_id)
    scheme = _validate_auth_scheme(auth_scheme)
    cleaned_uri, final_api_version = _reconcile_api_version(
        raw_uri, _optional_str(api_version, max_len=64)
    )
    # Re-validate length after stripping (cleaned_uri is always <= raw_uri).
    if len(cleaned_uri) > 512:
        raise HTTPException(status_code=400, detail="target_uri must be 512 characters or fewer")

    existing = (
        await session.execute(
            select(Model).where(Model.deployment_name == name)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Deployment already exists")

    session.add(
        Model(
            deployment_name=name,
            label=_optional_str(label, max_len=128),
            description=_optional_str(description, max_len=4000),
            target_uri=cleaned_uri or None,
            provider_id=prov_id,
            harnesses=harness_list,
            credential_id=cred_id,
            api_version=final_api_version,
            auth_scheme=scheme,
            enabled=(enabled == "on"),
            admin_only=(admin_only == "on"),
            cost_per_million_input=_parse_decimal(cost_per_million_input),
            cost_per_million_output=_parse_decimal(cost_per_million_output),
            cost_per_million_cache_write=_parse_decimal(cost_per_million_cache_write),
            cost_per_million_cache_read=_parse_decimal(cost_per_million_cache_read),
        )
    )
    await session.commit()
    return RedirectResponse("/admin/models", status_code=303)


@router.post("/admin/models/{model_id}")
async def update_model(
    model_id: int,
    request: Request,
    target_uri: str = Form(""),
    provider_id: str = Form(""),
    harnesses: str = Form(""),
    credential_id: str = Form(""),
    api_version: str = Form(""),
    auth_scheme: str = Form("bearer"),
    label: str = Form(""),
    description: str = Form(""),
    cost_per_million_input: str = Form(""),
    cost_per_million_output: str = Form(""),
    cost_per_million_cache_write: str = Form(""),
    cost_per_million_cache_read: str = Form(""),
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

    prov_id = await _resolve_provider_id(session, provider_id)
    raw_uri = _upstream_required(target_uri, prov_id)
    if len(raw_uri) > 512:
        raise HTTPException(status_code=400, detail="target_uri must be 512 characters or fewer")
    cleaned_uri, final_api_version = _reconcile_api_version(
        raw_uri, _optional_str(api_version, max_len=64)
    )
    target.target_uri = cleaned_uri or None
    target.provider_id = prov_id
    target.harnesses = _parse_harnesses(harnesses)
    target.credential_id = await _resolve_credential_id(session, credential_id)
    target.api_version = final_api_version
    target.auth_scheme = _validate_auth_scheme(auth_scheme)
    target.label = _optional_str(label, max_len=128)
    target.description = _optional_str(description, max_len=4000)
    target.enabled = (enabled == "on")
    target.admin_only = (admin_only == "on")
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
