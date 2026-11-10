"""Embeddings endpoint: `POST /v1/embeddings`, the OpenAI shape.

`{"model", "input": str | [str]}` in, `{"data": [{"embedding", "index"}],
"model"}` out. Authenticated with the same client keys as chat, and forwarded
unchanged (bar the model name) to the provider's embeddings path, which for
Ollama is its OpenAI-compatible `/v1/embeddings`. Only a model of purpose
'embedding' resolves here, and such a model resolves nowhere else.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import post_with_retries
from app.db import get_session
from app.deployments import EMBEDDINGS_WIRE, build_upstream_headers, resolve_deployment
from app.proxy.chat_completions import _upstream_error_response
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "embeddings"


def _valid_input(value: object) -> bool:
    if isinstance(value, str):
        return bool(value)
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) for item in value)
    )


@router.post("/v1/embeddings")
async def embeddings_openai_style(
    request: Request,
    authorization: str | None = Header(None),
    api_key: str | None = Header(None, alias="api-key"),
    session: AsyncSession = Depends(get_session),
):
    started_at = time.time()
    if not validate_api_key(authorization, api_key):
        raise HTTPException(
            status_code=401 if not (authorization or api_key) else 403,
            detail="Invalid API key",
        )
    principal = resolve_principal(authorization, api_key)
    is_admin = resolve_requester_role(authorization, api_key) == "admin"
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="The body must be a JSON object")
    return await forward_embeddings(
        session, body, model=body.get("model"), is_admin=is_admin,
        principal=principal, request=request, started_at=started_at,
    )


async def forward_embeddings(
    session: AsyncSession,
    body: dict,
    *,
    model: Optional[str],
    is_admin: bool = False,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> JSONResponse:
    """Send an embeddings body for `model` to its provider and answer with
    the provider's vectors. Everything after the caller is authenticated:
    the route calls it, and `app.cli check-model` calls it for an embedding
    model. With no `principal` nothing is metered."""
    if started_at is None:
        started_at = time.time()
    target = await resolve_deployment(
        session, model, is_admin=is_admin, wire=EMBEDDINGS_WIRE)
    if not _valid_input(body.get("input")):
        raise HTTPException(
            status_code=400,
            detail="'input' must be a non-empty string or a non-empty list of strings",
        )
    outbound = dict(body)
    outbound["model"] = target.name
    log(f"[Embeddings] {target.name} -> {target.upstream_url}")
    try:
        response = await post_with_retries(
            target.upstream_url, build_upstream_headers(target), outbound,
            stream=False, log_prefix="[Embeddings]",
        )
    except httpx.ConnectError:
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=502, error_type="connect_error", started_at=started_at,
        )
        raise HTTPException(status_code=502, detail="Failed to reach upstream")
    except httpx.TimeoutException:
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=504, error_type="timeout", started_at=started_at,
        )
        raise HTTPException(status_code=504, detail="Upstream timed out")

    if response.status_code != 200:
        log(f"[Embeddings] upstream {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}", started_at=started_at,
        )
        return _upstream_error_response(response.status_code, response.content)

    result = response.json()
    counted = (result.get("usage") or {}) if isinstance(result, dict) else {}
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage={
            "input_tokens": counted.get("prompt_tokens"),
            "output_tokens": 0,
            "total_tokens": counted.get("total_tokens"),
        },
    )
    return JSONResponse(content=result)
