"""POST /v1/systemone — the decision wire.

A decision model is not a chat model with a different body. It takes one
state and a set of named questions about it, and answers each with a choice,
a confidence and a probability over every option offered, in one forward pass
emitting no output tokens. There is nothing to stream, nothing to translate
and no message list to build, so this route is a passthrough: it
authenticates the caller, resolves the model to the provider path that serves
this wire, and forwards the body as it arrived.

It exists so that nothing on this network calls Ollama directly. Ollama binds
loopback on purpose; every machine that needs a model reaches it through this
proxy with a proxy key, which is also the only place a model's identity, its
admin gate and its usage are recorded.
"""

from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.db import get_session
from app.deployments import build_upstream_headers, resolve_deployment
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "systemone"
WIRE = "systemone"
# A decision model is loaded cold on the first call of a pass -- most of a
# minute for a 27b -- and one call carries a whole window of questions. The
# chat timeout would abort the pass that warms it.
TIMEOUT_SECONDS = 600.0


@router.post("/v1/systemone")
async def systemone(
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

    if not isinstance(body.get("questions"), dict) or not body["questions"]:
        # Nothing to decide. Forwarding it spends a cold model load on a call
        # whose answer is an empty mapping either way.
        raise HTTPException(
            status_code=400, detail="At least one entry in 'questions' is required"
        )

    target = await resolve_deployment(
        session, body.get("model"), is_admin=is_admin, wire=WIRE
    )

    # The model the upstream is told is the deployment's own name, as every
    # other wire does it: a caller addresses the name this proxy publishes
    # and the upstream is addressed by the name it knows.
    upstream_body = dict(body)
    upstream_body["model"] = target.name

    log(f"[SystemOne] {principal} -> {target.name} at {target.upstream_url}")
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                target.upstream_url,
                headers=build_upstream_headers(target),
                json=upstream_body,
            )
    except httpx.HTTPError as exc:
        log(f"[SystemOne] upstream unreachable for {target.name}: {exc}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=502, error_type="upstream_unreachable", started_at=started_at,
        )
        raise HTTPException(
            status_code=502, detail=f"decision model unreachable: {exc}"
        ) from exc

    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=response.status_code, started_at=started_at,
    )
    # The upstream's own status and body, unaltered. A refusal reported as
    # something else is a remedy applied to the wrong machine.
    try:
        payload = response.json()
    except ValueError:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "decision model returned a non-JSON body",
                               "type": "api_error"}},
        )
    return JSONResponse(status_code=response.status_code, content=payload)
