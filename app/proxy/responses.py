"""Responses API passthrough (POST /v1/responses) — used by Codex CLI."""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import (
    is_invalid_encrypted_content,
    post_with_retries,
    sanitize_stream_options,
    strip_encrypted_reasoning,
    strip_unsupported_input_fields,
)
from app.db import get_session
from app.deployments import (
    HARNESS_HEADER,
    DeploymentTarget,
    build_upstream_headers,
    reject_wrong_protocol,
    resolve_deployment,
)
from app.proxy.overflow import overflow_body, overflow_failed_event, overflow_message
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "responses"


def _refusal(status_code: int, raw: bytes | str) -> JSONResponse:
    """The upstream's refusal, at its own status, in words a caller can act on.

    A context overflow is named `context_length_exceeded` whatever dialect
    the upstream said it in — the caller rotates on that code, and a
    blanket "Upstream API error: 400" carries nothing to rotate on. Any other
    refusal passes through as the upstream sent it; only a body that is not
    JSON is wrapped.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    overflow = overflow_message(raw)
    if overflow:
        return JSONResponse(status_code=status_code, content=overflow_body(overflow))
    try:
        content = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        content = {
            "error": {
                "message": raw.strip() or f"Upstream API error: {status_code}",
                "type": "api_error" if status_code >= 500 else "invalid_request_error",
            }
        }
    return JSONResponse(status_code=status_code, content=content)


@router.post("/v1/responses")
async def responses_passthrough(
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

    model = body.get("model")
    target = await resolve_deployment(
        session, model, is_admin=is_admin, wire="openai_responses",
        harness=request.headers.get(HARNESS_HEADER),
    )
    reject_wrong_protocol(target, expected="openai_responses")

    is_streaming = body.get("stream", False)

    log(f"[Responses API] Received request (model={target.name})")
    log(f"[Responses API] Request type: {'streaming' if is_streaming else 'non-streaming'}")

    azure_body = dict(body)
    azure_body.setdefault("store", False)
    for k in ("type",):
        azure_body.pop(k, None)
    azure_body, stripped_fields = strip_unsupported_input_fields(azure_body)
    if stripped_fields:
        log(
            f"[Responses API] Stripped unsupported input field(s) from "
            f"{stripped_fields} item(s) before forwarding to Azure"
        )
    azure_body, dropped_opts = sanitize_stream_options(azure_body)
    if dropped_opts:
        log(
            f"[Responses API] Dropped unsupported stream_options key(s) "
            f"{dropped_opts} before forwarding to Azure"
        )

    upstream = target.upstream_url
    log(f"[Responses API] Forwarding to: {upstream}")

    headers = build_upstream_headers(target)

    try:
        if is_streaming:
            return await _handle_streaming(
                upstream, headers, azure_body, target,
                principal=principal, request=request, started_at=started_at,
            )
        return await _handle_non_streaming(
            upstream, headers, azure_body, target,
            principal=principal, request=request, started_at=started_at,
        )
    except httpx.ConnectError as exc:
        log(f"[Responses API] Returning 502 to client after connect failure: {exc!r}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=502, error_type="connect_error", started_at=started_at,
        )
        raise HTTPException(status_code=502, detail="Failed to reach upstream")
    except httpx.TimeoutException as exc:
        log(f"[Responses API] Returning 504 to client after upstream timeout: {exc!r}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=504, error_type="timeout", started_at=started_at,
        )
        raise HTTPException(status_code=504, detail="Upstream timed out")


async def _handle_non_streaming(
    url: str,
    headers: dict,
    body: dict,
    target: DeploymentTarget,
    *,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> JSONResponse:
    response = await post_with_retries(
        url, headers, body, stream=False, log_prefix="[Responses API]"
    )
    if response.status_code == 400 and is_invalid_encrypted_content(response.content):
        retry_body, removed = strip_encrypted_reasoning(body)
        if removed:
            log(
                f"[Responses API] invalid_encrypted_content 400 — stripped "
                f"{removed} reasoning item(s), retrying once"
            )
            response = await post_with_retries(
                url, headers, retry_body, stream=False, log_prefix="[Responses API]"
            )

    if response.status_code != 200:
        log(f"[Responses API] Upstream error {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _refusal(response.status_code, response.content)

    log(f"[Responses API] Response received: {len(response.content)} bytes")
    azure_result = response.json()
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage=azure_result.get("usage", {}),
    )
    return JSONResponse(content=azure_result)


async def _handle_streaming(
    url: str,
    headers: dict,
    body: dict,
    target: DeploymentTarget,
    *,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> StreamingResponse:
    response = await post_with_retries(
        url, headers, body, stream=True, log_prefix="[Responses API]"
    )

    if response.status_code == 400:
        error_body = await response.aread()
        await response.aclose()
        if is_invalid_encrypted_content(error_body):
            retry_body, removed = strip_encrypted_reasoning(body)
            if removed:
                log(
                    f"[Responses API] invalid_encrypted_content 400 — stripped "
                    f"{removed} reasoning item(s), retrying once (streaming)"
                )
                response = await post_with_retries(
                    url, headers, retry_body, stream=True,
                    log_prefix="[Responses API]",
                )
            else:
                log(f"[Responses API] Azure error 400: {error_body[:500]}")
                usage.schedule(
                    principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                    status_code=400, error_type="azure_400", started_at=started_at,
                )
                return _refusal(400, error_body)
        else:
            log(f"[Responses API] Azure error 400: {error_body[:500]}")
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=400,
                error_type="context_length_exceeded"
                if overflow_message(error_body) else "azure_400",
                started_at=started_at,
            )
            return _refusal(400, error_body)

    if response.status_code != 200:
        error_body = await response.aread()
        await response.aclose()
        log(f"[Responses API] Upstream error {response.status_code}: {error_body[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _refusal(response.status_code, error_body)

    async def passthrough():
        captured_usage: dict = {}
        captured_error: Optional[str] = None
        try:
            async for line in response.aiter_lines():
                stripped = line.strip()
                normalized = _normalized_overflow_line(stripped)
                if normalized is not None:
                    captured_error = "context_length_exceeded"
                    yield normalized
                    continue
                yield f"{line}\n"
                if stripped.startswith("data: "):
                    payload = stripped[6:]
                elif stripped.startswith("data:"):
                    payload = stripped[5:]
                else:
                    continue
                if not payload or payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "response.completed":
                    captured_usage = evt.get("response", {}).get("usage", {}) or {}
                elif evt.get("type") == "error":
                    captured_error = evt.get("error", {}).get("type", "stream_error")
        finally:
            await response.aclose()
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=200,
                error_type=captured_error,
                started_at=started_at,
                azure_usage=captured_usage,
            )

    return StreamingResponse(passthrough(), media_type="text/event-stream")


def _normalized_overflow_line(stripped: str) -> Optional[str]:
    """A mid-stream overflow the upstream did not name, renamed; else None.

    The Codex backend already ends an oversized turn with `response.failed`
    carrying `code: context_length_exceeded`, and that line passes untouched.
    An upstream that says it only in a sentence (Ollama's Responses route)
    has its failure event rewritten into that same shape, so a Responses
    caller reads one overflow whichever upstream is behind the model.
    """
    if not stripped.startswith("data:"):
        return None
    payload = stripped[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        evt = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(evt, dict) or evt.get("type") not in ("response.failed", "error"):
        return None
    message = overflow_message(evt)
    if not message:
        return None
    failure = (evt.get("response") or {}).get("error") if evt.get("type") == "response.failed" else evt.get("error")
    if isinstance(failure, dict) and failure.get("code") == "context_length_exceeded":
        return None
    return f"data: {json.dumps(overflow_failed_event(message))}\n"
