"""Native Anthropic Messages passthrough.

Forwards /v1/messages bodies untouched to whatever upstream the requested
deployment points at — typically Foundry's `/anthropic/v1/messages` surface.

The proxy still:
  - validates the caller's proxy API key
  - confirms the requested `model` resolves to a registered, enabled
    deployment whose `target_uri` is an Anthropic Messages endpoint
  - removes Foundry-unsupported Anthropic beta fields
  - logs usage to the usage_log table

No response transformation happens — the client speaks real Anthropic to the
proxy, and the proxy speaks real Anthropic to the upstream.
"""

from __future__ import annotations

import json
import re
import time
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import post_with_retries
from app.db import get_session
from app.deployments import (
    DeploymentTarget,
    build_upstream_headers,
    reject_wrong_protocol,
    resolve_deployment,
)
from app.orm import Model
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "messages"
DATED_MODEL_ALIAS_RE = re.compile(r"^(.+)-(\d{8})$")
# Context-window variant ids, e.g. `claude-fable-5[1m]`.
BRACKET_VARIANT_RE = re.compile(r"^(.+)\[[^\]]+\]$")
UNSUPPORTED_FOUNDRY_BODY_FIELDS = frozenset({"context_management"})

# Required when authenticating with an Anthropic subscription OAuth token.
OAUTH_BETA_FLAG = "oauth-2025-04-20"
OAUTH_SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."


def _extract_api_key_from_anthropic_headers(
    authorization: str | None,
    x_api_key: str | None,
) -> str | None:
    if x_api_key:
        return f"Bearer {x_api_key}"
    return authorization


def _forward_headers(request: Request, target: DeploymentTarget) -> dict:
    """Build the headers for the upstream call.

    Auth is per-deployment (`target.auth_scheme`) — Foundry's
    `/anthropic/v1/messages` route authenticates with `Authorization: Bearer
    <key>`, real `api.anthropic.com` wants `x-api-key: <key>` instead. Either
    way the standard Anthropic `anthropic-version` header is required. Claude
    Code may send Anthropic beta flags that Foundry's partner route does not
    recognize, so do not forward `anthropic-beta` there — real Anthropic does
    recognize them, so let those through unfiltered.
    """
    headers = build_upstream_headers(target)
    headers["anthropic-version"] = "2023-06-01"
    is_foundry = target.auth_scheme != "x_api_key" and not target.is_anthropic_oauth
    for name, value in request.headers.items():
        lname = name.lower()
        if lname == "anthropic-beta" and is_foundry:
            log(f"[Anthropic API] Dropping unsupported anthropic-beta header: {value}")
            continue
        if lname.startswith("anthropic-"):
            # Client-supplied anthropic-* headers override our default
            # (e.g. they may want a newer anthropic-version).
            headers[lname] = value

    if target.is_anthropic_oauth:
        # A subscription OAuth token is only honoured when the request carries
        # the OAuth beta flag; without it the API answers 429 rather than 401.
        # Merge rather than overwrite so a client's own beta flags survive.
        existing = [f.strip() for f in headers.get("anthropic-beta", "").split(",") if f.strip()]
        if OAUTH_BETA_FLAG not in existing:
            existing.insert(0, OAUTH_BETA_FLAG)
        headers["anthropic-beta"] = ",".join(existing)
    return headers


def apply_oauth_system_prompt(body: dict) -> dict:
    """Prepend the CLI identity block an Anthropic OAuth token requires.

    A subscription token is scoped to the Claude Code client, and the API only
    serves it when the first system block declares that identity. Anything the
    caller sent as `system` is preserved after it. Normalises a plain string
    `system` into block form, which is the only shape that can hold two blocks.
    """
    identity = {"type": "text", "text": OAUTH_SYSTEM_IDENTITY}
    existing = body.get("system")
    if existing is None:
        blocks = []
    elif isinstance(existing, str):
        blocks = [{"type": "text", "text": existing}]
    elif isinstance(existing, list):
        blocks = list(existing)
    else:
        blocks = [existing]

    first = blocks[0] if blocks else None
    if isinstance(first, dict) and first.get("text") == OAUTH_SYSTEM_IDENTITY:
        return body

    body["system"] = [identity, *blocks]
    return body


def _usage_from_anthropic(payload: dict) -> dict:
    """Normalize Anthropic's usage block into the shape `usage.schedule` expects."""
    u = payload.get("usage") or {}
    inp = u.get("input_tokens")
    out = u.get("output_tokens")
    total = None
    if isinstance(inp, int) and isinstance(out, int):
        total = inp + out
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
    }


def _base_model_from_dated_alias(model: object) -> str | None:
    if not isinstance(model, str):
        return None
    match = DATED_MODEL_ALIAS_RE.match(model)
    if not match:
        return None
    return match.group(1)


def _base_model_from_bracket_variant(model: object) -> str | None:
    """`claude-fable-5[1m]` -> `claude-fable-5`.

    Claude Code marks context-window variants with a bracket suffix. Unlike
    dated aliases, the bracketed id must be forwarded upstream unchanged —
    rewriting it to the base name would silently drop the variant.
    """
    if not isinstance(model, str):
        return None
    match = BRACKET_VARIANT_RE.match(model)
    if not match:
        return None
    return match.group(1)


async def _resolve_anthropic_deployment(
    session: AsyncSession,
    requested_model: object,
    *,
    is_admin: bool = False,
) -> tuple[DeploymentTarget, bool]:
    """Resolve Anthropic models, accepting Claude CLI aliases.

    Returns `(target, rewrite_model)`. Two alias shapes are normalized when no
    exact row matches:

    - Dated aliases (`claude-haiku-4-5-20251001`): Foundry rejects the dated
      name, so the forwarded body is rewritten to the base row's name
      (`rewrite_model=True`).
    - Bracket variants (`claude-fable-5[1m]`): the upstream understands the
      bracketed id and it carries meaning (context window), so the body keeps
      the requested name (`rewrite_model=False`).
    """
    try:
        return (
            await resolve_deployment(
                session, requested_model, error_kind="anthropic", is_admin=is_admin
            ),
            False,
        )
    except HTTPException as exc:
        if exc.status_code != 404:
            raise

        dated_base = _base_model_from_dated_alias(requested_model)
        bracket_base = _base_model_from_bracket_variant(requested_model)
        base_model = dated_base or bracket_base
        if not base_model:
            raise

        exact_row = (
            await session.execute(
                select(Model.id).where(Model.deployment_name == requested_model)
            )
        ).scalar_one_or_none()
        if exact_row is not None:
            raise

        target = await resolve_deployment(
            session, base_model, error_kind="anthropic", is_admin=is_admin
        )
        log(f"[Anthropic API] Normalized model alias {requested_model!r} -> {target.name!r}")
        return target, dated_base is not None


def _strip_unsupported_foundry_fields(body: dict) -> dict:
    """Remove Anthropic beta request fields that Foundry currently rejects."""
    dropped = [field for field in UNSUPPORTED_FOUNDRY_BODY_FIELDS if field in body]
    if not dropped:
        return body

    sanitized = dict(body)
    for field in dropped:
        sanitized.pop(field, None)
    log(f"[Anthropic API] Dropping unsupported request field(s): {', '.join(sorted(dropped))}")
    return sanitized


@router.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    authorization: str | None = Header(None, alias="Authorization"),
    x_api_key: str | None = Header(None, alias="x-api-key"),
    session: AsyncSession = Depends(get_session),
):
    started_at = time.time()
    effective_auth = _extract_api_key_from_anthropic_headers(authorization, x_api_key)
    if not validate_api_key(effective_auth):
        raise HTTPException(
            status_code=401,
            detail={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid API key"},
            },
        )
    principal = resolve_principal(effective_auth)
    is_admin = resolve_requester_role(effective_auth) == "admin"

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Invalid JSON in request body"},
            },
        )

    model = body.get("model")
    target, rewrite_model = await _resolve_anthropic_deployment(
        session, model, is_admin=is_admin
    )
    reject_wrong_protocol(target, expected="anthropic_messages", error_kind="anthropic")
    if rewrite_model and model != target.name:
        body = dict(body)
        body["model"] = target.name
    body = _strip_unsupported_foundry_fields(body)
    if target.is_anthropic_oauth:
        body = apply_oauth_system_prompt(dict(body))

    is_streaming = bool(body.get("stream", False))

    log(f"[Anthropic API] Received request (model={model}, upstream_model={target.name})")
    log(f"[Anthropic API] Request type: {'streaming' if is_streaming else 'non-streaming'}")

    upstream = target.upstream_url
    log(f"[Anthropic API] Forwarding to: {upstream}")

    headers = _forward_headers(request, target)

    try:
        if is_streaming:
            return await _handle_streaming(
                upstream, headers, body, target,
                principal=principal, request=request, started_at=started_at,
            )
        return await _handle_non_streaming(
            upstream, headers, body, target,
            principal=principal, request=request, started_at=started_at,
        )
    except httpx.ConnectError as exc:
        log(f"[Anthropic API] Returning 502 to client after connect failure: {exc!r}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=502, error_type="connect_error", started_at=started_at,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "type": "error",
                "error": {"type": "api_error", "message": "Failed to reach upstream"},
            },
        )
    except httpx.TimeoutException as exc:
        log(f"[Anthropic API] Returning 504 to client after upstream timeout: {exc!r}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=504, error_type="timeout", started_at=started_at,
        )
        raise HTTPException(
            status_code=504,
            detail={
                "type": "error",
                "error": {"type": "api_error", "message": "Upstream timed out"},
            },
        )


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
        url, headers, body, stream=False, log_prefix="[Anthropic API]"
    )

    if response.status_code != 200:
        log(f"[Anthropic API] Upstream error {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        # Pass through the upstream error body verbatim when possible. Native
        # Anthropic endpoints already return the expected error envelope.
        try:
            err_body = response.json()
        except Exception:
            err_body = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream API error: {response.status_code}",
                },
            }
        return JSONResponse(status_code=response.status_code, content=err_body)

    log(f"[Anthropic API] Response received: {len(response.content)} bytes")
    result = response.json()
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage=_usage_from_anthropic(result),
    )
    return JSONResponse(content=result)


async def _passthrough_stream(
    response: httpx.Response, *, usage_holder: dict
) -> AsyncIterator[bytes]:
    """Pipe upstream SSE bytes through unchanged while sniffing token usage.

    Anthropic SSE puts `input_tokens` in the `message_start.message.usage`
    block and the running `output_tokens` in subsequent `message_delta.usage`
    blocks. We capture the latest values seen and write them into
    `usage_holder` so the caller can log them after the stream closes.
    """
    buffer = b""
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        yield chunk

        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    parsed = json.loads(data)
                except (ValueError, json.JSONDecodeError):
                    continue
                etype = parsed.get("type")
                if etype == "message_start":
                    msg_usage = (parsed.get("message") or {}).get("usage") or {}
                    for field in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    ):
                        if field in msg_usage:
                            usage_holder[field] = msg_usage[field]
                elif etype == "message_delta":
                    delta_usage = parsed.get("usage") or {}
                    if "output_tokens" in delta_usage:
                        usage_holder["output_tokens"] = delta_usage["output_tokens"]


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
        url, headers, body, stream=True, log_prefix="[Anthropic API]"
    )

    if response.status_code != 200:
        error_body = await response.aread()
        await response.aclose()
        log(f"[Anthropic API] Upstream error {response.status_code}: {error_body[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        try:
            err_body = json.loads(error_body)
        except Exception:
            err_body = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream API error: {response.status_code}",
                },
            }
        return JSONResponse(status_code=response.status_code, content=err_body)

    async def passthrough():
        # Anthropic SSE reports usage in two places:
        #   - message_start.message.usage.input_tokens (final input count)
        #   - message_delta.usage.output_tokens        (final output count)
        # We merge both so the metering row gets accurate token counts.
        captured: dict = {}
        captured_error: Optional[str] = None
        try:
            async for chunk in _passthrough_stream(response, usage_holder=captured):
                yield chunk
        except httpx.HTTPError as exc:
            # Upstream died mid-stream. Bytes are already sent, so this is not
            # retryable — close out with a well-formed Anthropic error event
            # rather than a cold connection abort.
            captured_error = f"stream_{type(exc).__name__}"
            log(f"[Anthropic API] Upstream stream aborted mid-response: {exc!r}")
            yield (
                b'event: error\n'
                b'data: {"type":"error","error":{"type":"api_error",'
                b'"message":"Upstream connection dropped mid-stream"}}\n\n'
            )
        finally:
            await response.aclose()
            if "input_tokens" in captured or "output_tokens" in captured:
                captured["total_tokens"] = (
                    (captured.get("input_tokens") or 0)
                    + (captured.get("output_tokens") or 0)
                )
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=200,
                error_type=captured_error,
                started_at=started_at,
                azure_usage=captured,
            )

    return StreamingResponse(passthrough(), media_type="text/event-stream")
