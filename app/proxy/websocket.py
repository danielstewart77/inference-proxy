"""WebSocket Responses API endpoints (Codex CLI uses /v1/responses over wss)."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import (
    is_invalid_encrypted_content,
    retry_delay,
    sanitize_stream_options,
    strip_encrypted_reasoning,
    strip_unsupported_input_fields,
)
from app.db import SessionLocal
from app.deployments import reject_wrong_protocol, resolve_deployment
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "responses"


@router.websocket("/v1/responses")
async def responses_websocket(websocket: WebSocket):
    """WebSocket entry point for Codex CLI's Responses API."""
    log(
        f"[Responses WS] Subprotocols requested: "
        f"{websocket.headers.get('sec-websocket-protocol', 'none')}"
    )

    requested_protocol = websocket.headers.get("sec-websocket-protocol")
    if requested_protocol:
        first_protocol = requested_protocol.split(",")[0].strip()
        await websocket.accept(subprotocol=first_protocol)
    else:
        await websocket.accept()
    log("[Responses WS] Connection accepted, validating auth...")

    api_key = websocket.query_params.get("api_key")
    auth_header = websocket.headers.get("authorization")
    api_key_header = websocket.headers.get("api-key")

    if not validate_api_key(auth_header, api_key_header or api_key):
        log(
            f"[Responses WS] Auth failed — authorization={auth_header!r}, "
            f"api-key={api_key_header!r}, query api_key={api_key!r}"
        )
        await websocket.send_text(json.dumps({"error": "Invalid API key"}))
        await websocket.close(code=4003, reason="Invalid API key")
        return

    principal = resolve_principal(auth_header, api_key_header or api_key)
    is_admin = resolve_requester_role(auth_header, api_key_header or api_key) == "admin"
    started_at = time.time()
    captured_usage: dict = {}
    captured_error: str | None = None
    body: dict | None = None

    log("[Responses WS] Auth OK, waiting for client message...")

    try:
        raw = await websocket.receive_text()
        log(f"[Responses WS] Got message ({len(raw)} bytes)")
        body = json.loads(raw)

        model = body.get("model")
        if not model:
            await websocket.send_text(json.dumps({"error": "Missing required 'model' field"}))
            await websocket.close(code=4400, reason="Missing model")
            return
        log(f"[Responses WS] Received request (model={model})")

        try:
            async with SessionLocal() as session:
                target = await resolve_deployment(
                    session, model, is_admin=is_admin, wire="openai_responses"
                )
                reject_wrong_protocol(target, expected="openai_responses")
        except Exception as e:
            msg = getattr(e, "detail", str(e))
            await websocket.send_text(json.dumps({"error": msg}))
            await websocket.close(code=4404, reason="Unknown deployment")
            return

        azure_body = dict(body)
        azure_body.setdefault("store", False)
        azure_body["stream"] = True
        for k in ("type",):
            azure_body.pop(k, None)
        azure_body, stripped_fields = strip_unsupported_input_fields(azure_body)
        if stripped_fields:
            log(
                f"[Responses WS] Stripped unsupported input field(s) from "
                f"{stripped_fields} item(s) before forwarding to Azure"
            )
        azure_body, dropped_opts = sanitize_stream_options(azure_body)
        if dropped_opts:
            log(
                f"[Responses WS] Dropped unsupported stream_options key(s) "
                f"{dropped_opts} before forwarding to Azure"
            )

        azure_url = target.upstream_url
        headers = build_upstream_headers(target)

        max_retries = 5
        line_count = 0
        stripped_reasoning = False
        for attempt in range(1, max_retries + 1):
            log(f"[Responses WS] Forwarding to Azure (attempt {attempt}/{max_retries}): {azure_url}")

            got_completed = False
            got_error = False
            error_type = ""
            line_count = 0
            flushed = False
            buffer: list[str] = []

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.send(
                    client.build_request("POST", azure_url, headers=headers, json=azure_body),
                    stream=True,
                )

                log(f"[Responses WS] Azure response status: {response.status_code}")

                if response.status_code != 200:
                    error_body = await response.aread()
                    log(
                        f"[Responses WS] Azure HTTP error {response.status_code}: "
                        f"{error_body[:500]}"
                    )
                    if (
                        not stripped_reasoning
                        and response.status_code == 400
                        and is_invalid_encrypted_content(error_body)
                    ):
                        azure_body, removed = strip_encrypted_reasoning(azure_body)
                        if removed:
                            stripped_reasoning = True
                            log(
                                f"[Responses WS] invalid_encrypted_content 400 — "
                                f"stripped {removed} reasoning item(s), retrying "
                                f"immediately (attempt {attempt}/{max_retries})"
                            )
                            continue
                    if attempt < max_retries:
                        delay = retry_delay(
                            attempt, is_rate_limit=(response.status_code == 429)
                        )
                        log(f"[Responses WS] Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    await websocket.send_text(
                        json.dumps({"error": f"Azure API error: {response.status_code}"})
                    )
                    await websocket.close(code=1011)
                    return

                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    if stripped.startswith("data: "):
                        payload = stripped[6:]
                    elif stripped.startswith("data:"):
                        payload = stripped[5:]
                    elif stripped.startswith("event:"):
                        continue
                    else:
                        payload = stripped

                    if not payload or payload == "[DONE]":
                        continue

                    line_count += 1

                    if '"type":"error"' in payload or '"type": "error"' in payload:
                        got_error = True
                        try:
                            error_type = (
                                json.loads(payload).get("error", {}).get("type", "")
                            )
                        except Exception:
                            error_type = ""
                        log(
                            f"[Responses WS] Azure stream error ({error_type}) "
                            f"on line {line_count}: {payload[:300]}"
                        )
                        break

                    if "response.completed" in payload:
                        got_completed = True
                        try:
                            evt = json.loads(payload)
                            captured_usage = (
                                evt.get("response", {}).get("usage", {}) or {}
                            )
                        except Exception:
                            pass

                    if line_count <= 3 or got_completed:
                        log(f"[Responses WS] Line {line_count}: {payload[:200]}")

                    if not flushed:
                        buffer.append(payload)
                        if any(
                            t in payload
                            for t in (
                                "output_text.delta",
                                "output_item.added",
                                "content_part.added",
                                "response.completed",
                            )
                        ):
                            for buffered in buffer:
                                await websocket.send_text(buffered)
                            buffer.clear()
                            flushed = True
                    else:
                        await websocket.send_text(payload)

            if got_error and not flushed:
                if attempt < max_retries:
                    is_rate_limit = error_type == "too_many_requests"
                    delay = retry_delay(attempt, is_rate_limit=is_rate_limit)
                    log(
                        f"[Responses WS] Azure returned {error_type or 'error'} "
                        f"after {line_count} events, retrying in {delay}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                log(
                    f"[Responses WS] Azure errored on all {max_retries} attempts, "
                    f"forwarding error to client"
                )
                captured_error = error_type or "stream_error"
                for buffered in buffer:
                    await websocket.send_text(buffered)

            log(
                f"[Responses WS] Stream complete, sent {line_count} events to client, "
                f"got_completed={got_completed}, got_error={got_error}, attempts={attempt}"
            )
            break

        # Wait for client disconnect (with timeout) so the client can process
        # response.completed before we close.
        try:
            await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        except (asyncio.TimeoutError, WebSocketDisconnect):
            pass
        try:
            await websocket.close(code=1000)
        except Exception:
            pass

    except WebSocketDisconnect:
        log("[Responses WS] Client disconnected")
        captured_error = captured_error or "client_disconnect"
    except Exception as e:
        log(f"[Responses WS] Error: {type(e).__name__}: {e}")
        captured_error = captured_error or type(e).__name__
        try:
            await websocket.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass
    finally:
        model_name = body.get("model") if isinstance(body, dict) else None
        if model_name:
            usage.schedule(
                principal,
                websocket=websocket,
                model=model_name,
                endpoint=ENDPOINT_LABEL,
                status_code=200 if not captured_error else 500,
                error_type=captured_error,
                started_at=started_at,
                azure_usage=captured_usage,
            )


@router.websocket("/responses")
@router.websocket("/openai/v1/responses")
async def responses_websocket_alt(websocket: WebSocket):
    """Alternate paths Codex might try; forward to the main handler."""
    log(
        f"[Responses WS Alt] Hit alternate path: {websocket.url.path} "
        f"— forwarding to main handler"
    )
    await responses_websocket(websocket)
