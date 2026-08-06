"""POST /v1/responses/compact — Codex CLI context compaction."""

from __future__ import annotations

import json
import time

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import (
    COMPACT_SYSTEM_PROMPT,
    is_invalid_encrypted_content,
    strip_encrypted_reasoning,
)
from app.db import get_session
from app.deployments import reject_wrong_protocol, resolve_deployment
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "responses_compact"


@router.post("/v1/responses/compact")
async def responses_compact(
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

    log(f"\n{'='*60}")
    log(f"[COMPACT] >>>  POST /v1/responses/compact  <<<")
    log(f"{'='*60}")
    log(f"[COMPACT] Body keys: {list(body.keys())}")
    log(f"[COMPACT] Model: {body.get('model', 'NOT SET')}")
    log(f"[COMPACT] Instructions length: {len(body.get('instructions', '') or '')}")

    input_items = body.get("input", [])
    log(f"[COMPACT] Input items count: {len(input_items)}")
    for i, item in enumerate(input_items):
        item_type = item.get("type", "unknown")
        item_role = item.get("role", "none")
        content = item.get("content", "")
        if isinstance(content, str):
            preview = content[:200]
        elif isinstance(content, list):
            preview = json.dumps(content, default=str)[:200]
        else:
            preview = ""
        log(f"[COMPACT]   input[{i}]: type={item_type} role={item_role} content_preview={preview}")

    tools = body.get("tools", [])
    log(f"[COMPACT] Tools count: {len(tools)}")
    if tools:
        for i, tool in enumerate(tools[:5]):
            log(f"[COMPACT]   tool[{i}]: {tool.get('name', tool.get('type', 'unknown'))}")
        if len(tools) > 5:
            log(f"[COMPACT]   ... and {len(tools) - 5} more tools")

    if body.get("reasoning"):
        log(f"[COMPACT] Reasoning: {body['reasoning']}")
    if body.get("text"):
        log(f"[COMPACT] Text config: {body['text']}")
    log(f"{'='*60}")

    compact_model = body.get("model")
    target = await resolve_deployment(
        session, compact_model, is_admin=is_admin, wire="openai_responses"
    )
    reject_wrong_protocol(target, expected="openai_responses")

    azure_body = {
        "model": target.name,
        "input": input_items
        + [
            {
                "type": "message",
                "role": "user",
                "content": COMPACT_SYSTEM_PROMPT,
            }
        ],
        "store": False,
        "stream": False,
    }
    if body.get("instructions"):
        azure_body["instructions"] = body["instructions"]
    if tools:
        azure_body["tools"] = tools

    azure_url = target.upstream_url
    headers = {"api-key": target.api_key, "Content-Type": "application/json"}

    log(f"[COMPACT] Forwarding summarization request to upstream: {azure_url}")
    log(f"[COMPACT] Azure body keys: {list(azure_body.keys())}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(azure_url, headers=headers, json=azure_body)
            # Codex replays the whole conversation here, including reasoning items
            # whose encrypted blobs the current deployment instance can't decrypt.
            # Mirror the /v1/responses passthrough: strip reasoning items and retry once.
            if response.status_code == 400 and is_invalid_encrypted_content(response.content):
                retry_body, removed = strip_encrypted_reasoning(azure_body)
                if removed:
                    log(
                        f"[COMPACT] invalid_encrypted_content 400 — stripped "
                        f"{removed} reasoning item(s), retrying once"
                    )
                    response = await client.post(azure_url, headers=headers, json=retry_body)

        log(f"[COMPACT] Azure response status: {response.status_code}")

        if response.status_code != 200:
            log(f"[COMPACT] Azure error: {response.text[:1000]}")
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=response.status_code,
                error_type=f"azure_{response.status_code}",
                started_at=started_at,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Azure API error during compaction: {response.status_code}",
            )

        azure_result = response.json()
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=200, started_at=started_at,
            azure_usage=azure_result.get("usage", {}),
        )

        summary_text = ""
        for item in azure_result.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        summary_text += block.get("text", "")

        log(f"[COMPACT] Extracted summary length: {len(summary_text)}")
        log(f"[COMPACT] Summary preview: {summary_text[:500]}")

        compacted_output = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "SESSION RESUMPTION. You are continuing a prior session — "
                            "DO NOT re-introduce yourself, DO NOT acknowledge or reload "
                            "repository instructions, DO NOT restart the task. The summary "
                            "below IS your memory; treat it as your own prior work and "
                            "continue from exactly where it leaves off.\n\n" + summary_text
                        ),
                    }
                ],
            }
        ]

        log(f"[COMPACT] Response output items: {len(compacted_output)}")
        log(f"{'='*60}")

        return JSONResponse(content={"output": compacted_output})

    except httpx.ConnectError:
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=502, error_type="connect_error", started_at=started_at,
        )
        raise HTTPException(status_code=502, detail="Failed to reach Azure endpoint for compaction")
    except httpx.TimeoutException:
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=504, error_type="timeout", started_at=started_at,
        )
        raise HTTPException(status_code=504, detail="Azure endpoint timed out during compaction")
