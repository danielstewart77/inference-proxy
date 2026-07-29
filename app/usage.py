"""Per-request usage logging for /v1/* proxied calls."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from fastapi import Request

from app.db import SessionLocal
from app.orm import UsageLog


def client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def client_ip_from_ws(websocket) -> Optional[str]:
    fwd = websocket.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return websocket.client.host if websocket.client else None


def tokens_from_usage(azure_usage: dict) -> dict:
    """Normalize a provider `usage` dict into the columns `_insert` writes.

    Accepts the flat {input_tokens, output_tokens, total_tokens,
    cache_creation_input_tokens, cache_read_input_tokens} shape the route
    modules produce, plus the raw OpenAI shapes: Responses API nests cached
    reads in `input_tokens_details.cached_tokens`, Chat Completions in
    `prompt_tokens_details.cached_tokens`. Values are stored as delivered —
    Anthropic's input excludes cached tokens, OpenAI's includes them (see
    the `UsageLog` column comment).
    """
    cache_read = azure_usage.get("cache_read_input_tokens")
    if cache_read is None:
        details = (
            azure_usage.get("input_tokens_details")
            or azure_usage.get("prompt_tokens_details")
            or {}
        )
        cache_read = details.get("cached_tokens")
    return {
        "input_tokens": azure_usage.get("input_tokens"),
        "output_tokens": azure_usage.get("output_tokens"),
        "total_tokens": azure_usage.get("total_tokens"),
        "cache_creation_input_tokens": azure_usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": cache_read,
    }


async def _insert(
    *,
    client_id: int,
    key_id: Optional[int],
    model_name: str,
    endpoint: str,
    status_code: int,
    error_type: Optional[str],
    duration_ms: Optional[int],
    client_ip_str: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cache_creation_input_tokens: Optional[int],
    cache_read_input_tokens: Optional[int],
) -> None:
    try:
        async with SessionLocal() as session:
            session.add(
                UsageLog(
                    client_id=client_id,
                    key_id=key_id,
                    model_name=model_name,
                    endpoint=endpoint,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cache_creation_input_tokens=cache_creation_input_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    status_code=status_code,
                    error_type=error_type,
                    duration_ms=duration_ms,
                    client_ip=client_ip_str,
                )
            )
            await session.commit()
    except Exception as e:
        # Don't let logging errors take down request processing.
        print(f"[usage] failed to insert usage_log row: {type(e).__name__}: {e}", flush=True)


def schedule(
    principal: Optional[tuple[int, int]],
    *,
    request: Optional[Request] = None,
    websocket=None,
    model: str,
    endpoint: str,
    status_code: int,
    error_type: Optional[str] = None,
    started_at: Optional[float] = None,
    azure_usage: Optional[dict] = None,
) -> None:
    """Fire-and-forget log insert. No-op if principal is None.

    `azure_usage` is the dict from the upstream's `usage` field
    ({input_tokens, output_tokens, total_tokens, ...}); pass None on errors
    or when usage data isn't available.
    """
    if principal is None:
        return
    client_id, key_id = principal

    if azure_usage is None:
        azure_usage = {}

    duration_ms: Optional[int] = None
    if started_at is not None:
        duration_ms = int((time.time() - started_at) * 1000)

    if request is not None:
        ip = client_ip(request)
    elif websocket is not None:
        ip = client_ip_from_ws(websocket)
    else:
        ip = None

    tokens = tokens_from_usage(azure_usage)
    asyncio.create_task(
        _insert(
            client_id=client_id,
            key_id=key_id,
            model_name=model,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type,
            duration_ms=duration_ms,
            client_ip_str=ip,
            **tokens,
        )
    )
