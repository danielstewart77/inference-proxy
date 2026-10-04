"""Chat Completions endpoint.

The handler picks one of three upstream strategies based on the deployment's
target_uri:

  - `/openai/responses`  → translate Chat Completions ⇄ Responses API
  - `/v1/messages`       → translate Chat Completions ⇄ Anthropic Messages
                            (see `app/proxy/chat_to_anthropic.py`)
  - `/chat/completions`  → forward the body unchanged (native chat upstream,
                            e.g. Foundry Models-as-a-Service: DeepSeek, Kimi,
                            gpt-oss)

A caller on this endpoint therefore reaches every model the proxy serves
without knowing which vendor or wire is behind the name it asked for.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import usage
from app.auth import resolve_principal, resolve_requester_role, validate_api_key
from app.azure import post_with_retries
from app.config import config
from app.db import get_session
from app.deployments import (
    DeploymentTarget,
    build_upstream_headers,
    reject_wrong_protocol,
    resolve_deployment,
)
from app.proxy.anthropic import anthropic_upstream_headers, apply_oauth_system_prompt
from app.proxy.chat_to_anthropic import (
    stream_anthropic_to_completions,
    transform_anthropic_to_completions,
    transform_to_anthropic_messages,
)
from app.utils import log

router = APIRouter()
ENDPOINT_LABEL = "chat_completions"


# ---------- Translations (only used when upstream is /openai/responses) ----------


def _responses_tools(tools: object) -> list[dict]:
    """Chat Completions tool declarations as Responses API tool declarations.

    The Responses API flattens what Chat Completions nests: name, description
    and parameters sit on the tool itself rather than under `function`. A tool
    left in the nested shape is rejected, and a tool not copied at all is a
    turn the model answers in prose because it was never offered one.
    """
    out: list[dict] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        declared: dict = {
            "type": "function",
            "name": name,
            "parameters": fn.get("parameters")
            or fn.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        description = fn.get("description")
        if description:
            declared["description"] = description
        if "strict" in fn:
            declared["strict"] = fn["strict"]
        out.append(declared)
    return out


def _responses_tool_choice(tool_choice: object) -> object | None:
    """`tool_choice` in the Responses API's own spelling.

    The bare strings carry across unchanged; a named function loses the
    `function` wrapper, same as the declarations do.
    """
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice.lower() in ("auto", "required", "none") else None
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if name:
            return {"type": "function", "name": name}
    return None


def _responses_input_items(messages: list) -> list[dict]:
    """Non-system Chat Completions messages as Responses API input items.

    A tool round trip is two items rather than two messages: the assistant's
    call becomes a `function_call` carrying its arguments as the same JSON
    string it arrived as, and the caller's reply becomes a
    `function_call_output` keyed to the same `call_id`. Replaying them as
    chat messages instead drops both — the Responses API has no `tool` role
    and ignores `tool_calls` on a message — so a multi-step tool
    conversation never progresses past its first call.
    """
    items: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "tool":
            output = msg.get("content")
            if not isinstance(output, str):
                output = json.dumps(output) if output is not None else ""
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id") or msg.get("id") or "",
                    "output": output,
                }
            )
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content")
            if content:
                items.append({"role": "assistant", "content": content})
            for call in msg["tool_calls"]:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                arguments = fn.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": arguments,
                    }
                )
            continue

        items.append(msg)
    return items


def transform_to_responses_format(copilot_request: dict, streaming: bool, model: str) -> dict:
    """Chat Completions request -> Azure Responses API request body."""
    messages = copilot_request.get("messages", [])

    instructions: str | None = None
    conversation: list[dict] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") in ("system", "developer"):
            content = msg.get("content", "")
            instructions = f"{instructions}\n{content}" if instructions else content
        else:
            conversation.append(msg)

    azure_request: dict = {
        "model": model,
        "input": _responses_input_items(conversation),
        "stream": streaming,
        "store": False,
    }
    if instructions:
        azure_request["instructions"] = instructions

    tools = _responses_tools(copilot_request.get("tools"))
    if tools:
        azure_request["tools"] = tools
        choice = _responses_tool_choice(copilot_request.get("tool_choice"))
        if choice is not None:
            azure_request["tool_choice"] = choice

    param_map = {
        "temperature": "temperature",
        "max_tokens": "max_output_tokens",
        "top_p": "top_p",
        "stop": "stop",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
    }
    for src, dst in param_map.items():
        if src in copilot_request:
            azure_request[dst] = copilot_request[src]

    return azure_request


def _map_finish_reason(status: str) -> str:
    return {"completed": "stop", "incomplete": "length", "cancelled": "stop"}.get(status, "stop")


def _map_usage(responses_usage: dict) -> dict:
    return {
        "prompt_tokens": responses_usage.get("input_tokens", 0),
        "completion_tokens": responses_usage.get("output_tokens", 0),
        "total_tokens": responses_usage.get("total_tokens", 0),
    }


def transform_response_to_completions(azure_response: dict, model: str) -> dict:
    """Azure Responses API response body -> Chat Completions response body."""
    output_items = azure_response.get("output", []) or []

    tool_calls: list[dict] = []
    parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
        elif item.get("type") == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )

    assistant_content = azure_response.get("output_text")
    if assistant_content is None:
        assistant_content = "".join(parts)

    message: dict = {"role": "assistant", "content": assistant_content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = _map_finish_reason(azure_response.get("status", "completed"))
    if tool_calls and finish_reason == "stop":
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{azure_response.get('id', uuid.uuid4().hex[:29])}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _map_usage(azure_response.get("usage", {})),
    }


async def stream_responses_to_completions(
    response: httpx.Response, model: str, *, usage_holder: Optional[dict] = None
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    initial = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(initial)}\n\n"

    # Responses addresses a streamed tool call by its own output index; Chat
    # Completions numbers tool calls among themselves. Text and function-call
    # items interleave, so the two are counted separately or an SDK
    # reassembling by index sees gaps where the prose was.
    tool_index_for_item: dict[str, int] = {}
    next_tool_index = 0
    saw_tool_call = False
    finish_reason = "stop"

    async for line in response.aiter_lines():
        if not line.strip():
            continue
        if line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue

        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break

        try:
            event_data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        event_type = event_data.get("type", "")

        if event_type == "response.output_item.added":
            item = event_data.get("item") or {}
            if item.get("type") == "function_call":
                saw_tool_call = True
                index = next_tool_index
                next_tool_index += 1
                key = item.get("id") or item.get("call_id") or str(
                    event_data.get("output_index")
                )
                tool_index_for_item[key] = index
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": item.get("call_id") or item.get("id") or "",
                                        "type": "function",
                                        "function": {
                                            "name": item.get("name", ""),
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        elif event_type == "response.function_call_arguments.delta":
            key = event_data.get("item_id") or str(event_data.get("output_index"))
            index = tool_index_for_item.get(key)
            if index is None:
                index = next_tool_index
                next_tool_index += 1
                tool_index_for_item[key] = index
                saw_tool_call = True
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {
                                        "arguments": event_data.get("delta", "")
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        elif event_type == "response.output_text.delta":
            delta_text = event_data.get("delta", "")
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {"content": delta_text}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        elif event_type == "response.completed":
            if usage_holder is not None:
                u = event_data.get("response", {}).get("usage", {}) or {}
                usage_holder.update(u)
            status = (event_data.get("response") or {}).get("status", "completed")
            finish_reason = _map_finish_reason(status)
            if saw_tool_call and finish_reason == "stop":
                finish_reason = "tool_calls"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish_reason}
                ],
            }
            yield f"data: {json.dumps(final)}\n\n"

    yield "data: [DONE]\n\n"


# ---------- Routes ----------


@router.post("/openai/deployments/{deployment}/chat/completions")
async def chat_completions_azure_style(
    deployment: str,
    request: Request,
    authorization: str | None = Header(None),
    api_key: str | None = Header(None, alias="api-key"),
    session: AsyncSession = Depends(get_session),
):
    return await _handle_completions(
        request, authorization, deployment, api_key, session=session
    )


@router.post("/v1/chat/completions")
async def chat_completions_openai_style(
    request: Request,
    authorization: str | None = Header(None),
    api_key: str | None = Header(None, alias="api-key"),
    session: AsyncSession = Depends(get_session),
):
    return await _handle_completions(
        request, authorization, api_key_header=api_key, session=session
    )


@router.post("/openai/v1/chat/completions")
async def chat_completions_vs_style(
    request: Request,
    authorization: str | None = Header(None),
    api_key: str | None = Header(None, alias="api-key"),
    session: AsyncSession = Depends(get_session),
):
    return await _handle_completions(
        request, authorization, api_key_header=api_key, session=session
    )


async def _handle_completions(
    request: Request,
    authorization: str | None,
    deployment_override: str | None = None,
    api_key_header: str | None = None,
    *,
    session: AsyncSession,
):
    started_at = time.time()
    if not validate_api_key(authorization, api_key_header):
        raise HTTPException(
            status_code=401 if not (authorization or api_key_header) else 403,
            detail="Invalid API key",
        )
    principal = resolve_principal(authorization, api_key_header)
    is_admin = resolve_requester_role(authorization, api_key_header) == "admin"

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")

    model = deployment_override or body.get("model")
    target = await resolve_deployment(
        session, model, is_admin=is_admin, wire="chat_completions"
    )
    reject_wrong_protocol(target, expected="chat_completions_any")

    is_streaming = body.get("stream", False) and config.enable_streaming

    log(f"Received request for model: {target.name}")
    log(f"Request type: {'streaming' if is_streaming else 'non-streaming'}")
    log(f"Forwarding to: {target.upstream_url}")

    headers = build_upstream_headers(target)

    try:
        if target.is_anthropic_messages:
            anthropic_body = transform_to_anthropic_messages(
                body, streaming=is_streaming, model=target.name
            )
            if target.is_anthropic_oauth:
                anthropic_body = apply_oauth_system_prompt(anthropic_body)
            anthropic_headers = anthropic_upstream_headers(target)
            if is_streaming:
                return await _stream_via_anthropic_translation(
                    target.upstream_url, anthropic_headers, anthropic_body, target,
                    principal=principal, request=request, started_at=started_at,
                )
            return await _post_via_anthropic_translation(
                target.upstream_url, anthropic_headers, anthropic_body, target,
                principal=principal, request=request, started_at=started_at,
            )

        if target.is_openai_responses:
            azure_body = transform_to_responses_format(
                body, streaming=is_streaming, model=target.name
            )
            if is_streaming:
                return await _stream_via_responses_translation(
                    target.upstream_url, headers, azure_body, target,
                    principal=principal, request=request, started_at=started_at,
                )
            return await _post_via_responses_translation(
                target.upstream_url, headers, azure_body, target,
                principal=principal, request=request, started_at=started_at,
            )

        # Native Chat Completions upstream — passthrough.
        passthrough_body = dict(body)
        passthrough_body["model"] = target.name
        passthrough_body["stream"] = is_streaming
        if is_streaming:
            return await _stream_native_chat_passthrough(
                target.upstream_url, headers, passthrough_body, target,
                principal=principal, request=request, started_at=started_at,
            )
        return await _post_native_chat_passthrough(
            target.upstream_url, headers, passthrough_body, target,
            principal=principal, request=request, started_at=started_at,
        )

    except httpx.ConnectError as exc:
        log(f"[Chat API] Returning 502 to client after connect failure: {exc!r}")
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


# ---------- Translation path (Chat ⇄ Responses) ----------


async def _post_via_responses_translation(
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
        url, headers, body, stream=False, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        log(f"Upstream error {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _upstream_error_response(response.status_code, response.content)

    azure_result = response.json()
    log(f"Response received: {len(response.content)} bytes")
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage=azure_result.get("usage", {}),
    )
    return JSONResponse(
        content=transform_response_to_completions(azure_result, target.name)
    )


async def _stream_via_responses_translation(
    url: str,
    headers: dict,
    body: dict,
    target: DeploymentTarget,
    *,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> StreamingResponse | JSONResponse:
    response = await post_with_retries(
        url, headers, body, stream=True, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        error_body = await response.aread()
        await response.aclose()
        log(f"Upstream error {response.status_code}: {error_body[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _upstream_error_response(response.status_code, error_body)

    captured: dict = {}

    async def generate():
        try:
            async for chunk in stream_responses_to_completions(
                response, target.name, usage_holder=captured
            ):
                yield chunk
        finally:
            await response.aclose()
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=200, started_at=started_at,
                azure_usage=captured,
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------- Translation path (Chat ⇄ Anthropic Messages) ----------


def _anthropic_usage_to_azure_usage(anthropic_usage: dict) -> dict:
    """Anthropic's `usage` block in the shape `usage.schedule` expects."""
    inp = anthropic_usage.get("input_tokens")
    out = anthropic_usage.get("output_tokens")
    total = inp + out if isinstance(inp, int) and isinstance(out, int) else None
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "cache_creation_input_tokens": anthropic_usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": anthropic_usage.get("cache_read_input_tokens"),
    }


def _upstream_error_response(status_code: int, raw: bytes | str) -> JSONResponse:
    """The upstream's own words, at the upstream's own status.

    Anthropic refuses in its own envelope (`{"type": "error", "error": {...}}`),
    which an OpenAI client cannot read a message out of — it looks under
    `error.message`, and Anthropic puts the sentence exactly there. So the
    body passes through untouched; only a body that is not JSON at all is
    wrapped, and then in the shape the caller's SDK reads.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
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


async def _post_via_anthropic_translation(
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
        url, headers, body, stream=False, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        log(f"Upstream error {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _upstream_error_response(response.status_code, response.content)

    result = response.json()
    log(f"Response received: {len(response.content)} bytes")
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage=_anthropic_usage_to_azure_usage(result.get("usage", {}) or {}),
    )
    return JSONResponse(
        content=transform_anthropic_to_completions(result, target.name)
    )


async def _stream_via_anthropic_translation(
    url: str,
    headers: dict,
    body: dict,
    target: DeploymentTarget,
    *,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> StreamingResponse | JSONResponse:
    response = await post_with_retries(
        url, headers, body, stream=True, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        error_body = await response.aread()
        await response.aclose()
        log(f"Upstream error {response.status_code}: {error_body[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        return _upstream_error_response(response.status_code, error_body)

    captured: dict = {}

    async def generate():
        try:
            async for chunk in stream_anthropic_to_completions(
                response, target.name, usage_holder=captured
            ):
                yield chunk
        finally:
            await response.aclose()
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=200, started_at=started_at,
                azure_usage=_anthropic_usage_to_azure_usage(captured),
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------- Native passthrough path (upstream already speaks Chat Completions) ----------


def _chat_usage_to_azure_usage(chat_usage: dict) -> dict:
    """Convert OpenAI Chat Completions `usage` shape to the
    {input_tokens, output_tokens, total_tokens} shape usage.schedule expects.
    Cached prompt tokens (a subset of prompt_tokens, billed at a discount)
    ride along so the metering row records them."""
    details = chat_usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": chat_usage.get("prompt_tokens"),
        "output_tokens": chat_usage.get("completion_tokens"),
        "total_tokens": chat_usage.get("total_tokens"),
        "cache_read_input_tokens": details.get("cached_tokens"),
    }


async def _post_native_chat_passthrough(
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
        url, headers, body, stream=False, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        log(f"Upstream error {response.status_code}: {response.text[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        try:
            return JSONResponse(status_code=response.status_code, content=response.json())
        except ValueError:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Upstream API error: {response.status_code}",
            )

    log(f"Response received: {len(response.content)} bytes")
    result = response.json()
    usage.schedule(
        principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
        status_code=200, started_at=started_at,
        azure_usage=_chat_usage_to_azure_usage(result.get("usage", {}) or {}),
    )
    return JSONResponse(content=result)


async def _stream_native_chat_passthrough(
    url: str,
    headers: dict,
    body: dict,
    target: DeploymentTarget,
    *,
    principal: Optional[tuple[int, int]] = None,
    request: Optional[Request] = None,
    started_at: Optional[float] = None,
) -> StreamingResponse:
    # Ask for the usage chunk when the caller didn't — without it, streamed
    # chat completions carry no token counts and the metering row is empty.
    # Standard OpenAI shape; Ollama and Foundry MaaS both honour it.
    if "stream_options" not in body:
        body = {**body, "stream_options": {"include_usage": True}}
    response = await post_with_retries(
        url, headers, body, stream=True, log_prefix="[Chat API]"
    )

    if response.status_code != 200:
        error_body = await response.aread()
        await response.aclose()
        log(f"Upstream error {response.status_code}: {error_body[:500]}")
        usage.schedule(
            principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
            status_code=response.status_code,
            error_type=f"upstream_{response.status_code}",
            started_at=started_at,
        )
        try:
            content = json.loads(error_body)
        except (ValueError, json.JSONDecodeError):
            content = {"error": {"message": f"Upstream API error: {response.status_code}"}}
        return JSONResponse(status_code=response.status_code, content=content)

    captured: dict = {}

    async def generate():
        try:
            async for line in response.aiter_lines():
                yield f"{line}\n"
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                payload = stripped[5:].lstrip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Some providers (incl. Foundry MaaS) include usage in the
                # final chunk when stream_options.include_usage is set.
                if isinstance(evt, dict) and evt.get("usage"):
                    captured.update(evt["usage"])
        finally:
            await response.aclose()
            usage.schedule(
                principal, request=request, model=target.name, endpoint=ENDPOINT_LABEL,
                status_code=200, started_at=started_at,
                azure_usage=_chat_usage_to_azure_usage(captured),
            )

    return StreamingResponse(generate(), media_type="text/event-stream")
