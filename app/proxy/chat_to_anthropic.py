"""Chat Completions ⇄ Anthropic Messages translation.

The third translation the chat route can reach for. A caller that speaks only
Chat Completions — dsh's `llm-pi-ai` provider route, any OpenAI SDK — names a
Claude deployment and the proxy rewrites the request onto the Anthropic
Messages wire, then rewrites the reply back. No vendor knowledge lives in the
caller: the deployment's own `target_uri` is what decides which of the three
upstream shapes it gets.

Anthropic Messages differs from Chat Completions in every structural way that
matters here:

  - `system` is hoisted out of the message list into its own top-level field
  - content is a list of typed blocks, not a flat string
  - a tool call is a `tool_use` block on the assistant turn; its result is a
    `tool_result` block on a *user* turn, keyed by `tool_use_id`
  - a tool declares its schema as `input_schema`, not `parameters`
  - `max_tokens` is mandatory
  - the event stream is block-oriented (`content_block_delta`), not
    choice-oriented (`choices[].delta`)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator, Optional

import httpx

class UntranslatableRequest(ValueError):
    """The caller sent something this wire has no equivalent for.

    Raised rather than worked around, because every alternative is worse: a
    part quietly dropped produces a 200 answering the wrong question, and a
    part forwarded as-is produces an upstream 400 whose sentence names an
    Anthropic field the caller never sent. The route turns this into a 400 in
    the caller's own envelope, naming the thing it actually sent.
    """


#: Anthropic rejects a body with no `max_tokens`, so one has to be supplied
#: when the caller omitted it. Every current Claude model accepts this, which
#: a larger default would not be true of.
DEFAULT_MAX_TOKENS = 8192

#: `thinking.budget_tokens` per OpenAI-style reasoning effort. Anthropic's
#: floor is 1024; `off` and `none` mean no extended thinking at all, which is
#: an absent block rather than a zero budget.
THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
}

#: `max_tokens` has to exceed `budget_tokens` or the request is refused, so a
#: thinking budget raises the ceiling to leave room for a reply behind it.
THINKING_HEADROOM = 4096

_FINISH_FOR_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    # A refused generation is not a finished one. Reported as `stop` it is an
    # empty successful answer, which the caller has no reason to retry or
    # surface.
    "refusal": "content_filter",
}


# ---------- Request: Chat Completions -> Anthropic Messages ----------


#: Both this proxy's own translations and the SDKs calling it spell a text
#: part three ways. A part whose spelling is not recognised still carries its
#: text under a known key, and dropping it loses the turn's whole body.
TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})


def _text_of(content: object) -> str:
    """Flatten a Chat Completions `content` down to its text.

    Used for the system prompt and for a tool result, both of which Anthropic
    takes as text rather than as a part list.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES:
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    if content is None:
        return ""
    # A dict or a number is JSON on the way out, never a Python repr: `str`
    # on a dict yields single quotes and `True`/`None`, which the model then
    # misreads. The Responses translation already does this; a tool result
    # must not mean two different things depending on which wire answers.
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _image_block(part: dict) -> Optional[dict]:
    """An image part as an Anthropic image block.

    A data URL carries the bytes and the media type inline, which is the form
    every harness attaching a local file produces. A plain http(s) URL is
    handed over as a `url` source — Anthropic fetches it itself. A part that
    already carries an Anthropic `source` is passed through, since a caller
    may legitimately send the target wire's own spelling.

    An image that cannot be read is the one failure worth refusing locally:
    dropping it leaves a valid request that returns 200 and an answer about
    a picture the model never saw, which reads as correct and is not.
    """
    source = part.get("source")
    if isinstance(source, dict) and source.get("type"):
        return {"type": "image", "source": source}
    url = part.get("image_url") or part.get("url")
    if isinstance(url, dict):
        url = url.get("url")
    if not isinstance(url, str) or not url:
        raise UntranslatableRequest("an image part carries no url or source")
    if url.startswith("data:"):
        header, sep, data = url.partition(",")
        if not sep or not data:
            raise UntranslatableRequest("an image data url carries no data")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content: object) -> list[dict]:
    """A user or assistant `content` as a list of Anthropic blocks.

    A part whose `type` is none of the spellings known here still has its
    text kept, under whichever key holds it. Discarding it instead produces
    the worst available outcome: a structurally valid request that the
    upstream answers 200 to, with the turn's body missing — so the model
    answers a question it was never asked and nothing anywhere reports a
    problem.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        text = _text_of(content)
        return [{"type": "text", "text": text}] if text else []
    blocks: list[dict] = []
    for part in content:
        if isinstance(part, str):
            if part:
                blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in IMAGE_PART_TYPES or "image_url" in part:
            blocks.append(_image_block(part))
            continue
        text = part.get("text")
        if not isinstance(text, str):
            text = part.get("content") if isinstance(part.get("content"), str) else None
        if text:
            blocks.append({"type": "text", "text": text})
        elif kind not in TEXT_PART_TYPES:
            raise UntranslatableRequest(
                f"a content part of type {kind!r} carries no text this wire can send"
            )
    return blocks


def _tool_use_blocks(tool_calls: object) -> list[dict]:
    """Assistant `tool_calls` as Anthropic `tool_use` blocks.

    Arguments arrive as a JSON *string* on this wire and have to land as a
    real object. A string that will not parse is carried as-is under a single
    key rather than dropped — a malformed call the model made is evidence,
    and silently losing it turns a bad call into no call at all.
    """
    blocks: list[dict] = []
    if not isinstance(tool_calls, list):
        return blocks
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            fn = call if call.get("name") else {}
        raw = fn.get("arguments")
        if isinstance(raw, (dict, list)):
            parsed: object = raw
        else:
            try:
                parsed = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                parsed = {"__raw_arguments": raw}
        if not isinstance(parsed, dict):
            parsed = {"__raw_arguments": parsed}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name") or call.get("name") or "",
                "input": parsed,
            }
        )
    return blocks


def _tool_result_block(message: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": message.get("tool_call_id") or message.get("id") or "",
        "content": _text_of(message.get("content")),
    }


def _translate_tools(tools: object) -> list[dict]:
    """Chat Completions tool declarations as Anthropic tool declarations.

    The schema moves from `function.parameters` to `input_schema`, which is
    the whole reason a tool offered on this wire reaches a Claude model at
    all. The flat form (name and parameters at the top level) is accepted too
    — some SDKs still emit it, and refusing it would drop the tool.
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
            # A provider-native tool (`web_search_preview` and friends) has no
            # function name and no Anthropic equivalent. Dropping it silently
            # would leave the model answering in prose for no stated reason.
            raise UntranslatableRequest(
                f"tool of type {tool.get('type')!r} has no function name to translate"
            )
        schema = fn.get("parameters") or fn.get("input_schema")
        if isinstance(schema, str):
            # Some callers serialise the schema before sending it. Forwarded
            # as a string it is not a schema at all and the tool is refused.
            try:
                schema = json.loads(schema)
            except (TypeError, ValueError):
                schema = None
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        declared: dict = {"name": name, "input_schema": schema}
        description = fn.get("description")
        if description:
            declared["description"] = description
        out.append(declared)
    return out


def _translate_tool_choice(tool_choice: object) -> Optional[dict]:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        lowered = tool_choice.lower()
        if lowered == "auto":
            return {"type": "auto"}
        if lowered in ("required", "any"):
            return {"type": "any"}
        if lowered == "none":
            return {"type": "none"}
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function" or "function" in tool_choice:
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else fn) or tool_choice.get(
                "name"
            )
            if isinstance(name, str) and name:
                return {"type": "tool", "name": name}
            return None
        kind = tool_choice.get("type")
        if kind in ("auto", "any", "none"):
            return {"type": kind}
        if kind == "tool" and tool_choice.get("name"):
            return {"type": "tool", "name": tool_choice["name"]}
    return None


def _thinking_for(effort: object) -> Optional[dict]:
    if not isinstance(effort, str):
        return None
    budget = THINKING_BUDGETS.get(effort.strip().lower())
    if not budget:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def transform_to_anthropic_messages(
    chat_request: dict, *, streaming: bool, model: str
) -> dict:
    """Chat Completions request body -> Anthropic Messages request body."""
    system_text = ""
    messages: list[dict] = []

    for message in chat_request.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")

        if role in ("system", "developer"):
            text = _text_of(message.get("content"))
            if text:
                system_text = f"{system_text}\n{text}" if system_text else text
            continue

        if role == "tool":
            block = _tool_result_block(message)
            # Anthropic wants every result for one assistant turn on a single
            # user turn, so consecutive tool replies merge rather than each
            # opening a turn of its own.
            if messages and messages[-1]["role"] == "user" and all(
                b.get("type") == "tool_result" for b in messages[-1]["content"]
            ):
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks = _content_blocks(message.get("content"))
            blocks.extend(_tool_use_blocks(message.get("tool_calls")))
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue

        blocks = _content_blocks(message.get("content"))
        if blocks:
            messages.append({"role": "user", "content": blocks})

    max_tokens = (
        chat_request.get("max_tokens")
        or chat_request.get("max_completion_tokens")
        or DEFAULT_MAX_TOKENS
    )

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if system_text:
        body["system"] = system_text

    thinking = _thinking_for(chat_request.get("reasoning_effort"))
    if thinking is None:
        reasoning = chat_request.get("reasoning")
        if isinstance(reasoning, dict):
            thinking = _thinking_for(reasoning.get("effort"))
    if thinking:
        body["thinking"] = thinking
        needed = thinking["budget_tokens"] + THINKING_HEADROOM
        if body["max_tokens"] < needed:
            body["max_tokens"] = needed
        # Extended thinking fixes the sampling parameters; sending either
        # alongside it is a 400.
        chat_request = {
            k: v for k, v in chat_request.items() if k not in ("temperature", "top_p")
        }

    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if src in chat_request:
            body[dst] = chat_request[src]

    stop = chat_request.get("stop")
    if isinstance(stop, str):
        # Guarded the same way the list form is: an SDK defaulting `stop` to
        # the empty string would otherwise send a stop sequence Anthropic
        # refuses, and no Claude model would be reachable at all.
        if stop:
            body["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        body["stop_sequences"] = stop

    tools = _translate_tools(chat_request.get("tools"))
    if tools:
        body["tools"] = tools
        choice = _translate_tool_choice(chat_request.get("tool_choice"))
        if choice:
            body["tool_choice"] = choice

    return body


# ---------- Response: Anthropic Messages -> Chat Completions ----------


def _usage_to_chat(anthropic_usage: dict) -> dict:
    inp = anthropic_usage.get("input_tokens") or 0
    out = anthropic_usage.get("output_tokens") or 0
    usage: dict = {
        "prompt_tokens": inp,
        "completion_tokens": out,
        "total_tokens": inp + out,
    }
    cached = anthropic_usage.get("cache_read_input_tokens")
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def transform_anthropic_to_completions(payload: dict, model: str) -> dict:
    """Anthropic Messages response body -> Chat Completions response body."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)

    stop_reason = payload.get("stop_reason")
    finish_reason = _FINISH_FOR_STOP_REASON.get(stop_reason or "", "stop")
    if tool_calls and stop_reason in (None, "tool_use"):
        finish_reason = "tool_calls"

    return {
        "id": payload.get("id") or f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": _usage_to_chat(payload.get("usage") or {}),
    }


async def stream_anthropic_to_completions(
    response: httpx.Response, model: str, *, usage_holder: Optional[dict] = None
) -> AsyncIterator[str]:
    """Anthropic's block-oriented event stream as Chat Completions chunks.

    Anthropic opens a block, streams deltas into it and closes it; Chat
    Completions has one flat delta per chunk with tool calls addressed by
    their own index. So an index per *content block* is translated into an
    index per *tool call*, counted separately — text and tool blocks
    interleave, and a tool call numbered by its block index would leave gaps
    the SDK reads as missing calls.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    captured: dict = {}

    def chunk(delta: dict, finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""})

    tool_index_for_block: dict[int, int] = {}
    next_tool_index = 0
    saw_tool_use = False
    finish_reason = "stop"
    # A terminator is only earned. Anthropic ends a turn with `message_delta`
    # carrying a stop reason and then `message_stop`; a stream that simply
    # stops arriving — a dropped connection, an upstream that died — reaches
    # the end of iteration exactly as a finished one does. Emitting the
    # success chunk and `[DONE]` regardless is how a tool call truncated
    # mid-arguments is handed to the caller as a complete call with
    # unparseable JSON, and a severed answer as the whole answer.
    ended = False
    failed: Optional[dict] = None

    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("data:"):
            continue
        data_str = stripped[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        kind = event.get("type")

        if kind == "message_start":
            captured.update((event.get("message") or {}).get("usage") or {})

        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                saw_tool_use = True
                index = next_tool_index
                next_tool_index += 1
                tool_index_for_block[event.get("index")] = index
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )

        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    yield chunk({"content": text})
            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking", "")
                if thinking:
                    yield chunk({"reasoning_content": thinking})
            elif delta_type == "input_json_delta":
                index = tool_index_for_block.get(event.get("index"))
                if index is None:
                    continue
                yield chunk(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                },
                            }
                        ]
                    }
                )

        elif kind == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason")
            if stop_reason:
                finish_reason = _FINISH_FOR_STOP_REASON.get(stop_reason, "stop")
                ended = True
            usage_delta = event.get("usage") or {}
            if usage_delta:
                captured.update(usage_delta)

        elif kind == "message_stop":
            ended = True

        elif kind == "error":
            # The upstream failed mid-stream, after a 200 and after bytes have
            # already reached the caller. The status is spent, so the only
            # honest thing left is to pass its own words through in the shape
            # an OpenAI client reads an error out of — and then stop, rather
            # than following it with a terminator that says the turn
            # completed.
            failed = event.get("error") or {"message": "upstream stream error"}
            break

    if usage_holder is not None:
        usage_holder.update(captured)

    if failed is not None:
        yield f"data: {json.dumps({'error': failed})}\n\n"
        return

    if not ended:
        yield f"data: {json.dumps({'error': {'type': 'api_error', 'message': 'Upstream stream ended before the turn completed'}})}\n\n"
        return

    if saw_tool_use and finish_reason == "stop":
        finish_reason = "tool_calls"

    yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"
