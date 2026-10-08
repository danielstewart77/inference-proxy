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
#: thinking budget needs room for a reply behind it.
THINKING_HEADROOM = 4096

#: Anthropic's own floor. A budget under this is refused, so an effort that
#: cannot be fitted beneath the caller's own ceiling is not sent at all.
MIN_THINKING_BUDGET = 1024

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


def _reasoning_blocks(details: object) -> list[dict]:
    """Thinking blocks handed back by a caller replaying an assistant turn."""
    if not isinstance(details, list):
        return []
    return [
        block
        for block in details
        if isinstance(block, dict)
        and block.get("type") in ("thinking", "redacted_thinking")
    ]


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
            # Anthropic requires the signed thinking blocks back on an
            # assistant turn that carries `tool_use` while thinking is
            # enabled, and refuses the follow-up without them. The signature
            # is opaque bytes with no Chat Completions field of its own, so
            # the whole block list rides out and back under
            # `reasoning_details` — otherwise "reasoning effort plus tools"
            # is a one-turn conversation by construction, and the caller is
            # not even handed the bytes it would need to fix it.
            blocks = _reasoning_blocks(message.get("reasoning_details"))
            blocks.extend(_content_blocks(message.get("content")))
            blocks.extend(_tool_use_blocks(message.get("tool_calls")))
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue

        blocks = _content_blocks(message.get("content"))
        if blocks:
            messages.append({"role": "user", "content": blocks})

    ceiling = chat_request.get("max_tokens") or chat_request.get(
        "max_completion_tokens"
    )

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": ceiling or DEFAULT_MAX_TOKENS,
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
        budget = thinking["budget_tokens"]
        if ceiling:
            # A caller that named its own ceiling named it for a reason — a
            # cost cap, or a buffer downstream. Raising it to fit a thinking
            # budget it never asked for spends many times what it authorised,
            # silently. So the budget is fitted under the ceiling instead, and
            # where it cannot reach Anthropic's floor the effort is dropped
            # rather than the ceiling.
            budget = min(budget, ceiling - THINKING_HEADROOM)
            if budget < MIN_THINKING_BUDGET:
                budget = 0
        else:
            body["max_tokens"] = max(body["max_tokens"], budget + THINKING_HEADROOM)
        if budget:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Extended thinking fixes the sampling parameters; sending either
            # alongside it is a 400. Only when thinking survived, though — a
            # dropped effort must not take the caller's temperature with it.
            chat_request = {
                k: v
                for k, v in chat_request.items()
                if k not in ("temperature", "top_p")
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


#: The fields that make up an OpenAI prompt total. If none of them is known,
#: there is no prompt figure to report — only a zero that would read as one.
_PROMPT_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _as_count(value: object) -> Optional[int]:
    """One reported token count as an int, or nothing if it is not one.

    `bool` is excluded deliberately: it is an `int` subclass in Python, so a
    stray `true` would be recorded as a count of one and stored in an integer
    column. Floats and digit strings are accepted, because a shim that
    JSON-encodes its counts that way is otherwise dropped entirely — and a
    dropped count does not read as missing, it reads as zero.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _prompt_total(captured: dict) -> Optional[int]:
    """The inclusive prompt total, or nothing if no part of it was reported.

    Emptiness is the wrong question to ask of `captured`. An upstream that
    reports its prompt fields as null while reporting a real output count
    leaves a populated dict whose prompt side is pure fabrication — and a
    consumer cannot tell a fabricated zero from a turn that genuinely used no
    context, so it overwrites whatever it already knew with the zero. The
    honest answer to "how big was the prompt" is sometimes "I was not told".
    """
    known = [_as_count(captured.get(f)) for f in _PROMPT_FIELDS]
    if all(k is None for k in known):
        return None
    total = sum(k for k in known if k is not None)
    return total or None


def _usage_to_chat(anthropic_usage: dict) -> Optional[dict]:
    """Anthropic's usage block as an OpenAI `usage` object.

    The two wires disagree about what the prompt total means, and the
    disagreement is silent. Anthropic's `input_tokens` **excludes** the cached
    prefix — cache reads and cache creation are billed separately — while
    OpenAI's `prompt_tokens` **includes** it, with the cached portion named as
    a subset (the contract this repo states at `app/orm.py:287-290`). A
    consumer recovers fresh input by subtracting the named subsets back out, so
    handing it Anthropic's exclusive figure makes a cache-heavy turn's input
    read as zero, and a created prefix with no wire field at all simply
    vanishes — on the first turn of a conversation that is most of the context.
    """
    prompt = _prompt_total(anthropic_usage)
    if prompt is None:
        return None
    out = _as_count(anthropic_usage.get("output_tokens")) or 0
    usage: dict = {
        "prompt_tokens": prompt,
        "completion_tokens": out,
        "total_tokens": prompt + out,
    }
    details = {}
    cached = _as_count(anthropic_usage.get("cache_read_input_tokens"))
    created = _as_count(anthropic_usage.get("cache_creation_input_tokens"))
    if cached is not None:
        details["cached_tokens"] = cached
    if created is not None:
        details["cache_write_tokens"] = created
    if details:
        usage["prompt_tokens_details"] = details
    return usage


def transform_anthropic_to_completions(payload: dict, model: str) -> dict:
    """Anthropic Messages response body -> Chat Completions response body."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    thinking_blocks: list[dict] = []
    tool_calls: list[dict] = []

    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(block.get("text", ""))
        elif kind in ("thinking", "redacted_thinking"):
            thinking_blocks.append(block)
            thinking_parts.append(block.get("thinking") or "")
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
    if thinking_blocks:
        # Verbatim, signatures included: this is what the next turn has to
        # hand back for Anthropic to accept it.
        message["reasoning_details"] = thinking_blocks
    reasoning = "".join(thinking_parts)
    if reasoning:
        # Anthropic routinely returns a thinking block whose text is empty or
        # redacted. Surfacing the key anyway tells the caller there is
        # reasoning to read when there is not.
        message["reasoning_content"] = reasoning

    stop_reason = payload.get("stop_reason")
    finish_reason = _FINISH_FOR_STOP_REASON.get(stop_reason or "", "stop")
    if tool_calls and stop_reason in (None, "tool_use"):
        finish_reason = "tool_calls"

    upstream_id = payload.get("id")
    return {
        "id": f"chatcmpl-{upstream_id}" if upstream_id else f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": _usage_to_chat(payload.get("usage") or {}),
    }


async def stream_anthropic_to_completions(
    response: httpx.Response,
    model: str,
    *,
    usage_holder: Optional[dict] = None,
    outcome: Optional[dict] = None,
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

    def record_usage(reported: object) -> None:
        """Fold one event's usage into what we know, and publish it.

        Not a blind `update`. Anthropic reports the input side on
        `message_start` and the output side on `message_delta`, but an upstream
        — or a partner shim on this same route — may send a *whole* usage
        object on the later event with the fields it has nothing to say about
        zeroed. A plain merge lets that erase a count the first event got
        right, and because the metering row reads this same dict, the row and
        the caller would then agree perfectly on a wrong number.

        Published per event rather than once after the loop because there is no
        `try` around that loop: a transport error or a caller hanging up skips
        everything after it, and nothing can be yielded while a generator is
        being closed. The row survives even when no chunk can.
        """
        if not isinstance(reported, dict):
            return
        for key, raw in reported.items():
            count = _as_count(raw)
            if count is None:
                continue
            # High-water mark, because every counter on this wire is cumulative
            # and final — none of them legitimately decreases. That covers a
            # later event re-reporting the whole block with zeros, and also the
            # sentinel: Anthropic's own `message_start` carries
            # `output_tokens: 1` as a placeholder, so a rule that refused only
            # literal zeros would let that 1 stand over a real count, with the
            # metering row and the caller agreeing on it.
            previous = captured.get(key)
            captured[key] = count if previous is None else max(previous, count)
        if usage_holder is not None:
            usage_holder.update(captured)

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

    try:
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
                record_usage((event.get("message") or {}).get("usage"))

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
                record_usage(event.get("usage"))

            elif kind == "message_stop":
                # Anthropic reports on `message_start` and `message_delta`, but
                # shims on this same route attach final metrics here, and a count
                # ignored on this event is a turn that reads as unmeasured on both
                # the chunk and the metering row.
                record_usage(event.get("usage"))
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
    except BaseException:
        # The loop is the only place counts are gathered, and two ways out of
        # it reach none of the code below: the transport raising mid-stream,
        # and the caller hanging up, which arrives as GeneratorExit at a
        # yield. Neither can yield anything, so the metering row is all that
        # is left — and a row carrying full counts at status 200 with no
        # error named is indistinguishable from a turn that was served. The
        # counts are worth keeping; staying silent about how it ended is not,
        # because an empty token column was the only way an operator spotted
        # these before they carried numbers.
        if outcome is not None and not outcome.get("error_type"):
            outcome["error_type"] = "upstream_stream_aborted"
        raise

    # Counts only when the upstream actually reported some. `_usage_to_chat({})`
    # is a well-formed object of zeros, and a consumer overwrites its running
    # usage from every chunk that carries one, so a fabricated zero erases a
    # real measurement rather than adding nothing. A reader that is told
    # nothing can say "unmeasured"; one told zero cannot tell that from a turn
    # that genuinely used no context.
    def usage_only_chunk() -> Optional[str]:
        """The counts on their own, for the exits that have no finish chunk.

        Carries one `finish_reason: null` choice rather than OpenAI's canonical
        empty `choices: []`. Both are read correctly by the consumer this
        change exists for, which looks at `chunk.usage` before it looks at the
        choices — but an empty list is indexed blindly by plenty of SDKs, and
        by four tests in this file, and an abnormal exit is the worst moment to
        hand a client a shape it has never seen.
        """
        reported = _usage_to_chat(captured)
        if reported is None:
            return None
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "usage": reported,
        }
        return f"data: {json.dumps(payload)}\n\n"

    # A turn that failed after a 200 is still a failure. Metered as a success
    # it is invisible: the usage row says the request was served, and the only
    # evidence is a chunk the operator never sees.
    #
    # Whatever counts it reached still go out, and go out *first*: a partial
    # count beats none, and an OpenAI client raises on the error line and
    # discards everything behind it, so usage emitted after one never arrives.
    if failed is not None:
        if outcome is not None:
            outcome["error_type"] = failed.get("type") or "upstream_stream_error"
        partial = usage_only_chunk()
        if partial is not None:
            yield partial
        yield f"data: {json.dumps({'error': failed})}\n\n"
        return

    if not ended:
        if outcome is not None:
            outcome["error_type"] = "upstream_stream_truncated"
        partial = usage_only_chunk()
        if partial is not None:
            yield partial
        yield f"data: {json.dumps({'error': {'type': 'api_error', 'message': 'Upstream stream ended before the turn completed'}})}\n\n"
        return

    if saw_tool_use and finish_reason == "stop":
        finish_reason = "tool_calls"

    # On the ordinary path the counts ride the finish chunk the caller already
    # expects, rather than an extra one. The shape every other client of this
    # route sees is then unchanged — and this route serves a dozen of them.
    final = json.loads(chunk({}, finish_reason)[len("data: ") :])
    reported = _usage_to_chat(captured)
    if reported is not None:
        final["usage"] = reported
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"
