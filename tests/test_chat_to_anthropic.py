"""A Chat Completions caller reaching a Claude deployment.

Requirements 1 through 6: the chat endpoint reaches every model the proxy
serves without the caller naming a vendor or a wire, tools survive the round
trip in both directions, a tool result carries forward, the system prompt and
images land in the shape Anthropic reads, a failure arrives as the upstream's
own words at the upstream's own status, and a reasoning effort becomes a
thinking budget instead of being dropped.
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_session
from app.deployments import (
    DeploymentTarget,
    offerable_to,
    reject_wrong_protocol,
    resolve_target_uri,
    wires_for,
)
from app.main import create_app
from app.orm import Base, Credential, Model, Provider
from app.proxy.chat_to_anthropic import (
    UntranslatableRequest,
    stream_anthropic_to_completions,
    transform_anthropic_to_completions,
    transform_to_anthropic_messages,
)

ANTHROPIC_URI = "https://api.anthropic.com/v1/messages"


# ---------- Requirement 1: the chat wire reaches an Anthropic deployment ----------


def _messages_only_provider() -> Provider:
    return Provider(
        name="anthropic",
        label="Anthropic",
        base_url="https://api.anthropic.com",
        messages_path="/v1/messages",
    )


def test_the_chat_wire_resolves_a_messages_only_provider_to_its_messages_path():
    """Requirement 1 — asking for the chat shape yields the Anthropic URL."""
    row = Model(deployment_name="claude-opus-5")
    row.provider = _messages_only_provider()
    row.provider.enabled = True

    assert resolve_target_uri(row, "chat_completions") == ANTHROPIC_URI


def test_a_provider_serving_responses_is_not_sent_down_the_anthropic_detour():
    """The Responses translation is preferred where a provider serves both."""
    row = Model(deployment_name="both-wires")
    provider = _messages_only_provider()
    provider.enabled = True
    provider.responses_path = "/v1/responses"
    row.provider = provider

    assert resolve_target_uri(row, "chat_completions").endswith("/v1/responses")


def test_a_messages_only_model_lists_chat_completions_among_its_wires():
    """Requirement 1 — a chat-only picker is offered the Claude models."""
    row = Model(deployment_name="claude-opus-5", target_uri=ANTHROPIC_URI)
    row.provider = None

    assert "chat_completions" in wires_for(row)


def test_a_claude_model_is_not_offered_to_a_codex_picker():
    """Codex picks the Responses endpoint, so offering it one is the lie."""
    row = Model(deployment_name="claude-opus-5")
    row.provider = _messages_only_provider()
    row.provider.enabled = True

    assert offerable_to(row, "chat_completions", "codex") is False
    assert offerable_to(row, "chat_completions", "dsh") is True


def test_a_native_chat_model_is_still_offered_to_a_codex_picker():
    """The withholding is the Anthropic stand-in alone, not the whole shape.

    A Foundry Models-as-a-Service row — DeepSeek, Kimi, gpt-oss — is reached
    by forwarding the body unchanged. Nothing translates, so nothing about
    what Codex is offered there may change.
    """
    row = Model(deployment_name="deepseek-v3")
    provider = Provider(
        name="foundry",
        label="Foundry",
        base_url="https://foundry.test",
        chat_completions_path="/models/chat/completions",
    )
    provider.enabled = True
    row.provider = provider

    assert offerable_to(row, "chat_completions", "codex") is True


def test_the_chat_route_accepts_an_anthropic_messages_deployment():
    """Requirement 1 — the protocol guard no longer refuses the Claude wire."""
    target = DeploymentTarget(
        name="claude-opus-5", target_uri=ANTHROPIC_URI, api_key="k", api_version=None
    )

    reject_wrong_protocol(target, expected="chat_completions_any")


# ---------- Requirement 2, 3, 4, 6: request translation ----------


def _translate(**body) -> dict:
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return transform_to_anthropic_messages(
        body, streaming=False, model="claude-opus-5"
    )


def test_an_offered_tool_arrives_with_its_schema_under_input_schema():
    """Requirement 2 — a tool the caller offered reaches the model."""
    out = _translate(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
    )

    assert out["tools"] == [
        {
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "description": "Current conditions",
        }
    ]


def test_the_system_prompt_is_hoisted_out_of_the_message_list():
    """Requirement 4 — Anthropic reads `system`, not a system message."""
    out = _translate(
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "Also honest."},
        ]
    )

    assert out["system"] == "You are terse.\nAlso honest."
    assert [m["role"] for m in out["messages"]] == ["user"]


def test_a_tool_call_and_its_result_become_matching_use_and_result_blocks():
    """Requirement 3 — a multi-step tool conversation carries forward."""
    out = _translate(
        messages=[
            {"role": "user", "content": "weather in Dallas?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Dallas"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "88F and clear"},
        ]
    )

    assistant, result = out["messages"][1], out["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {
            "type": "tool_use",
            "id": "call_abc",
            "name": "get_weather",
            "input": {"city": "Dallas"},
        }
    ]
    assert result["role"] == "user"
    assert result["content"] == [
        {"type": "tool_result", "tool_use_id": "call_abc", "content": "88F and clear"}
    ]


def test_two_results_for_one_assistant_turn_share_a_single_user_turn():
    """Anthropic refuses a run of user turns each holding one result."""
    out = _translate(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "g", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "b", "content": "two"},
        ]
    )

    assert [m["role"] for m in out["messages"]] == ["assistant", "user"]
    assert [b["tool_use_id"] for b in out["messages"][1]["content"]] == ["a", "b"]


def test_a_developer_message_is_hoisted_like_a_system_one():
    """Requirement 4 — the newer spelling of a system prompt is not left inline."""
    out = _translate(
        messages=[
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "hello"},
        ]
    )

    assert out["system"] == "Be terse."
    assert [m["role"] for m in out["messages"]] == ["user"]


def test_a_hosted_image_arrives_as_a_url_source():
    """Requirement 4 — not every attachment is inlined as base64."""
    out = _translate(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
                ],
            }
        ]
    )

    assert out["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}}
    ]


def test_an_attached_image_becomes_a_base64_image_block():
    """Requirement 4 — an image arrives in the form this wire expects."""
    out = _translate(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
                    },
                ],
            }
        ]
    )

    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "what is this?"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
        },
    ]


@pytest.mark.parametrize("spelling", ["text", "input_text", "output_text"])
def test_a_text_part_reaches_the_model_however_it_is_spelled(spelling):
    """Requirement 4 — a turn's body is never dropped for its part type.

    `input_text` and `output_text` are this proxy's *own* other translation's
    spelling, so the same client body reached a Codex model intact and a
    Claude model with the text gone — a 200 answering a question the model
    was never asked.
    """
    out = _translate(
        messages=[{"role": "user", "content": [{"type": spelling, "text": "2 + 2?"}]}]
    )

    assert out["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "2 + 2?"}]}
    ]


def test_an_assistant_turn_is_not_dropped_for_its_part_spelling():
    """A lost assistant turn rewrites the conversation the model answers."""
    out = _translate(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
            {"role": "user", "content": "and now?"},
        ]
    )

    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user"]


def test_an_image_already_in_anthropic_form_passes_through():
    """Requirement 4 — the target wire's own spelling is not thrown away."""
    source = {"type": "base64", "media_type": "image/png", "data": "QUJD"}
    out = _translate(
        messages=[{"role": "user", "content": [{"type": "image", "source": source}]}]
    )

    assert out["messages"][0]["content"] == [{"type": "image", "source": source}]


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": ""}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64"}},
        {"type": "input_audio", "input_audio": {"data": "AAA"}},
    ],
)
def test_an_attachment_this_wire_cannot_carry_is_refused_not_dropped(part):
    """Dropping it leaves a 200 answering about a file the model never saw."""
    with pytest.raises(UntranslatableRequest):
        _translate(messages=[{"role": "user", "content": [part]}])


def test_a_structured_tool_result_reaches_the_model_as_json():
    """A Python repr hands the model single quotes, `True` and `None`."""
    out = _translate(
        messages=[
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": {"ok": True, "rows": 3}},
        ]
    )

    assert out["messages"][1]["content"][0]["content"] == '{"ok": true, "rows": 3}'


def test_an_empty_stop_string_is_not_sent_as_a_stop_sequence():
    """Anthropic refuses it, so a client defaulting `stop` to "" is locked out."""
    assert "stop_sequences" not in _translate(stop="")
    assert _translate(stop="END")["stop_sequences"] == ["END"]


def test_a_tool_whose_schema_arrived_serialised_is_still_a_tool():
    out = _translate(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": '{"type": "object", "properties": {}}',
                },
            }
        ]
    )

    assert out["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


def test_a_tool_with_no_function_name_is_refused_rather_than_dropped():
    """A silently dropped tool is a model answering in prose for no reason."""
    with pytest.raises(UntranslatableRequest):
        _translate(tools=[{"type": "web_search_preview"}])


@pytest.mark.parametrize(
    "body",
    [
        {"tool_choice": {"type": "function", "function": "f"}},
        {"messages": [{"role": "assistant", "tool_calls": [{"id": "x", "function": "f"}]}]},
    ],
)
def test_a_malformed_field_does_not_become_an_internal_error(body):
    """A 500 reads as a defect in the proxy; this is the caller's own input."""
    body.setdefault("tools", [{"type": "function", "function": {"name": "f"}}])
    _translate(**body)


def test_max_tokens_is_always_present_and_honours_the_caller():
    """Anthropic refuses a body with no `max_tokens`."""
    assert _translate()["max_tokens"] == 8192
    assert _translate(max_tokens=512)["max_tokens"] == 512
    assert _translate(max_completion_tokens=777)["max_tokens"] == 777


@pytest.mark.parametrize(
    "effort,budget",
    [
        ("minimal", 1024),
        ("low", 4096),
        ("medium", 8192),
        ("high", 16384),
        ("xhigh", 32768),
    ],
)
def test_a_reasoning_effort_becomes_a_thinking_budget(effort, budget):
    """Requirement 6 — an effort is translated, never dropped."""
    out = _translate(reasoning_effort=effort)

    assert out["thinking"] == {"type": "enabled", "budget_tokens": budget}
    # Anthropic's floor is 1024, and a budget below it is refused outright.
    assert budget >= 1024
    # `max_tokens` has to *exceed* the budget or the request is refused, so
    # with no ceiling named it rises to leave room for a reply behind the
    # thinking rather than a reply of nothing.
    assert out["max_tokens"] - budget >= 4096


def test_a_named_ceiling_is_not_raised_to_fit_a_thinking_budget():
    """A cap named for cost must not be overruled many times over, silently."""
    out = _translate(reasoning_effort="xhigh", max_tokens=12000)

    assert out["max_tokens"] == 12000
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 12000 - 4096}


def test_an_effort_that_cannot_fit_under_the_ceiling_is_dropped_not_the_ceiling():
    """Anthropic's floor is 1024; below that there is no budget to send."""
    out = _translate(reasoning_effort="high", max_tokens=256)

    assert out["max_tokens"] == 256
    assert "thinking" not in out
    # Dropping the effort is what lets the sampling parameters stay, too.
    assert _translate(reasoning_effort="high", max_tokens=256, temperature=0.2)[
        "temperature"
    ] == 0.2


def test_a_signed_thinking_block_survives_the_round_trip():
    """Requirement 3 and 6 together — Anthropic refuses the follow-up without it.

    A tool conversation with thinking enabled has to hand the signed blocks
    back on the assistant turn that carries the `tool_use`. Discarded, the
    conversation is one turn long by construction.
    """
    reply = transform_anthropic_to_completions(
        {
            "id": "msg_1",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "needs the tool", "signature": "SIGBYTES"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Dallas"},
                },
            ],
        },
        "claude-opus-5",
    )
    assistant = reply["choices"][0]["message"]
    assert assistant["reasoning_details"][0]["signature"] == "SIGBYTES"

    out = _translate(
        messages=[
            {"role": "user", "content": "weather?"},
            assistant,
            {"role": "tool", "tool_call_id": "toolu_1", "content": "88F"},
        ]
    )

    blocks = out["messages"][1]["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "needs the tool",
        "signature": "SIGBYTES",
    }
    assert blocks[-1]["type"] == "tool_use"


def test_the_nested_reasoning_form_is_read_too(effort="high"):
    """Requirement 6 — an SDK sending `reasoning.effort` is not dropped."""
    out = _translate(reasoning={"effort": effort})

    assert out["thinking"] == {"type": "enabled", "budget_tokens": 16384}


def test_thinking_is_absent_when_the_caller_asked_for_none():
    assert "thinking" not in _translate(reasoning_effort="off")
    assert "thinking" not in _translate()


def test_extended_thinking_drops_the_sampling_parameters_it_forbids():
    out = _translate(reasoning_effort="high", temperature=0.2, top_p=0.9)

    assert "temperature" not in out and "top_p" not in out
    assert _translate(temperature=0.2)["temperature"] == 0.2


@pytest.mark.parametrize(
    "given,expected",
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        ({"type": "function", "function": {"name": "get_weather"}},
         {"type": "tool", "name": "get_weather"}),
    ],
)
def test_tool_choice_is_translated_in_every_form_the_caller_may_send(given, expected):
    """Requirement 2 — a forced tool stays forced across the wire."""
    out = _translate(
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        tool_choice=given,
    )

    assert out["tool_choice"] == expected


# ---------- Requirement 2: response translation ----------


def test_a_tool_use_reply_comes_back_as_a_chat_completions_tool_call():
    """Requirement 2 — the caller reads a tool call in its own shape."""
    out = transform_anthropic_to_completions(
        {
            "id": "msg_1",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "get_weather",
                    "input": {"city": "Dallas"},
                },
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
        "claude-opus-5",
    )

    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "toolu_9",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Dallas"}',
            },
        }
    ]
    assert choice["message"]["content"] == "Checking."


def test_a_text_reply_comes_back_as_content_with_its_token_counts():
    out = transform_anthropic_to_completions(
        {
            "id": "msg_2",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "88F and clear."}],
            "usage": {"input_tokens": 12, "output_tokens": 5},
        },
        "claude-opus-5",
    )

    assert out["choices"][0]["message"]["content"] == "88F and clear."
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


class _FakeStream:
    """An httpx-shaped line source, standing in for the transport only."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _sse(events: list[dict]) -> list[str]:
    out = []
    for event in events:
        out.append(f"event: {event['type']}")
        out.append(f"data: {json.dumps(event)}")
        out.append("")
    return out


async def _collect(lines: list[str]) -> list[dict]:
    chunks = []
    async for raw in stream_anthropic_to_completions(
        _FakeStream(lines), "claude-opus-5"
    ):
        payload = raw[len("data: ") :].strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))
    return chunks


@pytest.mark.asyncio
async def test_a_streamed_tool_call_reassembles_from_its_argument_deltas():
    """Requirement 2 — streaming carries the call, not just the prose."""
    chunks = await _collect(
        _sse(
            [
                {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_4",
                        "name": "get_weather",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city"'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ': "Dallas"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 14},
                },
                {"type": "message_stop"},
            ]
        )
    )

    arguments = "".join(
        call["function"]["arguments"]
        for chunk in chunks
        for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if "arguments" in call.get("function", {})
    )
    assert json.loads(arguments) == {"city": "Dallas"}
    names = [
        call["function"]["name"]
        for chunk in chunks
        for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if call.get("function", {}).get("name")
    ]
    assert names == ["get_weather"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_a_streamed_tool_call_is_numbered_among_tool_calls_not_blocks():
    """A text block ahead of two calls must not leave an index gap."""
    chunks = await _collect(
        _sse(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Looking."},
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "a", "name": "f"},
                },
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "b", "name": "g"},
                },
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            ]
        )
    )

    indices = [
        call["index"]
        for chunk in chunks
        for call in chunk["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert indices == [0, 1]
    assert any(
        chunk["choices"][0]["delta"].get("content") == "Looking." for chunk in chunks
    )


def test_an_empty_thinking_block_is_not_offered_as_reasoning_to_read():
    """Anthropic routinely returns one; a live call against Opus 5 did."""
    out = transform_anthropic_to_completions(
        {
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "391"},
            ],
            "stop_reason": "end_turn",
        },
        "claude-opus-5",
    )

    assert "reasoning_content" not in out["choices"][0]["message"]


def test_thinking_that_has_text_is_handed_to_the_caller():
    out = transform_anthropic_to_completions(
        {
            "content": [
                {"type": "thinking", "thinking": "17*20 then 17*3"},
                {"type": "text", "text": "391"},
            ],
            "stop_reason": "end_turn",
        },
        "claude-opus-5",
    )

    assert out["choices"][0]["message"]["reasoning_content"] == "17*20 then 17*3"


def test_a_refused_generation_is_not_reported_as_a_finished_one():
    """`stop` on a refusal is an empty successful answer the caller won't retry."""
    out = transform_anthropic_to_completions(
        {"content": [{"type": "text", "text": ""}], "stop_reason": "refusal"},
        "claude-opus-5",
    )

    assert out["choices"][0]["finish_reason"] == "content_filter"


def test_running_out_of_context_is_reported_as_a_length_stop():
    out = transform_anthropic_to_completions(
        {
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "model_context_window_exceeded",
        },
        "claude-opus-5",
    )

    assert out["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_a_stream_cut_mid_tool_call_is_not_reported_as_a_finished_call():
    """A truncated stream looks identical to a finished one at end of iteration.

    Handed a success terminator anyway, the caller reassembles half a JSON
    object and is told the call is complete — and the turn meters 200.
    """
    lines = _sse(
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t1", "name": "delete_rows"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"path":"/ho'},
            },
        ]
    )
    raw = [
        line
        async for line in stream_anthropic_to_completions(
            _FakeStream(lines), "claude-opus-5"
        )
    ]

    assert not any(line.strip() == "data: [DONE]" for line in raw)
    assert not any(
        json.loads(line[len("data: ") :]).get("choices", [{}])[0].get("finish_reason")
        for line in raw
        if line.startswith("data: ")
    )
    assert "ended before the turn completed" in raw[-1]


@pytest.mark.asyncio
async def test_a_failure_mid_stream_is_not_followed_by_a_success_terminator():
    """A reader that skips chunks with no `choices` would see a clean answer."""
    lines = _sse(
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Here is "},
            },
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        ]
    )
    raw = [
        line
        async for line in stream_anthropic_to_completions(
            _FakeStream(lines), "claude-opus-5"
        )
    ]

    assert not any(line.strip() == "data: [DONE]" for line in raw)
    assert json.loads(raw[-1][len("data: ") :])["error"]["message"] == "Overloaded"


@pytest.mark.asyncio
async def test_a_streamed_turn_reports_its_token_counts_for_metering():
    """The holder is the channel by which a streamed turn gets metered.

    Named for what it guards: this is the proxy's own usage_log row, not
    anything the caller receives. The caller's copy is guarded by
    `test_a_streamed_turn_hands_the_caller_its_token_counts`.
    """
    captured: dict = {}
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 31}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
        ]
    )
    async for _ in stream_anthropic_to_completions(
        _FakeStream(lines), "claude-opus-5", usage_holder=captured
    ):
        pass

    assert captured["input_tokens"] == 31
    assert captured["output_tokens"] == 12


# ---------- Requirement 5: the upstream's own words ----------


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app on a fresh database, holding one Anthropic-wire deployment."""
    import app.main as main

    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as setup:
        credential = Credential(name="anthropic", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        provider = Provider(
            name="anthropic",
            label="Anthropic",
            base_url="https://api.anthropic.com",
            messages_path="/v1/messages",
            credential_id=credential.id,
        )
        setup.add(provider)
        await setup.flush()
        setup.add(
            Model(
                deployment_name="claude-opus-5",
                provider_id=provider.id,
                auth_scheme="x_api_key",
            )
        )
        responses = Provider(
            name="openai",
            label="OpenAI",
            base_url="https://foundry.test",
            responses_path="/v1/responses",
            credential_id=credential.id,
        )
        setup.add(responses)
        await setup.flush()
        setup.add(Model(deployment_name="gpt-6-sol", provider_id=responses.id))
        await setup.commit()

    async def _session():
        async with maker() as s:
            yield s

    import app.proxy.chat_completions as chat

    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr(main, "resolve_requester_role", lambda *a, **k: "user")
    monkeypatch.setattr(chat, "validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr(chat, "resolve_requester_role", lambda *a, **k: "user")
    monkeypatch.setattr(chat, "resolve_principal", lambda *a, **k: None)
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c
    await engine.dispose()


def test_an_upstream_refusal_reaches_the_caller_in_its_own_words(client, monkeypatch):
    """Requirement 5 — the upstream's status and sentence, not a reword."""
    import app.proxy.chat_completions as chat

    upstream_body = {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "quota exhausted until 4pm"},
    }

    async def _refuse(url, headers, body, *, stream, log_prefix):
        return httpx.Response(
            429,
            json=upstream_body,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat, "post_with_retries", _refuse)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 429
    assert response.json() == upstream_body


def test_a_chat_caller_reaches_a_claude_model_over_http(client, monkeypatch):
    """Requirement 1 and 2 end to end — no vendor named by the caller."""
    import app.proxy.chat_completions as chat

    sent: dict = {}

    async def _answer(url, headers, body, *, stream, log_prefix):
        sent["url"] = url
        sent["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "msg_x",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Dallas"},
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat, "post_with_retries", _answer)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "weather in Dallas?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    assert sent["url"] == ANTHROPIC_URI
    assert sent["body"]["tools"][0]["name"] == "get_weather"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def _refusal(status: int, body, *, raw: bool = False):
    """A stubbed upstream that refuses. The transport is the only thing faked."""

    async def _refuse(url, headers, body_sent, *, stream, log_prefix):
        request = httpx.Request("POST", url)
        if raw:
            return httpx.Response(status, text=body, request=request)
        return httpx.Response(status, json=body, request=request)

    return _refuse


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("model", ["claude-opus-5", "gpt-6-sol"])
def test_every_translating_path_hands_back_the_upstream_refusal(
    client, monkeypatch, model, streaming
):
    """Requirement 5 — on every path, not just the one that was easiest.

    Both translations rebuild the request, and both have a streaming and a
    non-streaming transport. A reworded refusal on any of the four is a
    sentence an OpenAI client renders as "no body", so the operator reading
    the status has nothing to act on.
    """
    import app.proxy.chat_completions as chat

    upstream_body = {
        "error": {"type": "rate_limit_error", "message": "quota exhausted until 4pm"}
    }
    monkeypatch.setattr(chat, "post_with_retries", _refusal(429, upstream_body))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "stream": streaming,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 429
    assert response.json() == upstream_body


def test_a_refusal_that_is_not_json_is_wrapped_so_a_client_can_read_it(
    client, monkeypatch
):
    """Requirement 5 — a gateway's HTML 502 still has to carry its words."""
    import app.proxy.chat_completions as chat

    monkeypatch.setattr(
        chat, "post_with_retries", _refusal(502, "<html>bad gateway</html>", raw=True)
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "<html>bad gateway</html>"


def test_a_streaming_chat_request_reaches_the_claude_wire_as_a_stream(
    client, monkeypatch
):
    """Requirement 1 and 2 — the streaming branch, the headers, the model.

    The non-streaming test said nothing about any of the three: a route that
    served every streaming request non-streaming, sent the caller's own model
    name upstream instead of the deployment's, or carried no credential at
    all would have left it green.
    """
    import app.proxy.chat_completions as chat

    sent: dict = {}

    async def _answer(url, headers, body, *, stream, log_prefix):
        sent["headers"] = headers
        sent["body"] = body
        sent["stream"] = stream
        lines = _sse(
            [
                {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "88F and clear."},
                },
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            ]
        )
        return httpx.Response(
            200,
            text="\n".join(lines) + "\n",
            headers={"content-type": "text/event-stream"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat, "post_with_retries", _answer)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "stream": True,
            "messages": [{"role": "user", "content": "weather?"}],
        },
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 200
    assert sent["stream"] is True
    assert sent["body"]["stream"] is True
    assert sent["body"]["model"] == "claude-opus-5"
    assert sent["headers"]["x-api-key"] == "s"
    assert sent["headers"]["anthropic-version"] == "2023-06-01"

    text = "".join(
        json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content", "")
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    )
    assert text == "88F and clear."


def test_an_untranslatable_request_is_refused_in_the_callers_own_envelope(
    client, monkeypatch
):
    """A 500 reads as a defect here; this names what the caller actually sent."""
    import app.proxy.chat_completions as chat

    reached = []

    async def _never(url, headers, body, *, stream, log_prefix):
        reached.append(url)
        raise AssertionError("the upstream must not be called")

    monkeypatch.setattr(chat, "post_with_retries", _never)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_preview"}],
        },
        headers={"Authorization": "Bearer test"},
    )

    assert reached == []
    assert response.status_code == 400
    assert "web_search_preview" in response.json()["error"]["message"]


def test_a_success_carrying_no_json_is_not_reported_as_our_own_failure(
    client, monkeypatch
):
    """A reverse proxy's interstitial at 200 is the upstream's problem, said so."""
    import app.proxy.chat_completions as chat

    async def _interstitial(url, headers, body, *, stream, log_prefix):
        return httpx.Response(
            200, text="<html>checking your browser</html>",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(chat, "post_with_retries", _interstitial)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 502
    assert "checking your browser" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_turn_that_failed_after_a_200_is_not_metered_as_a_success():
    """Metered 200 the failure is invisible — the usage row says it was served."""
    outcome: dict = {}
    lines = _sse(
        [
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        ]
    )
    async for _ in stream_anthropic_to_completions(
        _FakeStream(lines), "claude-opus-5", outcome=outcome
    ):
        pass

    assert outcome["error_type"] == "overloaded_error"

    truncated: dict = {}
    async for _ in stream_anthropic_to_completions(
        _FakeStream(
            _sse(
                [
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "half"},
                    }
                ]
            )
        ),
        "claude-opus-5",
        outcome=truncated,
    ):
        pass

    assert truncated["error_type"] == "upstream_stream_truncated"


# ---------- Streamed turns report their token counts to the caller ----------
#
# Cypher reaches this proxy as `api: openai-completions`, so her turns run
# through this converter. Before this section existed she recorded zero tokens
# for turns the proxy's own usage_log recorded in full, and her rotation hook
# could never measure context.


@pytest.mark.asyncio
async def test_a_streamed_turn_hands_the_caller_its_token_counts():
    """The counts reach the caller, not just the metering holder."""
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 16271}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 127},
            },
        ]
    )

    chunks = await _collect(lines)
    carrying = [c for c in chunks if "usage" in c]

    assert carrying, "no chunk carried usage"
    usage = carrying[-1]["usage"]
    assert usage["prompt_tokens"] == 16271
    assert usage["completion_tokens"] == 127
    assert usage["total_tokens"] == 16271 + 127


@pytest.mark.asyncio
async def test_a_cached_turn_reports_a_prompt_total_the_caller_can_subtract_from():
    """`prompt_tokens` is inclusive, as every OpenAI consumer reads it.

    Anthropic's `input_tokens` excludes the cached prefix; OpenAI's
    `prompt_tokens` includes it, with the cached portion named as a subset.
    pi-ai recovers fresh input as `prompt_tokens - cached - cache_write`, so an
    exclusive total drives its answer to zero on a cache-heavy turn and loses
    the created prefix entirely.
    """
    fresh, read, created = 300, 15971, 4096
    lines = _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": fresh,
                        "cache_read_input_tokens": read,
                        "cache_creation_input_tokens": created,
                    }
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 11},
            },
        ]
    )

    usage = [c for c in await _collect(lines) if "usage" in c][-1]["usage"]
    details = usage["prompt_tokens_details"]

    assert usage["prompt_tokens"] == fresh + read + created
    assert details["cached_tokens"] == read
    assert details["cache_write_tokens"] == created
    # The consumer recovers fresh input as prompt - cached - written; that it
    # comes out to `fresh` follows from the three assertions above, so there is
    # nothing left for a fourth to detect.


@pytest.mark.asyncio
async def test_a_truncated_turn_still_reports_the_counts_it_reached():
    """A partial count beats no count, and it has to precede the error line.

    The OpenAI SDK raises on a `{"error": ...}` line and discards whatever
    follows it, so usage emitted after one never reaches the caller.
    """
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 16271}}},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        ]
    )

    raw = [
        line
        async for line in stream_anthropic_to_completions(_FakeStream(lines), "claude-opus-5")
    ]
    payloads = [json.loads(l[len("data: ") :]) for l in raw if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    usage_at = [i for i, p in enumerate(payloads) if "usage" in p]
    error_at = [i for i, p in enumerate(payloads) if "error" in p]

    assert usage_at, "a truncated turn carried no usage"
    assert payloads[usage_at[-1]]["usage"]["prompt_tokens"] == 16271
    assert error_at and usage_at[-1] < error_at[0], "usage must precede the error line"


@pytest.mark.asyncio
async def test_a_turn_whose_upstream_reported_nothing_carries_no_counts():
    """Unmeasured is reported as unmeasured rather than as a measured zero.

    `_usage_to_chat({})` yields a well-formed all-zero dict, and pi-ai
    overwrites usage on every usage-bearing chunk, so a fabricated zero wipes a
    real measurement. Cypher's hook reads an all-zero usage dict as absence and
    walks further back, which hands her the previous turn's figure.
    """
    lines = _sse(
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
    )

    assert not [c for c in await _collect(lines) if "usage" in c]


@pytest.mark.asyncio
async def test_an_aborted_turn_still_records_what_it_had_for_metering():
    """The metering row survives a transport failure mid-stream.

    There is no `try` around the event loop, so an httpx error or a caller
    hanging up skips everything after it. The live usage_log carries rows at
    status 200 with NULL tokens from exactly this.
    """

    class _DyingStream:
        async def aiter_lines(self):
            for line in _sse(
                [{"type": "message_start", "message": {"usage": {"input_tokens": 16271}}}]
            ):
                yield line
            raise httpx.ReadError("upstream went away")

    captured: dict = {}
    with pytest.raises(httpx.ReadError):
        async for _ in stream_anthropic_to_completions(
            _DyingStream(), "claude-opus-5", usage_holder=captured
        ):
            pass

    assert captured.get("input_tokens") == 16271


@pytest.mark.asyncio
async def test_a_later_usage_event_does_not_zero_an_earlier_count():
    """A blind merge lets a zeroed `message_delta` overwrite a good count.

    The metering row and the chunk would then agree perfectly on a wrong
    number, which is the one disagreement nothing downstream could detect.
    """
    lines = _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {"input_tokens": 16271, "cache_read_input_tokens": 4096}
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 127,
                },
            },
        ]
    )

    captured: dict = {}
    async for _ in stream_anthropic_to_completions(
        _FakeStream(lines), "claude-opus-5", usage_holder=captured
    ):
        pass

    assert captured["input_tokens"] == 16271
    assert captured["cache_read_input_tokens"] == 4096
    assert captured["output_tokens"] == 127


@pytest.mark.asyncio
async def test_an_errored_turn_still_reports_the_counts_it_reached():
    """The upstream-error exit, not just the truncation one.

    They are independent branches, and a count emitted from one proves nothing
    about the other.
    """
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 16271}}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
        ]
    )

    payloads = [
        json.loads(l[len("data: ") :])
        async for l in stream_anthropic_to_completions(_FakeStream(lines), "claude-opus-5")
        if l.startswith("data: ") and l[6:].strip() != "[DONE]"
    ]
    usage_at = [i for i, p in enumerate(payloads) if "usage" in p]
    error_at = [i for i, p in enumerate(payloads) if "error" in p]

    assert usage_at, "an errored turn carried no usage"
    assert payloads[usage_at[-1]]["usage"]["prompt_tokens"] == 16271
    assert error_at and usage_at[-1] < error_at[0]


@pytest.mark.asyncio
async def test_an_errored_turn_that_reached_no_counts_carries_none():
    """The abnormal exits need their own gate, not the normal exit's.

    This is the path most likely to have reached no counts at all, so a
    fabricated zero is likeliest here — and it is the shape that wipes a
    consumer's running measurement.
    """
    lines = _sse([{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}])

    payloads = [
        json.loads(l[len("data: ") :])
        async for l in stream_anthropic_to_completions(_FakeStream(lines), "claude-opus-5")
        if l.startswith("data: ") and l[6:].strip() != "[DONE]"
    ]

    assert not [p for p in payloads if "usage" in p]


@pytest.mark.asyncio
async def test_the_caller_and_the_metering_row_describe_one_turn():
    """One turn's chunk against that same turn's metering holder.

    The two express the same counts in different conventions — the row keeps
    Anthropic's native fields, the caller gets OpenAI's inclusive total — so
    the check is that the row's parts add up to what the caller was told. A
    field dropped on the way to either side is a per-turn divergence between
    what Cypher measures and what the usage log says she used.
    """
    fresh, read, created, out = 300, 15971, 4096, 11
    lines = _sse(
        [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": fresh,
                        "cache_read_input_tokens": read,
                        "cache_creation_input_tokens": created,
                    }
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": out},
            },
        ]
    )

    row: dict = {}
    chunks = []
    async for raw in stream_anthropic_to_completions(
        _FakeStream(lines), "claude-opus-5", usage_holder=row
    ):
        payload = raw[len("data: ") :].strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))

    told = [c for c in chunks if "usage" in c][-1]["usage"]

    assert told["prompt_tokens"] == (
        row["input_tokens"] + row["cache_read_input_tokens"] + row["cache_creation_input_tokens"]
    )
    assert told["completion_tokens"] == row["output_tokens"]


@pytest.mark.asyncio
async def test_a_turn_whose_prompt_was_never_reported_carries_no_counts():
    """A real output count does not license a fabricated prompt of zero.

    Shims on this route report their prompt fields as null while reporting a
    genuine output figure. A gate that only asked whether anything at all was
    reported would emit `prompt_tokens: 0` here, which pi-ai reads as a
    measured zero and writes over what it already knew.
    """
    lines = _sse(
        [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": None, "cache_read_input_tokens": None}},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 127},
            },
        ]
    )

    assert not [c for c in await _collect(lines) if "usage" in c]


@pytest.mark.asyncio
async def test_counts_an_upstream_sends_as_floats_are_not_lost():
    """A shim that JSON-encodes its counts as floats is still measured.

    Dropping them does not read as missing — it reads as zero, on both the
    chunk and the row, which agree and are both wrong.
    """
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 16271.0}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 127.0},
            },
        ]
    )

    usage = [c for c in await _collect(lines) if "usage" in c][-1]["usage"]

    assert usage["prompt_tokens"] == 16271
    assert usage["completion_tokens"] == 127


@pytest.mark.asyncio
async def test_counts_reported_on_message_stop_are_folded_in():
    """Anthropic reports earlier, but shims on this route report here."""
    lines = _sse(
        [
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop", "usage": {"input_tokens": 16271, "output_tokens": 127}},
        ]
    )

    usage = [c for c in await _collect(lines) if "usage" in c][-1]["usage"]

    assert usage["prompt_tokens"] == 16271
    assert usage["completion_tokens"] == 127


@pytest.mark.asyncio
async def test_an_aborted_turn_is_recorded_as_aborted():
    """Carrying counts must not make an abort look like a served turn.

    The metering row reads its status from `outcome`, and an empty token
    column was the only way an operator ever spotted these. Now that the
    counts survive the abort, the outcome has to say what happened.
    """

    class _DyingStream:
        async def aiter_lines(self):
            for line in _sse(
                [{"type": "message_start", "message": {"usage": {"input_tokens": 16271}}}]
            ):
                yield line
            raise httpx.ReadError("upstream went away")

    outcome: dict = {}
    with pytest.raises(httpx.ReadError):
        async for _ in stream_anthropic_to_completions(
            _DyingStream(), "claude-opus-5", outcome=outcome
        ):
            pass

    assert outcome["error_type"] == "upstream_stream_aborted"
