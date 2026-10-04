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
    reject_wrong_protocol,
    resolve_target_uri,
    wires_for,
)
from app.main import create_app
from app.orm import Base, Credential, Model, Provider
from app.proxy.chat_to_anthropic import (
    DEFAULT_MAX_TOKENS,
    THINKING_BUDGETS,
    THINKING_HEADROOM,
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


def test_max_tokens_is_always_present_and_honours_the_caller():
    """Anthropic refuses a body with no `max_tokens`."""
    assert _translate()["max_tokens"] == DEFAULT_MAX_TOKENS
    assert _translate(max_tokens=512)["max_tokens"] == 512


@pytest.mark.parametrize("effort", sorted(THINKING_BUDGETS))
def test_a_reasoning_effort_becomes_a_thinking_budget(effort):
    """Requirement 6 — an effort is translated, never dropped."""
    out = _translate(reasoning_effort=effort, max_tokens=256)

    budget = THINKING_BUDGETS[effort]
    assert out["thinking"] == {"type": "enabled", "budget_tokens": budget}
    # `max_tokens` has to clear the budget or the request is refused.
    assert out["max_tokens"] == budget + THINKING_HEADROOM


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
