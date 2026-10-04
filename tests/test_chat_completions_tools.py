"""Tools on the Chat Completions ⇄ Responses translation.

Requirements 2 and 3 on the one path that rebuilds the request field by
field. A tool the caller offered was never copied onto the Responses body, so
a dsh turn reached the model with nothing to call and answered in prose; an
assistant `tool_calls` message and its `role: tool` reply were replayed as
chat messages the Responses API ignores, so the conversation could not get
past its first call.
"""

from __future__ import annotations

import json

import pytest

from app.proxy.chat_completions import (
    stream_responses_to_completions,
    transform_response_to_completions,
    transform_to_responses_format,
)

WEATHER_TOOL = {
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


def _translate(**body) -> dict:
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return transform_to_responses_format(body, False, "gpt-6-sol")


def test_an_offered_tool_is_copied_onto_the_responses_body():
    """Requirement 2 — the model is actually offered the function."""
    out = _translate(tools=[WEATHER_TOOL], tool_choice="auto")

    assert out["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": WEATHER_TOOL["function"]["parameters"],
            "description": "Current conditions",
        }
    ]
    assert out["tool_choice"] == "auto"


def test_a_named_tool_choice_loses_the_chat_wrapper():
    out = _translate(
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    assert out["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_a_tool_call_and_its_result_become_responses_input_items():
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

    assert out["input"][1] == {
        "type": "function_call",
        "call_id": "call_abc",
        "name": "get_weather",
        "arguments": '{"city": "Dallas"}',
    }
    assert out["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_abc",
        "output": "88F and clear",
    }


def test_a_function_call_in_the_output_comes_back_as_a_tool_call():
    """Requirement 2 — the caller reads the call in its own shape."""
    out = transform_response_to_completions(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_xyz",
                    "name": "get_weather",
                    "arguments": '{"city":"Dallas"}',
                }
            ],
            "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
        },
        "gpt-6-sol",
    )

    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Dallas"}'},
        }
    ]


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _sse(events: list[dict]) -> list[str]:
    out = []
    for event in events:
        out.append(f"data: {json.dumps(event)}")
        out.append("")
    return out


@pytest.mark.asyncio
async def test_a_streamed_tool_call_reassembles_from_its_argument_deltas():
    """Requirement 2 — streaming carries the call, not just the prose."""
    chunks = []
    stream = stream_responses_to_completions(
        _FakeStream(
            _sse(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_xyz",
                            "name": "get_weather",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": '{"city"',
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": ': "Dallas"}',
                    },
                    {"type": "response.completed", "response": {"status": "completed"}},
                ]
            )
        ),
        "gpt-6-sol",
    )
    async for raw in stream:
        payload = raw[len("data: ") :].strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))

    arguments = "".join(
        call["function"]["arguments"]
        for chunk in chunks
        for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        if "arguments" in call.get("function", {})
    )
    assert json.loads(arguments) == {"city": "Dallas"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
