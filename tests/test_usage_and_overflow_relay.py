"""What a caller needs to size and rotate a conversation, on every route.

T1: a streamed turn on the chat-to-Responses translation ends with the
upstream's own token counts, and a translated turn whose upstream reported
none carries none — a fabricated zero is read as an empty context and never
rotates. The Anthropic translation's counts are pinned in
`test_chat_to_anthropic.py`; the native chat passthrough relays the
upstream's lines untouched.

Overflow: every route hands an upstream's "this did not fit" back as an error
whose code is `context_length_exceeded`, whichever dialect the upstream said
it in, and never as a `length` finish or a bare "Azure API error: 400".
Only the transport is stubbed.
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_session
from app.main import create_app
from app.orm import Base, Credential, Model, Provider
from app.proxy.chat_completions import (
    stream_responses_to_completions,
    transform_response_to_completions,
)
from app.proxy.chat_to_anthropic import stream_anthropic_to_completions

#: Anthropic's refusal for a prompt over the window: no code, only a sentence.
ANTHROPIC_OVERFLOW = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "prompt is too long: 215000 tokens > 200000 maximum",
    },
}

#: Ollama cloud's, verbatim from the proxy's own log: an OpenAI envelope whose
#: code is null.
OLLAMA_OVERFLOW = {
    "error": {
        "message": "The prompt is too long: 131621, model maximum context length: "
        "131072 (ref: 60c2911c-5fee-4e79-8716-7a65e970f11b)",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
}

#: Ollama's own envelope: the error is a bare sentence, with no code field at all.
OLLAMA_BARE_OVERFLOW = {
    "error": "The prompt is too long: 40961, model maximum context length: 40960"
}

#: The Codex backend's: already coded.
CODEX_OVERFLOW = {
    "error": {
        "message": "Your input exceeds the context window of this model. "
        "Please adjust your input and try again.",
        "type": "invalid_request_error",
        "param": "input",
        "code": "context_length_exceeded",
    }
}


def _message(refusal: dict) -> str:
    error = refusal["error"]
    return error if isinstance(error, str) else error["message"]


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


async def _chunks(gen) -> list[dict]:
    chunks = []
    async for raw in gen:
        payload = raw[len("data: ") :].strip()
        if payload != "[DONE]":
            chunks.append(json.loads(payload))
    return chunks


# ---------- T1: counts on the chat-to-Responses translation ----------


@pytest.mark.asyncio
async def test_a_streamed_responses_turn_ends_with_the_upstreams_token_counts():
    """T1. Responses' `input_tokens` already includes the cached prefix, so
    it is the caller's `prompt_tokens` exactly."""
    lines = _sse(
        [
            {"type": "response.output_text.delta", "delta": "hi"},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 407329,
                        "input_tokens_details": {"cached_tokens": 400000},
                        "output_tokens": 2584,
                        "total_tokens": 409913,
                    },
                },
            },
        ]
    )

    chunks = await _chunks(stream_responses_to_completions(_FakeStream(lines), "gpt-6-luna"))
    usage = chunks[-1]["usage"]

    assert usage["prompt_tokens"] == 407329
    assert usage["completion_tokens"] == 2584
    assert usage["prompt_tokens_details"]["cached_tokens"] == 400000
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reported",
    [None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}],
    ids=["absent", "zero"],
)
async def test_a_streamed_responses_turn_with_no_real_count_ends_without_usage(reported):
    """T1. Absent and zero are both "not measured". A zero on the last chunk is
    read as an empty context, which never rotates."""
    completed = {"status": "completed"}
    if reported is not None:
        completed["usage"] = reported
    lines = _sse(
        [
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.completed", "response": completed},
        ]
    )

    chunks = await _chunks(stream_responses_to_completions(_FakeStream(lines), "gpt-6-luna"))

    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in chunks[-1]


def test_a_translated_reply_whose_upstream_counted_zero_input_carries_no_usage():
    """T1. Zero input is not a count; relayed, it reads as an empty context."""
    out = transform_response_to_completions(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
        "gpt-6-luna",
    )

    assert "usage" not in out


def test_a_translated_reply_whose_upstream_reported_no_usage_carries_none():
    """T1. A zero the upstream never sent is a measurement nobody took."""
    out = transform_response_to_completions(
        {
            "id": "resp_1",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
        },
        "gpt-6-luna",
    )

    assert "usage" not in out


# ---------- Overflow, streamed translations ----------


@pytest.mark.asyncio
async def test_an_overflow_mid_stream_on_the_responses_translation_is_named_as_one():
    """The Codex backend ends an oversized turn with `response.failed`. Dropped,
    the caller is handed `[DONE]` and an empty answer that reads as served."""
    lines = _sse(
        [
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "Your input exceeds the context window of this model.",
                    },
                },
            }
        ]
    )

    raw = [c async for c in stream_responses_to_completions(_FakeStream(lines), "gpt-6-luna")]

    assert "data: [DONE]\n\n" not in raw
    last = json.loads(raw[-1][len("data: ") :])
    assert last["error"]["code"] == "context_length_exceeded"
    # pi-ai keeps only the message of a mid-stream error, so the code has to
    # survive inside it.
    assert "context_length_exceeded" in last["error"]["message"]


@pytest.mark.asyncio
async def test_a_claude_turn_that_runs_into_its_window_is_an_overflow_not_a_length_stop():
    """`model_context_window_exceeded` as `length` reads as `max_tokens`, which
    a caller retries unchanged. The counts it reached still go out first."""
    lines = _sse(
        [
            {"type": "message_start", "message": {"usage": {"input_tokens": 999000}}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "model_context_window_exceeded"},
                "usage": {"output_tokens": 1000},
            },
            {"type": "message_stop"},
        ]
    )

    chunks = await _chunks(
        stream_anthropic_to_completions(_FakeStream(lines), "claude-opus-5")
    )

    assert not [c for c in chunks if (c.get("choices") or [{}])[0].get("finish_reason")]
    assert chunks[-1]["error"]["code"] == "context_length_exceeded"
    assert "context_length_exceeded" in chunks[-1]["error"]["message"]
    assert [c["usage"]["prompt_tokens"] for c in chunks if "usage" in c] == [999000]


# ---------- Overflow over HTTP ----------


@pytest_asyncio.fixture
async def maker():
    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    made = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with made() as setup:
        credential = Credential(name="c", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        anthropic = Provider(
            name="anthropic", base_url="https://api.anthropic.com",
            messages_path="/v1/messages", credential_id=credential.id,
        )
        openai = Provider(
            name="openai", base_url="https://chatgpt.test",
            responses_path="/v1/responses", credential_id=credential.id,
        )
        ollama = Provider(
            name="ollama", base_url="http://127.0.0.1:11434",
            chat_completions_path="/v1/chat/completions", credential_id=credential.id,
        )
        setup.add_all([anthropic, openai, ollama])
        await setup.flush()
        setup.add_all(
            [
                Model(deployment_name="claude-opus-5", provider_id=anthropic.id,
                      auth_scheme="x_api_key", context_window=1_000_000),
                Model(deployment_name="gpt-6-sol", provider_id=openai.id,
                      context_window=872_000),
                Model(deployment_name="gpt-oss:120b-cloud", provider_id=ollama.id,
                      context_window=131_072),
            ]
        )
        await setup.commit()
    yield made
    await engine.dispose()


@pytest.fixture
def client(maker, monkeypatch):
    import app.main as main
    import app.proxy.chat_completions as chat
    import app.proxy.responses as responses
    import app.proxy.websocket as ws

    async def _session():
        async with maker() as s:
            yield s

    for module in (main, chat, responses, ws):
        monkeypatch.setattr(module, "validate_api_key", lambda *a, **k: True)
        monkeypatch.setattr(module, "resolve_requester_role", lambda *a, **k: "user")
    for module in (chat, responses, ws):
        monkeypatch.setattr(module, "resolve_principal", lambda *a, **k: None)
    monkeypatch.setattr(ws, "SessionLocal", maker)
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c


def _upstream(status: int, body: dict):
    async def _send(url, headers, body_sent, *, stream, log_prefix):
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    return _send


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "model, refusal",
    [("claude-opus-5", ANTHROPIC_OVERFLOW), ("gpt-6-sol", OLLAMA_OVERFLOW),
     ("gpt-oss:120b-cloud", OLLAMA_OVERFLOW),
     ("gpt-oss:120b-cloud", OLLAMA_BARE_OVERFLOW)],
)
def test_an_overflow_refusal_reaches_a_chat_caller_named_as_one(
    client, monkeypatch, model, refusal, streaming
):
    """Every chat path — Anthropic translation, Responses translation, native
    passthrough — names the overflow the same way, keeping the upstream's
    sentence and its numbers."""
    import app.proxy.chat_completions as chat

    monkeypatch.setattr(chat, "post_with_retries", _upstream(400, refusal))

    response = client.post(
        "/v1/chat/completions",
        json={"model": model, "stream": streaming,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert _message(refusal) in error["message"]


def test_a_claude_reply_cut_off_by_its_window_reaches_a_chat_caller_as_an_overflow(
    client, monkeypatch
):
    """The non-streamed half of the stop reason above."""
    import app.proxy.chat_completions as chat

    monkeypatch.setattr(
        chat,
        "post_with_retries",
        _upstream(
            200,
            {
                "id": "msg_1",
                "content": [{"type": "text", "text": "partial"}],
                "stop_reason": "model_context_window_exceeded",
                "usage": {"input_tokens": 999000, "output_tokens": 1000},
            },
        ),
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refusal", [CODEX_OVERFLOW, OLLAMA_OVERFLOW, OLLAMA_BARE_OVERFLOW])
def test_an_overflow_refusal_reaches_a_responses_caller_named_as_one(
    client, monkeypatch, refusal, streaming
):
    """Not "Azure API error: 400": that sentence is every 400 at once, and a
    caller cannot rotate on it."""
    import app.proxy.responses as responses

    monkeypatch.setattr(responses, "post_with_retries", _upstream(400, refusal))

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-6-sol", "input": "hi", "stream": streaming},
        headers={"Authorization": "Bearer test"},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert _message(refusal) in error["message"]


def test_an_unnamed_overflow_mid_stream_reaches_a_responses_caller_named(
    client, monkeypatch
):
    """An upstream that fails the turn after the 200 and says why only in a
    sentence has its failure rewritten into the event Codex reads."""
    import app.proxy.responses as responses

    sse = "\n".join(
        _sse([{"type": "error", "error": {"message": OLLAMA_OVERFLOW["error"]["message"]}}])
    )

    async def _send(url, headers, body_sent, *, stream, log_prefix):
        return httpx.Response(200, text=sse, request=httpx.Request("POST", url))

    monkeypatch.setattr(responses, "post_with_retries", _send)

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-6-sol", "input": "hi", "stream": True},
        headers={"Authorization": "Bearer test"},
    )

    events = [
        json.loads(line[len("data:") :])
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "context_length_exceeded"


def _ws_upstream(monkeypatch, handler):
    import app.proxy.websocket as ws

    real = httpx.AsyncClient
    monkeypatch.setattr(
        ws.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )


@pytest.mark.parametrize(
    "upstream",
    [
        lambda request: httpx.Response(400, json=OLLAMA_OVERFLOW),
        lambda request: httpx.Response(
            200,
            text="\n".join(
                _sse([{"type": "error",
                       "error": {"message": OLLAMA_OVERFLOW["error"]["message"]}}])
            ),
        ),
    ],
    ids=["refused", "mid-stream"],
)
def test_an_overflow_reaches_a_websocket_caller_as_a_failed_response(
    client, monkeypatch, upstream
):
    """Codex over the socket: the overflow arrives once, as the
    `response.failed` event it reads a full window from, rather than being
    retried five times and reported as "Azure API error: 400"."""
    calls: list[int] = []

    def handler(request):
        calls.append(1)
        return upstream(request)

    _ws_upstream(monkeypatch, handler)

    with client.websocket_connect("/v1/responses") as sock:
        sock.send_text(json.dumps({"model": "gpt-6-sol", "input": "hi"}))
        event = json.loads(sock.receive_text())

    assert event["type"] == "response.failed"
    assert event["response"]["error"]["code"] == "context_length_exceeded"
    assert event["response"]["error"]["message"] == OLLAMA_OVERFLOW["error"]["message"]
    assert len(calls) == 1
