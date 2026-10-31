"""The decision wire: resolution onto it, and the route over HTTP.

Nothing on this network calls Ollama directly, so a decision call has to
arrive here and be forwarded. Two things can quietly go wrong: a provider
that does not serve the wire having its chat path used instead, and the body
being rebuilt on the way through. Both would look like a working route.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_session
from app.deployments import resolve_deployment
from app.main import create_app
from app.orm import Base, Credential, Model, Provider

DECIDER = "pplx-decider:27b"
ANSWER = {
    "model": DECIDER,
    "answers": {
        "shape_1": {
            "type": "choice",
            "choice": "routine_external_background",
            "confidence": 0.87,
            "probabilities": {"routine_external_background": 0.87},
        }
    },
    "usage": {"input_tokens": 874, "output_tokens": 0},
}


async def _store(session):
    credential = Credential(name="local", kind="static", secret="a-secret")
    session.add(credential)
    await session.flush()
    serves = Provider(
        name="ollama",
        label="Ollama",
        base_url="http://127.0.0.1:11434",
        chat_completions_path="/v1/chat/completions",
        systemone_path="/v1/systemone",
        credential_id=credential.id,
    )
    chat_only = Provider(
        name="chat-only",
        label="Chat only",
        base_url="http://127.0.0.1:9999",
        chat_completions_path="/v1/chat/completions",
        credential_id=credential.id,
    )
    session.add_all([serves, chat_only])
    await session.flush()
    session.add_all([
        Model(deployment_name=DECIDER, provider_id=serves.id, context_window=262_144),
        Model(deployment_name="chatty", provider_id=chat_only.id, context_window=8_192),
        Model(deployment_name="reserved-decider", provider_id=serves.id,
              context_window=262_144, admin_only=True),
    ])
    await session.commit()


# --- resolution ------------------------------------------------------------


async def test_the_decision_wire_resolves_to_the_path_the_provider_declares_for_it(
    session,
):
    await _store(session)

    target = await resolve_deployment(session, DECIDER, wire="systemone")

    assert target.upstream_url == "http://127.0.0.1:11434/v1/systemone"


async def test_a_provider_that_does_not_serve_the_wire_refuses_rather_than_using_its_chat_path(
    session,
):
    """A decision call answered as chat is a placement from a chat model."""
    await _store(session)

    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "chatty", wire="systemone")

    assert exc.value.status_code == 503


# --- the route -------------------------------------------------------------


@pytest_asyncio.fixture
async def client(monkeypatch):
    import app.main as main
    import app.proxy.systemone as route

    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as setup:
        await _store(setup)

    async def _session():
        async with maker() as s:
            yield s

    monkeypatch.setattr(route, "validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr(route, "resolve_requester_role", lambda *a, **k: "user")
    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr(main, "resolve_requester_role", lambda *a, **k: "user")
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c
    await engine.dispose()


def _call(**overrides):
    body = {
        "model": DECIDER,
        "state": {"events": [{"id": 1, "raw": "a scanner user-agent on 443"}]},
        "questions": {
            "shape_1": {
                "type": "choice",
                "instructions": "which condition is this evidence of",
                "criteria": {"routine_external_background": "background noise"},
            }
        },
        "keep_alive": "15m",
    }
    body.update(overrides)
    return body


def _captures(monkeypatch, status=200, payload=None, raise_with=None):
    seen: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        if raise_with is not None:
            raise raise_with
        return httpx.Response(
            status, json=ANSWER if payload is None else payload,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return seen


def test_the_state_and_the_questions_travel_to_the_upstream_unaltered(
    client, monkeypatch
):
    """The caller owns the question; the proxy forwards, it does not compose."""
    seen = _captures(monkeypatch)

    response = client.post(
        "/v1/systemone", headers={"Authorization": "Bearer k"}, json=_call()
    )

    assert response.status_code == 200
    assert response.json() == ANSWER
    assert seen["url"] == "http://127.0.0.1:11434/v1/systemone"
    assert seen["json"]["state"]["events"][0]["raw"] == "a scanner user-agent on 443"
    assert seen["json"]["questions"]["shape_1"]["criteria"] == {
        "routine_external_background": "background noise"
    }
    assert seen["json"]["keep_alive"] == "15m"
    assert seen["json"]["model"] == DECIDER


def test_a_call_asking_nothing_is_refused_before_a_model_is_loaded(
    client, monkeypatch
):
    seen = _captures(monkeypatch)

    response = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer k"},
        json=_call(questions={}),
    )

    assert response.status_code == 400
    assert seen == {}


def test_a_model_this_proxy_does_not_serve_on_the_wire_is_refused_not_forwarded(
    client, monkeypatch
):
    seen = _captures(monkeypatch)

    response = client.post(
        "/v1/systemone", headers={"Authorization": "Bearer k"}, json=_call(model="chatty")
    )

    assert response.status_code == 503
    assert seen == {}


def test_an_unreachable_decision_model_is_a_bad_gateway(client, monkeypatch):
    """A refusal reported as an empty answer is a placement nobody made."""
    _captures(
        monkeypatch,
        raise_with=httpx.ConnectError(
            "refused", request=httpx.Request("POST", "http://x")
        ),
    )

    response = client.post(
        "/v1/systemone", headers={"Authorization": "Bearer k"}, json=_call()
    )

    assert response.status_code == 502


def test_the_upstreams_own_refusal_is_passed_through_with_its_status(
    client, monkeypatch
):
    _captures(monkeypatch, status=404, payload={"error": "model not found"})

    response = client.post(
        "/v1/systemone", headers={"Authorization": "Bearer k"}, json=_call()
    )

    assert response.status_code == 404
    assert response.json() == {"error": "model not found"}


def test_the_decision_wire_is_not_public(client, monkeypatch):
    import app.proxy.systemone as route

    monkeypatch.setattr(route, "validate_api_key", lambda *a, **k: False)

    assert client.post("/v1/systemone", json=_call()).status_code == 401


def test_a_reserved_model_is_invisible_to_an_ordinary_key(client, monkeypatch):
    seen = _captures(monkeypatch)

    response = client.post(
        "/v1/systemone",
        headers={"Authorization": "Bearer k"},
        json=_call(model="reserved-decider"),
    )

    assert response.status_code == 404
    assert seen == {}
