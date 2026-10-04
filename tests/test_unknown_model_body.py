"""The refusal body an OpenAI-compatible client actually parses.

``resolve_deployment`` builds the envelope; the app's own
``StarletteHTTPException`` handler decides what reaches the wire. Asserting
the raised exception alone leaves the handler free to bury the envelope under
``detail``, where no SDK looks — the failure mode this file exists to catch.
"""

from __future__ import annotations

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_session
from app.main import create_app
from app.orm import Base


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app on an empty database: every model name is unregistered."""
    import app.main as main
    import app.proxy.chat_completions as chat_completions
    import app.proxy.anthropic as anthropic

    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def _session():
        async with maker() as s:
            yield s

    # The credential is not what this file is about; each route authenticates
    # for itself, so each route's own check is the one stubbed.
    for module in (main, chat_completions, anthropic):
        monkeypatch.setattr(module, "validate_api_key", lambda *a, **k: True)
        monkeypatch.setattr(module, "resolve_requester_role", lambda *a, **k: "user")
    for module in (chat_completions, anthropic):
        monkeypatch.setattr(module, "resolve_principal", lambda *a, **k: (1, 1))
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c
    await engine.dispose()


def test_chat_completions_names_the_model_where_the_openai_sdk_reads_it(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model-9f3a", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == 404
    body = response.json()
    # Top level, not nested under `detail`: this is the path the SDK reads.
    assert "no-such-model-9f3a" in body["error"]["message"]


def test_messages_names_the_model_where_an_anthropic_sdk_reads_it(client):
    response = client.post(
        "/v1/messages",
        json={"model": "no-such-model-9f3a", "max_tokens": 8,
              "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert "no-such-model-9f3a" in body["error"]["message"]
