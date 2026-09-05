"""The merged listing over HTTP, not just the function beneath it.

Requirements 1, 2, 4, 5 and 12 were all asserted against ``listing_for``
called in-process. Everything the route itself does — authenticating the
caller, deciding whether the key is privileged, and passing the harness
through — sat below the tests, so the route could hand a typo'd harness the
full union, or drop the parameter entirely, with the suite green.
"""

from __future__ import annotations

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_session
from app.main import create_app
from app.orm import Base, Credential, Model, Provider


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app, on a fresh in-memory database, with a key that validates."""
    import app.main as main

    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as setup:
        credential = Credential(name="local", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        provider = Provider(
            name="ollama",
            label="Ollama",
            base_url="http://192.168.4.64:11434",
            messages_path="/v1/messages",
            responses_path="/v1/responses",
            chat_completions_path="/v1/chat/completions",
            credential_id=credential.id,
        )
        setup.add(provider)
        await setup.flush()
        setup.add(
            Model(
                deployment_name="claude-only",
                provider_id=provider.id,
                harnesses="claude",
            )
        )
        setup.add(Model(deployment_name="everyones", provider_id=provider.id))
        setup.add(
            Model(
                deployment_name="reserved",
                provider_id=provider.id,
                admin_only=True,
            )
        )
        await setup.commit()

    async def _session():
        async with maker() as s:
            yield s

    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr(main, "resolve_requester_role", lambda *a, **k: "user")
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c
    await engine.dispose()


def _names(response) -> list[str]:
    return [row["id"] for row in response.json()["data"]]


def test_the_route_hands_the_harness_to_the_listing(client):
    """Requirement 5 — the caller's own name reaches the withholding check."""
    codex = client.get("/v1/models?harness=codex")
    claude = client.get("/v1/models?harness=claude")

    assert "claude-only" not in _names(codex)
    assert "claude-only" in _names(claude)


def test_the_route_refuses_a_harness_it_does_not_know(client):
    """Requirement 5 — a typo is refused, never widened to the union."""
    response = client.get("/v1/models?harness=cladue")

    assert response.status_code == 400


def test_an_unprivileged_key_is_not_offered_reserved_models(client):
    """Requirement 12 — admin-only rows stay out of an ordinary listing."""
    assert "reserved" not in _names(client.get("/v1/models"))


def test_the_listing_is_not_public(client, monkeypatch):
    """Requirement 1 — a listing is what this key may address, so it needs one."""
    import app.main as main

    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: False)

    assert client.get("/v1/models").status_code == 401
