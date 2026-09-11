"""`/v1/providers/limits` over HTTP, and the context window on the listing.

The parse is covered in `test_provider_limits.py`. What these pin is the
part a console actually consumes: that the reading is attributed to the
right provider row, that a provider nobody has called is reported as *not
reporting* rather than as idle at zero, that the route refuses an unknown
key, and that a model's declared context window is relayed — including the
null that means nobody declared one.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import provider_limits
from app.db import get_session
from app.main import create_app
from app.orm import Base, Credential, Model, Provider


ANTHROPIC_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.42",
    "anthropic-ratelimit-unified-5h-reset": "1789110000",
}
CODEX_HEADERS = {
    "x-codex-primary-used-percent": "13",
    "x-codex-primary-window-minutes": "300",
}


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app on a fresh in-memory db, with two providers and a valid key."""
    import app.main as main

    provider_limits.reset()
    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as setup:
        credential = Credential(name="c", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        anthropic = Provider(
            name="anthropic",
            label="Anthropic",
            base_url="https://api.anthropic.com",
            messages_path="/v1/messages",
            credential_id=credential.id,
        )
        openai = Provider(
            name="openai",
            label="Codex",
            base_url="https://chatgpt.com",
            responses_path="/backend-api/codex/responses",
            credential_id=credential.id,
        )
        setup.add_all([anthropic, openai])
        await setup.flush()
        setup.add_all(
            [
                Model(
                    deployment_name="claude-opus-5",
                    provider_id=anthropic.id,
                    enabled=True,
                    context_window=1_000_000,
                ),
                Model(
                    deployment_name="gpt-5.6",
                    provider_id=openai.id,
                    enabled=True,
                    context_window=None,
                ),
            ]
        )
        await setup.commit()

    async def _override():
        async with maker() as s:
            yield s

    app = create_app()
    app.dependency_overrides[get_session] = _override
    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: True, raising=False)
    monkeypatch.setattr("app.main.validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr("app.main.resolve_requester_role", lambda *a, **k: "admin")
    yield TestClient(app)
    provider_limits.reset()
    await engine.dispose()


def _by_name(payload: dict) -> dict:
    return {row["name"]: row for row in payload["providers"]}


def test_each_provider_carries_only_the_reading_from_its_own_host(client):
    """Two providers, two different numbers, neither borrowed nor summed."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)
    provider_limits.record("chatgpt.com", CODEX_HEADERS)

    providers = _by_name(client.get("/v1/providers/limits").json())

    assert providers["anthropic"]["windows"][0]["used_percent"] == 42.0
    assert providers["openai"]["windows"][0]["used_percent"] == 13.0


def test_a_provider_that_has_taken_no_traffic_says_so_rather_than_reading_zero(client):
    """Requirement 12. An empty bar and an unobserved provider are opposite
    claims, and only one of them means there is headroom."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)

    providers = _by_name(client.get("/v1/providers/limits").json())

    assert providers["openai"]["reporting"] is False
    assert providers["openai"]["windows"] == []
    assert providers["anthropic"]["reporting"] is True


def test_every_configured_provider_appears_even_when_silent(client):
    """Dropping a silent provider would read as 'not configured', which sends
    someone to add a row that already exists."""
    providers = _by_name(client.get("/v1/providers/limits").json())

    assert set(providers) == {"anthropic", "openai"}


def test_a_reading_reports_its_age(client):
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)

    row = _by_name(client.get("/v1/providers/limits").json())["anthropic"]

    assert row["age_seconds"] is not None
    assert row["observed_at"] is not None


def test_an_unknown_key_is_refused(client, monkeypatch):
    """The report names what the hive is spending and how close it is to
    being cut off; it answers on the same port the models listing does."""
    monkeypatch.setattr("app.main.validate_api_key", lambda *a, **k: False)

    assert client.get("/v1/providers/limits").status_code == 401


def test_the_listing_relays_a_declared_context_window(client):
    """Requirement 9's denominator. The proxy owns the model map, so this is
    where the number lives."""
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}

    assert models["claude-opus-5"]["context_window"] == 1_000_000


def test_an_undeclared_context_window_relays_as_null_not_zero(client):
    """Requirement 10c. Null is what lets a reader render 'unknown'; a zero
    would be divided into and render every conversation as infinitely full."""
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}

    assert models["gpt-5.6"]["context_window"] is None
