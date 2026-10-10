"""A caller that names its harness is judged as that harness.

dsh and Codex both send chat completions, so the endpoint alone judged every
dsh request as Codex: a model listed for dsh and withheld from Codex was
offered in dsh's picker and then refused on its first turn. dsh's hive
profile now declares itself on every request, and the request path honours
the declaration.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.admin.models import _parse_harnesses
from app.db import get_session
from app.deployments import HARNESS_HEADER
from app.main import create_app
from app.orm import Base, Credential, Model, Provider


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app with one native chat upstream serving a model granted to dsh."""
    import app.main as main
    import app.proxy.chat_completions as chat

    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as setup:
        credential = Credential(name="local", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        provider = Provider(
            name="llama",
            label="llama.cpp",
            base_url="http://192.168.4.64:8080",
            chat_completions_path="/v1/chat/completions",
            credential_id=credential.id,
        )
        setup.add(provider)
        await setup.flush()
        setup.add(
            Model(deployment_name="qwen35-dsh", provider_id=provider.id, harnesses="dsh")
        )
        await setup.commit()

    async def _session():
        async with maker() as s:
            yield s

    async def _answer(url, headers, body, *, stream, log_prefix):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
            request=httpx.Request("POST", url),
        )

    for module in (main, chat):
        monkeypatch.setattr(module, "validate_api_key", lambda *a, **k: True)
        monkeypatch.setattr(module, "resolve_requester_role", lambda *a, **k: "user")
    monkeypatch.setattr(chat, "resolve_principal", lambda *a, **k: None)
    monkeypatch.setattr(chat, "post_with_retries", _answer)
    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        yield c
    await engine.dispose()


def _ask(client, **headers):
    return client.post(
        "/v1/chat/completions",
        json={"model": "qwen35-dsh", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer test", **headers},
    )


def test_a_request_identified_as_dsh_is_served_a_model_withheld_from_codex(client):
    """Test 27 — the declared harness, not the wire, decides withholding."""
    served = _ask(client, **{HARNESS_HEADER: "dsh"})
    anonymous = _ask(client)

    assert served.status_code == 200
    assert served.json()["choices"][0]["message"]["content"] == "hello"
    # Unidentified, the same request is still judged as its wire's default.
    assert anonymous.status_code == 404


def test_the_listing_offers_dsh_what_its_requests_will_be_served(client):
    """The picker and the request path agree: dsh sees it, codex does not."""
    dsh = [row["id"] for row in client.get("/v1/models?harness=dsh").json()["data"]]
    codex = [row["id"] for row in client.get("/v1/models?harness=codex").json()["data"]]

    assert dsh == ["qwen35-dsh"]
    assert codex == []


def test_the_admin_form_can_withhold_a_model_from_dsh():
    """A model can be granted to claude and codex and so withheld from dsh."""
    assert _parse_harnesses("Codex, dsh") == "codex,dsh"
    with pytest.raises(HTTPException) as refused:
        _parse_harnesses("dhs")
    assert refused.value.status_code == 400
