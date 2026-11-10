"""`POST /v1/embeddings`: a client key's request reaches the embedding model's
provider and the provider's vectors come back. Only the transport is stubbed."""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from app import cli


async def test_embeddings_with_a_client_key_forward_to_the_ollama_provider_and_return_its_vectors(
    db_maker, monkeypatch
):
    import app.azure
    import app.main as main

    async with db_maker() as s:
        await cli.add_provider(s, name="ollama", kind="ollama", base_url="http://ollama:11434")
        await cli.add_model(s, provider="ollama", model="qwen3-embedding:8b",
                            context_window=32768, purpose="embedding")
        key = (await cli.mint_key(s, client="lucent", kind="app"))["key"]

    vectors = [[0.25, -0.5, 0.125], [1.0, 0.0, -1.0]]
    sent: list[httpx.Request] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={
            "object": "list",
            "model": "qwen3-embedding:8b",
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        })

    async def _no_probe():
        return None

    monkeypatch.setattr(main, "connectivity_probe_loop", _no_probe)
    monkeypatch.setattr(
        app.azure, "_shared_client", httpx.AsyncClient(transport=httpx.MockTransport(ollama)))

    with TestClient(main.create_app()) as proxy:
        answer = proxy.post(
            "/v1/embeddings",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "qwen3-embedding:8b", "input": ["alpha", "beta"]},
        )

    assert answer.status_code == 200, answer.text
    assert [row["embedding"] for row in answer.json()["data"]] == vectors
    assert [str(r.url) for r in sent] == ["http://ollama:11434/v1/embeddings"]
    assert json.loads(sent[0].content) == {
        "model": "qwen3-embedding:8b", "input": ["alpha", "beta"]}
