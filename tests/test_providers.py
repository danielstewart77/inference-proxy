"""Providers own the upstream; models are listed per harness.

The three behaviours here are the ones the model rows could not express while
each carried a complete target URI: one local model serving both request
shapes, a model withheld from one harness, and a brand-new upstream becoming
selectable without a code change.
"""

from __future__ import annotations

import pytest

from app.deployments import listing_for, resolve_deployment
from app.orm import Credential, Model, Provider

pytestmark = pytest.mark.asyncio


async def _ollama(session) -> Provider:
    credential = Credential(name="local", kind="static", secret="s")
    session.add(credential)
    await session.flush()
    provider = Provider(
        name="ollama",
        label="Ollama",
        base_url="http://192.168.4.64:11434",
        messages_path="/v1/messages",
        responses_path="/v1/responses",
        chat_completions_path="/v1/chat/completions",
        credential_id=credential.id,
    )
    session.add(provider)
    await session.flush()
    return provider


async def _anthropic(session) -> Provider:
    credential = Credential(name="anthropic", kind="static", secret="s")
    session.add(credential)
    await session.flush()
    provider = Provider(
        name="anthropic",
        label="Anthropic",
        base_url="https://api.anthropic.com",
        messages_path="/v1/messages",
        credential_id=credential.id,
    )
    session.add(provider)
    await session.flush()
    return provider


def _names(payload: dict) -> list[str]:
    return [row["id"] for row in payload["data"]]


async def test_one_local_model_answers_both_request_shapes(session):
    """Requirement 5 — the endpoint decides the path, not the model row."""
    provider = await _ollama(session)
    session.add(Model(deployment_name="qwen35-131k", provider_id=provider.id))
    await session.commit()

    claude = await resolve_deployment(session, "qwen35-131k", wire="anthropic_messages")
    codex = await resolve_deployment(session, "qwen35-131k", wire="openai_responses")

    assert claude.target_uri == "http://192.168.4.64:11434/v1/messages"
    assert codex.target_uri == "http://192.168.4.64:11434/v1/responses"


async def test_a_model_is_withheld_only_from_the_harness_named_out(session):
    """Requirement 6 — unset means every harness; a list withholds."""
    provider = await _ollama(session)
    session.add(Model(deployment_name="qwen35-131k", provider_id=provider.id))
    session.add(
        Model(
            deployment_name="skippy-harness",
            provider_id=provider.id,
            harnesses="claude",
        )
    )
    await session.commit()

    claude = _names(await listing_for(session, wire="anthropic_messages", is_admin=False))
    codex = _names(await listing_for(session, wire="openai_responses", is_admin=False))

    assert claude == ["qwen35-131k", "skippy-harness"]
    assert codex == ["qwen35-131k"]

    with pytest.raises(Exception) as refused:
        await resolve_deployment(session, "skippy-harness", wire="openai_responses")
    assert refused.value.status_code == 404


async def test_a_new_provider_and_model_are_offered_with_no_code_change(session):
    """Requirement 12 — adding an upstream is two rows."""
    await _ollama(session)  # an unrelated upstream already registered
    mistral = Provider(
        name="mistral",
        label="Mistral",
        base_url="https://api.mistral.example",
        chat_completions_path="/v1/chat/completions",
        responses_path="/v1/responses",
    )
    session.add(mistral)
    await session.flush()
    session.add(Model(deployment_name="mistral-large", provider_id=mistral.id))
    await session.commit()

    listed = (await listing_for(session, wire="openai_responses", is_admin=False))["data"]
    row = next(item for item in listed if item["id"] == "mistral-large")
    assert row["provider"] == "mistral"
    assert row["provider_label"] == "Mistral"


async def test_a_provider_that_cannot_speak_a_shape_hides_its_models(session):
    """Anthropic serves Messages only, so its models never reach a Codex list."""
    provider = await _anthropic(session)
    session.add(Model(deployment_name="claude-opus-5", provider_id=provider.id))
    await session.commit()

    assert _names(await listing_for(session, wire="anthropic_messages", is_admin=False)) == [
        "claude-opus-5"
    ]
    assert _names(await listing_for(session, wire="openai_responses", is_admin=False)) == []


async def test_the_listing_reports_the_label_a_picker_shows(session):
    """Requirement 2 — the display name travels with the deployment name."""
    provider = await _anthropic(session)
    session.add(
        Model(deployment_name="claude-opus-5", label="Opus 5", provider_id=provider.id)
    )
    await session.commit()

    row = (await listing_for(session, wire="anthropic_messages", is_admin=False))["data"][0]
    assert (row["id"], row["label"]) == ("claude-opus-5", "Opus 5")
