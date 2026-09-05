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

    claude = _names(await listing_for(session, harness="claude", is_admin=False))
    codex = _names(await listing_for(session, harness="codex", is_admin=False))

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

    listed = (await listing_for(session, harness="codex", is_admin=False))["data"]
    row = next(item for item in listed if item["id"] == "mistral-large")
    assert row["provider"] == "mistral"
    assert row["provider_label"] == "Mistral"


async def test_a_provider_that_cannot_speak_a_shape_hides_its_models(session):
    """Anthropic serves Messages only, so its models never reach a Codex list."""
    provider = await _anthropic(session)
    session.add(Model(deployment_name="claude-opus-5", provider_id=provider.id))
    await session.commit()

    assert _names(await listing_for(session, harness="claude", is_admin=False)) == [
        "claude-opus-5"
    ]
    assert _names(await listing_for(session, harness="codex", is_admin=False)) == []


async def test_the_listing_reports_the_label_a_picker_shows(session):
    """Requirement 2 — the display name travels with the deployment name."""
    provider = await _anthropic(session)
    session.add(
        Model(deployment_name="claude-opus-5", label="Opus 5", provider_id=provider.id)
    )
    await session.commit()

    row = (await listing_for(session, harness="claude", is_admin=False))["data"][0]
    assert (row["id"], row["label"]) == ("claude-opus-5", "Opus 5")


async def test_disabling_a_provider_takes_its_models_with_it(session):
    """A switched-off upstream must not be reachable through a stale URI.

    A migrated row still carries the target URI it had before providers
    existed. Falling back to it would send traffic to the upstream that was
    just disabled — and that URI serves one shape while the row is listed for
    every shape the provider serves, so a Codex harness would be offered a
    model only a Claude harness could address.
    """
    provider = await _ollama(session)
    session.add(
        Model(
            deployment_name="qwen35-131k",
            provider_id=provider.id,
            target_uri="http://192.168.4.64:11434/v1/messages",
        )
    )
    await session.commit()
    provider.enabled = False
    await session.commit()

    assert _names(await listing_for(session, harness="claude", is_admin=False)) == []
    assert _names(await listing_for(session, harness="codex", is_admin=False)) == []

    with pytest.raises(Exception) as refused:
        await resolve_deployment(session, "qwen35-131k", wire="anthropic_messages")
    assert refused.value.status_code == 503


# ---------------------------------------------------------------------------
# One merged listing (story 02)
# ---------------------------------------------------------------------------


async def test_one_listing_returns_every_kind_of_upstream_together(session):
    """Requirement 1 — Anthropic, OpenAI and Ollama models in one response."""
    ollama = await _ollama(session)
    anthropic = await _anthropic(session)
    openai = Provider(
        name="openai",
        label="OpenAI",
        base_url="https://chatgpt.example",
        responses_path="/backend-api/codex/responses",
    )
    session.add(openai)
    await session.flush()
    session.add(Model(deployment_name="claude-opus-5", provider_id=anthropic.id))
    session.add(Model(deployment_name="gpt-5.6", provider_id=openai.id))
    session.add(Model(deployment_name="qwen35-131k", provider_id=ollama.id))
    await session.commit()

    assert _names(await listing_for(session, is_admin=False)) == [
        "claude-opus-5",
        "gpt-5.6",
        "qwen35-131k",
    ]


async def test_each_model_names_the_shapes_it_can_be_reached_on(session):
    """Requirement 2 — the row carries its wires, so a caller can route on it."""
    ollama = await _ollama(session)
    anthropic = await _anthropic(session)
    session.add(Model(deployment_name="qwen35-131k", provider_id=ollama.id))
    session.add(Model(deployment_name="claude-opus-5", provider_id=anthropic.id))
    await session.commit()

    rows = {r["id"]: r["wires"] for r in (await listing_for(session, is_admin=False))["data"]}

    assert rows["qwen35-131k"] == [
        "anthropic_messages",
        "openai_responses",
        "chat_completions",
    ]
    assert rows["claude-opus-5"] == ["anthropic_messages"]


async def test_a_claude_caller_is_not_offered_a_responses_only_model(session):
    """Requirement 4a — a picker never shows what its harness cannot send."""
    openai = Provider(
        name="openai",
        label="OpenAI",
        base_url="https://chatgpt.example",
        responses_path="/backend-api/codex/responses",
    )
    session.add(openai)
    await session.flush()
    session.add(Model(deployment_name="gpt-5.6", provider_id=openai.id))
    await session.commit()

    assert _names(await listing_for(session, harness="claude", is_admin=False)) == []
    assert _names(await listing_for(session, harness="codex", is_admin=False)) == ["gpt-5.6"]


async def test_a_codex_caller_is_not_offered_a_messages_only_model(session):
    """Requirement 4b — the reverse direction breaks independently."""
    anthropic = await _anthropic(session)
    session.add(Model(deployment_name="claude-opus-5", provider_id=anthropic.id))
    await session.commit()

    assert _names(await listing_for(session, harness="codex", is_admin=False)) == []
    assert _names(await listing_for(session, harness="claude", is_admin=False)) == [
        "claude-opus-5"
    ]


async def test_a_withheld_model_is_absent_from_the_harness_named_out(session):
    """Requirement 5a — withholding still bites on the merged listing."""
    provider = await _ollama(session)
    session.add(
        Model(deployment_name="skippy-harness", provider_id=provider.id, harnesses="claude")
    )
    await session.commit()

    assert _names(await listing_for(session, harness="codex", is_admin=False)) == []


async def test_hermes_is_offered_only_the_wires_the_grant_covers(session):
    """Requirement 5a — a listing may not offer what the request path refuses.

    Hermes speaks all three shapes and cannot be granted any of them by name:
    the admin form takes claude and codex only. Judging its listing by its own
    name therefore hid every withheld model from it while the request path,
    which sees only the shape, let it through. Judged per wire, the two agree.
    """
    provider = await _ollama(session)
    session.add(
        Model(deployment_name="skippy-harness", provider_id=provider.id, harnesses="claude")
    )
    await session.commit()

    listing = await listing_for(session, harness="hermes", is_admin=False)

    assert _names(listing) == ["skippy-harness"]
    assert listing["data"][0]["wires"] == ["anthropic_messages"]


async def test_a_withheld_model_is_still_offered_to_the_harness_it_names(session):
    """Requirement 5b — withholding must not hide it from everyone."""
    provider = await _ollama(session)
    session.add(
        Model(deployment_name="skippy-harness", provider_id=provider.id, harnesses="claude")
    )
    await session.commit()

    assert _names(await listing_for(session, harness="claude", is_admin=False)) == [
        "skippy-harness"
    ]


async def test_hermes_is_offered_every_shape_of_model(session):
    """Requirement 1, from the caller that speaks all three wires."""
    ollama = await _ollama(session)
    anthropic = await _anthropic(session)
    openai = Provider(
        name="openai",
        label="OpenAI",
        base_url="https://chatgpt.example",
        responses_path="/backend-api/codex/responses",
    )
    session.add(openai)
    await session.flush()
    session.add(Model(deployment_name="claude-opus-5", provider_id=anthropic.id))
    session.add(Model(deployment_name="gpt-5.6", provider_id=openai.id))
    session.add(Model(deployment_name="qwen35-131k", provider_id=ollama.id))
    await session.commit()

    assert _names(await listing_for(session, harness="hermes", is_admin=False)) == [
        "claude-opus-5",
        "gpt-5.6",
        "qwen35-131k",
    ]
