"""app.deployments: auth_scheme header building and protocol classification."""

from __future__ import annotations

import pytest

from app.deployments import DeploymentTarget, build_upstream_headers


def _target(target_uri: str, auth_scheme: str = "api_key_header") -> DeploymentTarget:
    return DeploymentTarget(
        name="test-model",
        target_uri=target_uri,
        api_key="secret-key",
        api_version=None,
        auth_scheme=auth_scheme,
    )


# ---- build_upstream_headers ------------------------------------------------


def test_api_key_header_scheme_sends_azure_style_header():
    headers = build_upstream_headers(_target("https://x.services.ai.azure.com/openai/responses"))
    assert headers["api-key"] == "secret-key"
    assert "Authorization" not in headers
    assert "x-api-key" not in headers


def test_bearer_scheme_sends_authorization_header():
    headers = build_upstream_headers(
        _target("https://api.openai.com/v1/responses", auth_scheme="bearer")
    )
    assert headers["Authorization"] == "Bearer secret-key"
    assert "api-key" not in headers
    assert "x-api-key" not in headers


def test_x_api_key_scheme_sends_anthropic_style_header():
    headers = build_upstream_headers(
        _target("https://api.anthropic.com/v1/messages", auth_scheme="x_api_key")
    )
    assert headers["x-api-key"] == "secret-key"
    assert "Authorization" not in headers
    assert "api-key" not in headers


def test_unknown_auth_scheme_raises():
    with pytest.raises(ValueError):
        build_upstream_headers(_target("https://example.com", auth_scheme="carrier-pigeon"))


def _fake_codex_jwt(account_id: str = "acct-42") -> str:
    import base64
    import json as _json

    def seg(obj) -> str:
        return base64.urlsafe_b64encode(_json.dumps(obj).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'none'})}.{seg({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.s"


def test_codex_backend_gets_chatgpt_identity_headers():
    target = DeploymentTarget(
        name="gpt-5.4",
        target_uri="https://chatgpt.com/backend-api/codex/responses",
        api_key=_fake_codex_jwt(),
        api_version=None,
        auth_scheme="bearer",
        credential_kind="openai_oauth",
    )
    headers = build_upstream_headers(target)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["chatgpt-account-id"] == "acct-42"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["originator"] == "codex_cli_rs"


def test_real_openai_host_gets_no_identity_headers():
    headers = build_upstream_headers(
        _target("https://api.openai.com/v1/responses", auth_scheme="bearer")
    )
    assert "chatgpt-account-id" not in headers
    assert "OpenAI-Beta" not in headers


def test_codex_backend_classifies_as_responses_protocol():
    t = _target("https://chatgpt.com/backend-api/codex/responses")
    assert t.is_codex_backend
    assert t.is_openai_responses
    assert not t.is_anthropic_messages
    assert not t.is_native_chat_completions


def test_ollama_anthropic_compat_classifies_as_messages():
    t = _target("http://192.168.4.64:11434/v1/messages")
    assert t.is_anthropic_messages
    assert not t.is_openai_responses
    assert not t.is_native_chat_completions


def test_ollama_chat_completions_classifies_natively():
    t = _target("http://192.168.4.64:11434/v1/chat/completions")
    assert t.is_native_chat_completions
    assert not t.is_anthropic_messages


def test_ollama_responses_classifies_as_responses_protocol():
    t = _target("http://192.168.4.64:11434/v1/responses")
    assert t.is_openai_responses
    assert not t.is_codex_backend
    assert not t.is_anthropic_messages


# ---- protocol classification: Foundry-shaped URIs (existing behavior) -----


def test_foundry_anthropic_messages_still_classifies():
    t = _target("https://x.services.ai.azure.com/anthropic/v1/messages")
    assert t.is_anthropic_messages
    assert not t.is_openai_responses
    assert not t.is_native_chat_completions


def test_foundry_openai_responses_still_classifies():
    t = _target("https://x.services.ai.azure.com/openai/responses")
    assert t.is_openai_responses
    assert not t.is_anthropic_messages
    assert not t.is_native_chat_completions


def test_foundry_native_chat_completions_still_classifies():
    t = _target("https://x.services.ai.azure.com/models/chat/completions")
    assert t.is_native_chat_completions
    assert not t.is_openai_responses
    assert not t.is_anthropic_messages


# ---- protocol classification: real provider hostnames (new) ---------------


def test_real_anthropic_messages_classifies():
    t = _target("https://api.anthropic.com/v1/messages", auth_scheme="x_api_key")
    assert t.is_anthropic_messages
    assert not t.is_openai_responses
    assert not t.is_native_chat_completions


def test_real_openai_responses_classifies():
    t = _target("https://api.openai.com/v1/responses", auth_scheme="bearer")
    assert t.is_openai_responses
    assert not t.is_anthropic_messages
    assert not t.is_native_chat_completions


def test_real_openai_chat_completions_classifies():
    t = _target("https://api.openai.com/v1/chat/completions", auth_scheme="bearer")
    assert t.is_native_chat_completions
    assert not t.is_openai_responses
    assert not t.is_anthropic_messages


def test_real_anthropic_host_without_messages_path_does_not_classify():
    t = _target("https://api.anthropic.com/v1/complete", auth_scheme="x_api_key")
    assert not t.is_anthropic_messages
