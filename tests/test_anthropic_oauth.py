"""An Anthropic subscription OAuth token needs more than an auth header.

Without the OAuth beta flag and the CLI identity system block, the Messages API
answers 429 rather than 401 — so both are asserted here.
"""

from __future__ import annotations

from app.deployments import DeploymentTarget
from app.proxy.anthropic import (
    OAUTH_BETA_FLAG,
    OAUTH_SYSTEM_IDENTITY,
    _forward_headers,
    apply_oauth_system_prompt,
)


class _FakeRequest:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


def _target(*, credential_kind: str = "static", auth_scheme: str = "bearer") -> DeploymentTarget:
    return DeploymentTarget(
        name="claude-opus-4-8",
        target_uri="https://api.anthropic.com/v1/messages",
        api_key="secret-token",
        api_version=None,
        auth_scheme=auth_scheme,
        credential_kind=credential_kind,
    )


# ---- headers ---------------------------------------------------------------


def test_oauth_target_gets_the_beta_flag():
    headers = _forward_headers(_FakeRequest(), _target(credential_kind="anthropic_oauth"))
    assert OAUTH_BETA_FLAG in headers["anthropic-beta"]
    assert headers["Authorization"] == "Bearer secret-token"


def test_static_target_gets_no_beta_flag():
    headers = _forward_headers(_FakeRequest(), _target())
    assert "anthropic-beta" not in headers


def test_client_beta_flags_survive_alongside_the_oauth_flag():
    request = _FakeRequest({"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"})
    headers = _forward_headers(request, _target(credential_kind="anthropic_oauth"))
    flags = headers["anthropic-beta"].split(",")
    assert OAUTH_BETA_FLAG in flags
    assert "fine-grained-tool-streaming-2025-05-14" in flags


def test_the_oauth_flag_is_not_duplicated():
    request = _FakeRequest({"anthropic-beta": OAUTH_BETA_FLAG})
    headers = _forward_headers(request, _target(credential_kind="anthropic_oauth"))
    assert headers["anthropic-beta"].split(",").count(OAUTH_BETA_FLAG) == 1


def test_foundry_target_still_drops_client_beta_flags():
    request = _FakeRequest({"anthropic-beta": "some-flag-foundry-hates"})
    target = DeploymentTarget(
        name="claude-opus-4-8",
        target_uri="https://x.services.ai.azure.com/anthropic/v1/messages",
        api_key="k",
        api_version=None,
        auth_scheme="bearer",
    )
    assert "anthropic-beta" not in _forward_headers(request, target)


def test_real_anthropic_x_api_key_target_keeps_client_beta_flags():
    request = _FakeRequest({"anthropic-beta": "a-real-beta"})
    headers = _forward_headers(request, _target(auth_scheme="x_api_key"))
    assert headers["anthropic-beta"] == "a-real-beta"


# ---- system prompt ---------------------------------------------------------


def test_identity_block_is_added_when_there_is_no_system():
    body = apply_oauth_system_prompt({"messages": []})
    assert body["system"] == [{"type": "text", "text": OAUTH_SYSTEM_IDENTITY}]


def test_string_system_is_normalised_and_preserved_after_the_identity():
    body = apply_oauth_system_prompt({"system": "You are a helpful bee."})
    assert body["system"][0]["text"] == OAUTH_SYSTEM_IDENTITY
    assert body["system"][1] == {"type": "text", "text": "You are a helpful bee."}


def test_block_list_system_is_preserved_after_the_identity():
    original = [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}]
    body = apply_oauth_system_prompt({"system": list(original)})
    assert body["system"][0]["text"] == OAUTH_SYSTEM_IDENTITY
    assert body["system"][1:] == original


def test_identity_is_not_prepended_twice():
    once = apply_oauth_system_prompt({"system": "hello"})
    twice = apply_oauth_system_prompt(dict(once))
    assert twice["system"] == once["system"]
