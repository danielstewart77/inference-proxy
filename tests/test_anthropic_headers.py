"""app.proxy.anthropic: header building per auth_scheme, beta-flag filtering."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.requests import Request

from app.deployments import DeploymentTarget
from app.proxy.anthropic import _forward_headers


def _request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope)


def _target(target_uri: str, auth_scheme: str) -> DeploymentTarget:
    return DeploymentTarget(
        name="test-model",
        target_uri=target_uri,
        api_key="secret-key",
        api_version=None,
        auth_scheme=auth_scheme,
    )


def test_foundry_target_uses_bearer_and_drops_anthropic_beta():
    target = _target("https://x.services.ai.azure.com/anthropic/v1/messages", "bearer")
    req = _request({"anthropic-beta": "some-beta-flag", "anthropic-version": "2099-01-01"})
    headers = _forward_headers(req, target)
    assert headers["Authorization"] == "Bearer secret-key"
    assert "anthropic-beta" not in headers
    # client-supplied anthropic-version overrides our default
    assert headers["anthropic-version"] == "2099-01-01"


def test_real_anthropic_target_uses_x_api_key_and_keeps_anthropic_beta():
    target = _target("https://api.anthropic.com/v1/messages", "x_api_key")
    req = _request({"anthropic-beta": "some-beta-flag"})
    headers = _forward_headers(req, target)
    assert headers["x-api-key"] == "secret-key"
    assert "Authorization" not in headers
    assert headers["anthropic-beta"] == "some-beta-flag"
    assert headers["anthropic-version"] == "2023-06-01"
