"""A model authenticates with its provider's credential and auth scheme.

Changing a provider's authentication has to move every model under it. The
model row still carries `credential_id` and `auth_scheme` columns from before
providers existed, so each test here gives the model a legacy value of its
own and checks that nothing reads it.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.deployments import resolve_deployment
from app.orm import Credential, Model, Provider


async def _credential(session, name: str, *, kind: str = "static") -> Credential:
    credential = Credential(name=name, kind=kind, secret=f"{name}-secret")
    session.add(credential)
    await session.flush()
    return credential


async def _anthropic(session, credential: Credential | None) -> Provider:
    provider = Provider(
        name="anthropic",
        base_url="https://api.anthropic.com",
        messages_path="/v1/messages",
        credential_id=credential.id if credential else None,
        auth_scheme="x_api_key",
        enabled=True,
    )
    session.add(provider)
    await session.flush()
    return provider


async def _resolved_auth(session, name: str) -> tuple[str, str, str]:
    target = await resolve_deployment(session, name, wire="anthropic_messages")
    return target.api_key, target.auth_scheme, target.credential_kind


async def test_changing_a_providers_authentication_changes_it_for_every_model(session):
    legacy = await _credential(session, "legacy")
    api_key = await _credential(session, "api")
    oauth = await _credential(session, "oauth", kind="anthropic_oauth")
    provider = await _anthropic(session, api_key)
    for name in ("claude-opus-5", "claude-haiku-4-5"):
        session.add(Model(deployment_name=name, provider_id=provider.id,
                          credential_id=legacy.id, auth_scheme="bearer", enabled=True))
    await session.commit()

    for name in ("claude-opus-5", "claude-haiku-4-5"):
        assert await _resolved_auth(session, name) == ("api-secret", "x_api_key", "static")

    provider.credential = oauth
    provider.auth_scheme = "bearer"
    await session.commit()

    for name in ("claude-opus-5", "claude-haiku-4-5"):
        assert await _resolved_auth(session, name) == (
            "oauth-secret", "bearer", "anthropic_oauth")


async def test_a_model_without_a_provider_is_unavailable_whatever_it_carries(session):
    legacy = await _credential(session, "legacy")
    session.add(Model(deployment_name="claude-opus-5",
                      target_uri="https://api.anthropic.com/v1/messages",
                      credential_id=legacy.id, auth_scheme="x_api_key", enabled=True))
    await session.commit()

    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "claude-opus-5", wire="anthropic_messages")
    assert exc.value.status_code == 503


async def test_a_provider_with_no_credential_does_not_fall_back_to_the_models(session):
    legacy = await _credential(session, "legacy")
    provider = await _anthropic(session, None)
    session.add(Model(deployment_name="claude-opus-5", provider_id=provider.id,
                      credential_id=legacy.id, enabled=True))
    await session.commit()

    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "claude-opus-5", wire="anthropic_messages")
    assert exc.value.status_code == 503
    assert "credential unavailable" in exc.value.detail["error"]["message"]
