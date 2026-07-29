"""app.deployments.resolve_deployment: credential wiring and access gating."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.deployments import resolve_deployment
from app.orm import Credential, Model


async def _add_model(session, **overrides) -> Model:
    cred = Credential(name=f"cred-{overrides.get('deployment_name', 'x')}", kind="static",
                      secret="upstream-secret")
    session.add(cred)
    await session.flush()
    fields = {
        "deployment_name": "claude-opus",
        "target_uri": "https://api.anthropic.com/v1/messages",
        "credential_id": cred.id,
        "auth_scheme": "x_api_key",
        "enabled": True,
        "admin_only": False,
    }
    fields.update(overrides)
    model = Model(**fields)
    session.add(model)
    await session.commit()
    return model


async def test_resolves_the_secret_from_the_attached_credential(session):
    await _add_model(session)
    target = await resolve_deployment(session, "claude-opus")
    assert target.api_key == "upstream-secret"
    assert target.auth_scheme == "x_api_key"
    assert target.is_anthropic_messages


async def test_model_with_no_credential_is_unavailable_not_unauthenticated(session):
    await _add_model(session, credential_id=None)
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "claude-opus")
    assert exc.value.status_code == 503


async def test_missing_model_name_is_a_400(session):
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, None)
    assert exc.value.status_code == 400


async def test_unknown_model_is_a_404(session):
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "not-registered")
    assert exc.value.status_code == 404


async def test_disabled_model_is_a_404(session):
    await _add_model(session, enabled=False)
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "claude-opus")
    assert exc.value.status_code == 404


async def test_admin_only_model_is_indistinguishable_from_missing(session):
    await _add_model(session, admin_only=True)
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "claude-opus", is_admin=False)
    assert exc.value.status_code == 404
    # ...but resolves for a privileged caller.
    target = await resolve_deployment(session, "claude-opus", is_admin=True)
    assert target.name == "claude-opus"


async def test_anthropic_error_shape_is_an_envelope(session):
    with pytest.raises(HTTPException) as exc:
        await resolve_deployment(session, "nope", error_kind="anthropic")
    assert exc.value.detail["type"] == "error"
