import pytest
from fastapi import HTTPException

from app.deployments import resolve_deployment
from app.orm import Credential, Model, Provider


@pytest.mark.asyncio
async def test_provider_switch_controls_every_model_even_with_legacy_overrides(session):
    old = Credential(name='old', kind='static', secret='old-secret')
    api = Credential(name='api', kind='static', secret='api-secret')
    oauth = Credential(name='oauth', kind='anthropic_oauth', secret='oauth-secret')
    session.add_all([old, api, oauth]); await session.flush()
    provider = Provider(name='vendor', base_url='https://api.anthropic.com', messages_path='/v1/messages', credential_id=api.id, auth_scheme='x_api_key', enabled=True)
    session.add(provider); await session.flush()
    session.add_all([Model(deployment_name=name, provider_id=provider.id, credential_id=old.id, auth_scheme='bearer', enabled=True) for name in ('first', 'second')]); await session.commit()
    for name in ('first', 'second'):
        target = await resolve_deployment(session, name, wire='anthropic_messages')
        assert (target.api_key, target.auth_scheme, target.credential_kind) == ('api-secret', 'x_api_key', 'static')
    provider.credential = oauth; provider.auth_scheme = 'bearer'; await session.commit()
    for name in ('first', 'second'):
        target = await resolve_deployment(session, name, wire='anthropic_messages')
        assert (target.api_key, target.auth_scheme, target.credential_kind) == ('oauth-secret', 'bearer', 'anthropic_oauth')
    provider.credential = api; provider.auth_scheme = 'x_api_key'; await session.commit()
    target = await resolve_deployment(session, 'first', wire='anthropic_messages')
    assert (target.api_key, target.auth_scheme, target.credential_kind) == ('api-secret', 'x_api_key', 'static')


@pytest.mark.asyncio
async def test_providerless_model_with_legacy_settings_is_refused(session):
    credential = Credential(name='legacy', kind='static', secret='legacy-secret')
    session.add(credential); await session.flush()
    session.add(Model(deployment_name='legacy', target_uri='https://api.anthropic.com/v1/messages', credential_id=credential.id, auth_scheme='x_api_key', enabled=True)); await session.commit()
    with pytest.raises(HTTPException):
        await resolve_deployment(session, 'legacy', wire='anthropic_messages')


@pytest.mark.asyncio
async def test_missing_provider_credential_does_not_use_legacy_model_credential(session):
    credential = Credential(name='legacy', kind='static', secret='legacy-secret')
    session.add(credential); await session.flush()
    provider = Provider(name='vendor', base_url='https://api.anthropic.com', messages_path='/v1/messages', auth_scheme='x_api_key', enabled=True)
    session.add(provider); await session.flush()
    session.add(Model(deployment_name='model', provider_id=provider.id, credential_id=credential.id, enabled=True)); await session.commit()
    with pytest.raises(HTTPException):
        await resolve_deployment(session, 'model', wire='anthropic_messages')
