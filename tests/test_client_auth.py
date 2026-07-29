"""app.auth: proxy keys are issued to clients, not users."""

from __future__ import annotations

from app.auth import (
    _hash_key,
    generate_proxy_key,
    refresh_key_cache,
    resolve_principal,
    resolve_requester_role,
    validate_api_key,
    verify_admin_credentials,
)
from app.config import config
from app.orm import ApiKey, Client


async def _issue(session, name: str, *, privileged=False, disabled=False, revoked=False) -> str:
    client = Client(name=name, kind="agent", privileged=privileged, disabled=disabled)
    session.add(client)
    await session.flush()
    raw = generate_proxy_key()
    session.add(
        ApiKey(client_id=client.id, label="default", key_hash=_hash_key(raw), revoked=revoked)
    )
    await session.commit()
    return raw


async def test_issued_key_validates_and_resolves_to_its_client(session):
    raw = await _issue(session, "frame-main")
    await refresh_key_cache(session)

    assert validate_api_key(f"Bearer {raw}", None)
    assert validate_api_key(None, raw)
    principal = resolve_principal(None, raw)
    assert principal is not None and principal[0] > 0


async def test_unknown_key_is_rejected(session):
    await _issue(session, "frame-main")
    await refresh_key_cache(session)

    assert not validate_api_key("Bearer sk-ant-someone-elses-token", None)
    assert resolve_principal(None, "hmp-not-a-real-key") is None
    assert resolve_requester_role(None, "hmp-not-a-real-key") is None


async def test_privileged_client_is_admin_everyone_else_is_client(session):
    plain = await _issue(session, "frame-main")
    admin = await _issue(session, "skippy", privileged=True)
    await refresh_key_cache(session)

    assert resolve_requester_role(None, plain) == "client"
    assert resolve_requester_role(None, admin) == "admin"


async def test_disabled_client_keys_drop_out_of_the_cache(session):
    raw = await _issue(session, "retired-agent", disabled=True)
    await refresh_key_cache(session)
    assert not validate_api_key(None, raw)


async def test_revoked_keys_drop_out_of_the_cache(session):
    raw = await _issue(session, "rotating-agent", revoked=True)
    await refresh_key_cache(session)
    assert not validate_api_key(None, raw)


async def test_a_client_may_hold_several_live_keys(session):
    client = Client(name="multi", kind="app")
    session.add(client)
    await session.flush()
    first, second = generate_proxy_key(), generate_proxy_key()
    session.add(ApiKey(client_id=client.id, label="old", key_hash=_hash_key(first)))
    session.add(ApiKey(client_id=client.id, label="new", key_hash=_hash_key(second)))
    await session.commit()
    await refresh_key_cache(session)

    assert validate_api_key(None, first)
    assert validate_api_key(None, second)
    assert resolve_principal(None, first)[0] == resolve_principal(None, second)[0]


async def test_malformed_authorization_header_is_rejected(session):
    await _issue(session, "frame-main")
    await refresh_key_cache(session)
    assert not validate_api_key("Basic dXNlcjpwYXNz", None)
    assert not validate_api_key("", None)


def test_admin_login_requires_configured_credentials(monkeypatch):
    monkeypatch.setattr(config, "admin_username", "")
    monkeypatch.setattr(config, "admin_password", "")
    assert not verify_admin_credentials("", "")
    assert not verify_admin_credentials("admin", "hunter2")


def test_admin_login_matches_configured_credentials(monkeypatch):
    monkeypatch.setattr(config, "admin_username", "daniel")
    monkeypatch.setattr(config, "admin_password", "correct-horse")
    assert verify_admin_credentials("daniel", "correct-horse")
    assert not verify_admin_credentials("daniel", "wrong")
    assert not verify_admin_credentials("someone", "correct-horse")
