"""`/credentials`: the hive console gives each mind a proxy key of its own.

Story 2.1.9, DECIDED 2026-11-15 by Daniel: one proxy key per mind, issued
automatically when the mind is created, so every call it makes is metered to
it. Driven over HTTP against a scratch schema, with the real key cache: the
point is that the key the console hands a mind is the one the proxy accepts.
"""

from __future__ import annotations

import secrets

import pytest_asyncio
from fastapi.testclient import TestClient

from app.auth import _hash_key, generate_proxy_key, refresh_key_cache, validate_api_key
from app.config import config
from app.db import get_session
from app.main import create_app
from app.orm import ApiKey, Client


async def _client_with_key(session, name, *, kind="app", privileged=False) -> str:
    client = Client(name=name, kind=kind, privileged=privileged)
    session.add(client)
    await session.flush()
    raw = generate_proxy_key()
    session.add(ApiKey(client_id=client.id, label="default", key_hash=_hash_key(raw)))
    await session.commit()
    return raw


@pytest_asyncio.fixture
async def proxy(db_maker, monkeypatch):
    async with db_maker() as setup:
        console = await _client_with_key(setup, "hive-console")
        stash_old = await _client_with_key(setup, "Stash", kind="mind")
        skippy = await _client_with_key(setup, "Skippy", kind="mind", privileged=True)
        await refresh_key_cache(setup)
    monkeypatch.setattr(config, "credential_admin_clients", frozenset({"hive-console"}))

    async def _session():
        async with db_maker() as s:
            yield s

    app = create_app()
    app.dependency_overrides[get_session] = _session
    with TestClient(app) as c:
        c.keys = {"console": console, "stash_old": stash_old, "skippy": skippy}
        yield c


def _as(key):
    return {"Authorization": f"Bearer {key}"}


def test_a_new_minds_key_is_accepted_at_once_and_metered_to_its_own_client(proxy):
    value = secrets.token_urlsafe(32)

    response = proxy.put(
        "/credentials/aiden-mind", headers=_as(proxy.keys["console"]),
        json={"username": "aiden", "token": value, "role": "mind"})

    assert response.status_code == 200, response.text
    assert response.json()["created"] is True
    assert validate_api_key(f"Bearer {value}", None)
    listed = {r["holder"]: r for r in proxy.get("/credentials", headers=_as(proxy.keys["console"])).json()}
    assert listed["aiden-mind"]["role"] == "mind" and listed["aiden-mind"]["live"] is True
    # Its own client, not the console's or anyone else's.
    assert listed["aiden-mind"]["client"] == "aiden-mind"


def test_minting_for_an_existing_mind_replaces_its_key_rather_than_adding_one(proxy):
    """Matched without case: the console's `stash` is the proxy's `Stash`."""
    value = secrets.token_urlsafe(32)

    response = proxy.put(
        "/credentials/stash", headers=_as(proxy.keys["console"]),
        json={"username": "daniel", "token": value, "role": "mind"})

    assert response.status_code == 200, response.text
    assert response.json() == {"holder": "stash", "client": "Stash", "created": False, "replaced": 1}
    assert validate_api_key(f"Bearer {value}", None)
    assert not validate_api_key(f"Bearer {proxy.keys['stash_old']}", None)


def test_only_a_named_credential_admin_may_mint(proxy):
    """A mind's own key - even a privileged one - must not issue keys."""
    value = secrets.token_urlsafe(32)

    by_mind = proxy.put("/credentials/aiden-mind", headers=_as(proxy.keys["skippy"]),
                        json={"token": value, "role": "mind"})
    by_nobody = proxy.put("/credentials/aiden-mind", json={"token": value, "role": "mind"})

    assert by_mind.status_code == 403
    assert by_nobody.status_code == 401
    assert not validate_api_key(f"Bearer {value}", None)


def test_revoking_a_holder_stops_its_key_and_no_other(proxy):
    response = proxy.delete("/credentials/stash", headers=_as(proxy.keys["console"]))

    assert response.json()["revoked"] is True
    assert not validate_api_key(f"Bearer {proxy.keys['stash_old']}", None)
    assert validate_api_key(f"Bearer {proxy.keys['skippy']}", None)
    listed = {r["holder"]: r for r in proxy.get("/credentials", headers=_as(proxy.keys["console"])).json()}
    assert listed["stash"]["live"] is False and listed["skippy"]["live"] is True


def test_a_rotation_that_failed_can_put_the_previous_key_back(proxy):
    """The console re-sends the value it replaced when it could not write the
    new one; refusing it would leave the mind with no working key."""
    value = secrets.token_urlsafe(32)
    proxy.put("/credentials/stash", headers=_as(proxy.keys["console"]),
              json={"token": value, "role": "mind"})

    back = proxy.put("/credentials/stash", headers=_as(proxy.keys["console"]),
                     json={"token": proxy.keys["stash_old"], "role": "mind"})

    assert back.status_code == 200, back.text
    assert validate_api_key(f"Bearer {proxy.keys['stash_old']}", None)
    assert not validate_api_key(f"Bearer {value}", None)


def test_a_new_mind_cannot_take_over_the_admin_or_a_privileged_client(proxy):
    """A mind created under the name of the minting client or of a privileged
    one would otherwise revoke its key and inherit its client."""
    for holder in ("hive-console", "skippy"):
        response = proxy.put(f"/credentials/{holder}", headers=_as(proxy.keys["console"]),
                             json={"token": secrets.token_urlsafe(32), "role": "mind"})
        assert response.status_code == 409, holder
    assert proxy.delete("/credentials/hive-console", headers=_as(proxy.keys["console"])).status_code == 409
    assert validate_api_key(f"Bearer {proxy.keys['console']}", None)
    assert validate_api_key(f"Bearer {proxy.keys['skippy']}", None)


def test_an_unknown_route_never_logs_the_key_it_was_sent(proxy, capsys):
    """A console minting into a proxy that predates the route used to write
    its key, and the new mind's, into the journal."""
    proxy.put("/no-such-route", headers=_as(proxy.keys["console"]),
              json={"token": "the-new-minds-key-value-0123456789"})

    logged = capsys.readouterr().out
    assert proxy.keys["console"] not in logged
    assert "the-new-minds-key-value" not in logged


def test_usage_is_answered_per_holder_with_a_quiet_one_at_zero(proxy, db_maker):
    """Story 2.1.17: the console totals usage per person from this, so each
    mind's own calls have to come back under that mind."""
    import asyncio

    from sqlalchemy import select as _select

    from app.usage import store_usage_row

    async def seed():
        async with db_maker() as s:
            stash = (await s.execute(_select(Client).where(Client.name == "Stash"))).scalar_one()
            for tokens in (100, 250):
                await store_usage_row(
                    s, client_id=stash.id, key_id=None, model_name="m", endpoint="/v1/messages",
                    status_code=200, error_type=None, duration_ms=5, client_ip_str="127.0.0.1",
                    input_tokens=tokens, output_tokens=0, total_tokens=tokens,
                    cache_creation_input_tokens=0, cache_read_input_tokens=0)
            await s.commit()

    asyncio.run(seed())

    rows = {r["holder"]: r for r in proxy.get(
        "/credentials/usage", headers=_as(proxy.keys["console"])).json()}

    assert (rows["stash"]["requests"], rows["stash"]["total_tokens"]) == (2, 350)
    assert (rows["skippy"]["requests"], rows["skippy"]["total_tokens"]) == (0, 0)
    assert rows["stash"]["role"] == "mind"


def test_a_minds_own_key_cannot_read_everyones_usage(proxy):
    assert proxy.get("/credentials/usage", headers=_as(proxy.keys["skippy"])).status_code == 403
