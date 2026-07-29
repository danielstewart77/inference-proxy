"""/admin/clients/{id} update: edits land and the key cache is refreshed."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.admin import clients as admin_clients
from app.db import get_session
from app.orm import Client


@pytest.fixture
def app(session, monkeypatch):
    async def _allow_admin(request):
        return None

    refreshed = {"count": 0}

    async def _fake_refresh(_session):
        refreshed["count"] += 1

    monkeypatch.setattr(admin_clients, "require_html_admin", _allow_admin)
    monkeypatch.setattr(admin_clients, "refresh_key_cache", _fake_refresh)

    application = FastAPI()
    application.include_router(admin_clients.router)
    application.dependency_overrides[get_session] = lambda: session
    application.state.refreshed = refreshed
    return application


async def test_update_flips_privileged_and_refreshes_cache(app, session):
    client_row = Client(name="Skippy", kind="mind", privileged=False)
    session.add(client_row)
    await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        resp = await http.post(
            f"/admin/clients/{client_row.id}",
            data={"name": "Skippy", "kind": "mind", "privileged": "on"},
        )
    assert resp.status_code == 303
    await session.refresh(client_row)
    assert client_row.privileged is True
    # Privileged rides the in-memory key cache — no refresh, no effect.
    assert app.state.refreshed["count"] == 1


async def test_update_unchecking_privileged_revokes_it(app, session):
    client_row = Client(name="Ada", kind="mind", privileged=True)
    session.add(client_row)
    await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        resp = await http.post(
            f"/admin/clients/{client_row.id}",
            data={"name": "Ada", "kind": "mind"},  # checkbox absent = off
        )
    assert resp.status_code == 303
    await session.refresh(client_row)
    assert client_row.privileged is False


async def test_update_rejects_name_clash(app, session):
    session.add_all(
        [Client(name="Bob", kind="mind"), Client(name="Bilby", kind="mind")]
    )
    await session.commit()
    bilby = (
        await session.execute(select(Client).where(Client.name == "Bilby"))
    ).scalar_one()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http:
        resp = await http.post(
            f"/admin/clients/{bilby.id}", data={"name": "Bob", "kind": "mind"}
        )
    assert resp.status_code == 409
