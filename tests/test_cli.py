"""`python -m app.cli`: the rows an installer makes, and what a rerun does.

The end-to-end test runs the command as the installer does, as a process of its
own beside a proxy that is already serving, because that is the case the
console never had: a key minted elsewhere has to reach the server's key cache
without a restart.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import func, select, text

from app import cli
from app.orm import ApiKey, Client, Credential, Model, Provider

REPO = Path(__file__).resolve().parent.parent


async def _schema_of(maker) -> str:
    async with maker() as s:
        return (await s.execute(text("SELECT current_schema()"))).scalar()


def _run_cli(dsn: str, schema: str, *args: str) -> dict:
    env = {**os.environ, "DATABASE_URL": dsn, "DB_SCHEMA": schema, "PYTHONPATH": str(REPO)}
    done = subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


async def test_cli_provider_model_and_key_are_served_by_a_running_proxy(
    db_maker, scratch_database, monkeypatch
):
    import app.main as main

    async def _no_probe():
        return None

    # The probe would call the upstream for real; nothing here is about it.
    monkeypatch.setattr(main, "connectivity_probe_loop", _no_probe)
    monkeypatch.setattr(main.config, "key_cache_refresh_seconds", 0.1)
    schema = await _schema_of(db_maker)

    with TestClient(main.create_app()) as proxy:
        _run_cli(scratch_database, schema, "add-provider", "--name", "anthropic",
                 "--kind", "anthropic", "--api-key", "sk-ant-test")
        _run_cli(scratch_database, schema, "add-model", "--provider", "anthropic",
                 "--model", "claude-test-1", "--context-window", "200000")
        minted = _run_cli(scratch_database, schema, "mint-key", "--client", "mind-a",
                          "--kind", "mind")
        listed = _run_cli(scratch_database, schema, "list-models", "--harness", "dsh")

        deadline = time.monotonic() + 10
        while True:
            answer = proxy.get(
                "/v1/models", params={"harness": "dsh"},
                headers={"Authorization": f"Bearer {minted['key']}"},
            )
            if answer.status_code != 401 or time.monotonic() > deadline:
                break
            time.sleep(0.1)

    assert answer.status_code == 200, answer.text
    served = [{k: v for k, v in m.items() if k != "created"} for m in answer.json()["data"]]
    printed = [{k: v for k, v in m.items() if k != "created"} for m in listed["data"]]
    assert [m["id"] for m in served] == ["claude-test-1"]
    assert printed == served


async def _counts(maker) -> dict:
    async with maker() as s:
        return {
            table.__name__: (await s.execute(select(func.count()).select_from(table))).scalar()
            for table in (Credential, Provider, Model, Client, ApiKey)
        }


async def test_cli_commands_repeated_reuse_rows_and_mint_nothing(db_maker):
    async with db_maker() as s:
        await cli.add_provider(s, name="openai", kind="openai", api_key="sk-first")
        await cli.add_model(s, provider="openai", model="gpt-test", context_window=128000)
        first = await cli.mint_key(s, client="mind-a", kind="mind")
    before = await _counts(db_maker)

    async with db_maker() as s:
        provider = await cli.add_provider(s, name="openai", kind="openai", api_key="sk-other")
        model = await cli.add_model(s, provider="openai", model="gpt-test", context_window=1)
        again = await cli.mint_key(s, client="mind-a", kind="mind")

    assert first["minted"] and first["key"]
    assert (provider["created"], model["created"]) == (False, False)
    assert model["model"]["context_window"] == 128000
    assert (again["minted"], again["key"]) == (False, None)
    assert await _counts(db_maker) == before
    async with db_maker() as s:
        secret = (await s.execute(select(Credential.secret))).scalar_one()
    assert secret == "sk-first"


async def test_cli_mints_for_an_existing_client_that_holds_no_live_key(db_maker):
    async with db_maker() as s:
        await cli.mint_key(s, client="mind-a")
        await s.execute(ApiKey.__table__.update().values(revoked=True))
        await s.commit()
        result = await cli.mint_key(s, client="mind-a")

    assert result["minted"] and result["key"].startswith("hmp-")
    assert result["created"] is False
