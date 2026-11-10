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

import httpx
import pytest
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


async def _stored(maker, provider_name: str) -> tuple[Provider, Credential]:
    async with maker() as s:
        provider = (
            await s.execute(select(Provider).where(Provider.name == provider_name))
        ).scalar_one()
        credential = (
            await s.execute(select(Credential).where(Credential.id == provider.credential_id))
        ).scalar_one()
    return provider, credential


async def test_cli_update_rewrites_an_existing_providers_key_and_address(db_maker):
    async with db_maker() as s:
        await cli.add_provider(s, name="anthropic", kind="anthropic",
                               base_url="https://wrong.example", api_key="sk-wrong")
    before = await _counts(db_maker)

    async with db_maker() as s:
        result = await cli.add_provider(s, name="anthropic", kind="anthropic",
                                        api_key="sk-right", update=True)

    provider, credential = await _stored(db_maker, "anthropic")
    assert (result["created"], result["updated"]) == (False, True)
    assert provider.base_url == "https://api.anthropic.com"
    assert credential.secret == "sk-right"
    assert await _counts(db_maker) == before


async def test_cli_update_given_no_key_keeps_the_stored_one(db_maker):
    async with db_maker() as s:
        await cli.add_provider(s, name="anthropic", kind="anthropic", api_key="sk-kept")
        await cli.add_provider(s, name="anthropic", kind="anthropic",
                               base_url="https://gateway.example", update=True)

    provider, credential = await _stored(db_maker, "anthropic")
    assert provider.base_url == "https://gateway.example"
    assert credential.secret == "sk-kept"


async def test_cli_update_moves_a_model_and_rewrites_its_window_keeping_what_was_not_given(
    db_maker,
):
    async with db_maker() as s:
        await cli.add_provider(s, name="first", kind="openai", api_key="sk-a")
        await cli.add_provider(s, name="second", kind="openai", api_key="sk-b")
        await cli.add_model(s, provider="first", model="gpt-test", context_window=1000,
                            rotation_threshold_percent=45, label="Kept label")
        result = await cli.add_model(s, provider="second", model="gpt-test",
                                     context_window=128000, update=True)

    assert (result["created"], result["updated"]) == (False, True)
    async with db_maker() as s:
        row = (await s.execute(select(Model))).scalar_one()
        host = (await s.execute(select(Provider.name).where(Provider.id == row.provider_id))
                ).scalar_one()
    assert (host, row.context_window) == ("second", 128000)
    assert (row.rotation_threshold_percent, row.label) == (45, "Kept label")


@pytest.mark.parametrize("key_args, env_extra, stdin", [
    (("--api-key-env", "PROVIDER_KEY"), {"PROVIDER_KEY": "sk-from-env"}, None),
    (("--api-key", "-"), {}, "sk-from-stdin\n"),
], ids=["env", "stdin"])
async def test_cli_takes_the_provider_key_from_outside_argv(
    db_maker, scratch_database, key_args, env_extra, stdin
):
    schema = await _schema_of(db_maker)
    env = {**os.environ, **env_extra, "DATABASE_URL": scratch_database,
           "DB_SCHEMA": schema, "PYTHONPATH": str(REPO)}
    done = subprocess.run(
        [sys.executable, "-m", "app.cli", "add-provider", "--name", "openai",
         "--kind", "openai", *key_args],
        cwd=REPO, env=env, input=stdin or "", capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stdout + done.stderr

    _, credential = await _stored(db_maker, "openai")
    expected = (env_extra.get("PROVIDER_KEY") or stdin).strip()
    assert credential.secret == expected


def _upstream(monkeypatch, status: int, body: dict) -> list[httpx.Request]:
    """The proxy's upstream client, answering every request with `body`."""
    import app.azure

    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(
        app.azure, "_shared_client", httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    return sent


async def test_check_model_reports_the_upstreams_refusal_and_exits_2(
    db_maker, monkeypatch, capsys
):
    async with db_maker() as s:
        await cli.add_provider(s, name="anthropic", kind="anthropic", api_key="sk-bad")
        await cli.add_model(s, provider="anthropic", model="claude-test-1",
                            context_window=200000)
    sent = _upstream(monkeypatch, 401, {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key"},
    })

    code = await cli.run(["check-model", "--model", "claude-test-1"])

    printed = json.loads(capsys.readouterr().out)
    assert code == 2
    assert printed["status"] == 401
    assert "invalid x-api-key" in printed["error"]
    assert [str(r.url) for r in sent] == ["https://api.anthropic.com/v1/messages"]
    assert sent[0].headers["x-api-key"] == "sk-bad"


async def test_check_model_reports_a_completed_turn_and_exits_0(db_maker, monkeypatch, capsys):
    async with db_maker() as s:
        await cli.add_provider(s, name="openai", kind="openai", api_key="sk-good")
        await cli.add_model(s, provider="openai", model="gpt-test", context_window=128000)
    sent = _upstream(monkeypatch, 200, {
        "id": "c1", "object": "chat.completion", "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
    })

    code = await cli.run(["check-model", "--model", "gpt-test"])

    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (printed["ok"], printed["status"], printed["reply"]) == (True, 200, "ok")
    assert [str(r.url) for r in sent] == ["https://api.openai.com/v1/chat/completions"]
