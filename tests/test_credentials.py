"""app.credentials: secret resolution for static and OAuth credential kinds."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app import credentials as creds
from app.credentials import (
    CredentialError,
    codex_account_id,
    read_anthropic_oauth_file,
    read_codex_oauth_tokens,
    resolve_secret,
)
from app.orm import Credential


def _epoch_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)


def _fake_jwt(*, expires_in_seconds: int, account_id: str = "acct-123") -> str:
    import base64

    def seg(obj) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    exp = int(datetime.now(timezone.utc).timestamp()) + expires_in_seconds
    claims = {
        "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    return f"{seg({'alg': 'none'})}.{seg(claims)}.sig"


def _write_codex_file(
    tmp_path, *, expires_in_seconds: int, api_key: str | None = None
) -> str:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": api_key,
                "tokens": {
                    "access_token": _fake_jwt(expires_in_seconds=expires_in_seconds),
                    "refresh_token": "codex-refresh",
                    "account_id": "acct-123",
                },
            }
        )
    )
    return str(path)


def _write_claude_file(tmp_path, *, expires_in_seconds: int) -> str:
    expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
        seconds=expires_in_seconds
    )
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat-from-file",
                    "refreshToken": "sk-ant-ort-refresh",
                    "expiresAt": _epoch_ms(expiry),
                }
            }
        )
    )
    return str(path)


# ---- file readers ----------------------------------------------------------


def test_reads_claude_oauth_file(tmp_path):
    path = _write_claude_file(tmp_path, expires_in_seconds=3600)
    access, refresh, expires_at = read_anthropic_oauth_file(path)
    assert access == "sk-ant-oat-from-file"
    assert refresh == "sk-ant-ort-refresh"
    assert expires_at is not None


def test_missing_claude_file_raises(tmp_path):
    with pytest.raises(CredentialError, match="not found"):
        read_anthropic_oauth_file(str(tmp_path / "nope.json"))


def test_claude_file_without_tokens_raises(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"claudeAiOauth": {}}))
    with pytest.raises(CredentialError, match="no OAuth tokens"):
        read_anthropic_oauth_file(str(path))


def test_reads_codex_oauth_tokens_with_jwt_expiry(tmp_path):
    path = _write_codex_file(tmp_path, expires_in_seconds=3600)
    access, refresh, expires_at = read_codex_oauth_tokens(path)
    assert refresh == "codex-refresh"
    assert expires_at is not None and expires_at > creds._now()


def test_codex_file_without_tokens_raises(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"tokens": {}}))
    with pytest.raises(CredentialError, match="no OAuth tokens"):
        read_codex_oauth_tokens(str(path))


def test_codex_account_id_comes_from_the_jwt():
    assert codex_account_id(_fake_jwt(expires_in_seconds=60, account_id="acct-9")) == "acct-9"
    assert codex_account_id("not-a-jwt") is None


# ---- resolve_secret --------------------------------------------------------


async def test_static_credential_returns_its_secret(session):
    cred = Credential(name="openai", kind="static", secret="sk-live")
    session.add(cred)
    await session.commit()
    assert await resolve_secret(session, cred) == "sk-live"


async def test_static_credential_without_secret_raises(session):
    cred = Credential(name="empty", kind="static")
    session.add(cred)
    await session.commit()
    with pytest.raises(CredentialError, match="no secret set"):
        await resolve_secret(session, cred)


async def test_missing_credential_raises(session):
    with pytest.raises(CredentialError, match="no credential attached"):
        await resolve_secret(session, None)


async def test_unknown_kind_raises(session):
    cred = Credential(name="weird", kind="static")
    session.add(cred)
    await session.commit()
    cred.kind = "smoke-signal"
    with pytest.raises(CredentialError, match="unknown credential kind"):
        await resolve_secret(session, cred)


async def test_fresh_cached_token_is_reused_without_touching_the_file(session, tmp_path):
    cred = Credential(
        name="claude",
        kind="anthropic_oauth",
        source_path=str(tmp_path / "does-not-exist.json"),
        access_token="cached-token",
        expires_at=creds._now() + timedelta(hours=2),
    )
    session.add(cred)
    await session.commit()
    # source_path points at nothing; a cache hit must not read it.
    assert await resolve_secret(session, cred) == "cached-token"


async def test_stale_cache_adopts_the_token_the_cli_already_refreshed(session, tmp_path):
    path = _write_claude_file(tmp_path, expires_in_seconds=7200)
    cred = Credential(
        name="claude",
        kind="anthropic_oauth",
        source_path=path,
        access_token="stale-token",
        expires_at=creds._now() - timedelta(minutes=1),
    )
    session.add(cred)
    await session.commit()

    assert await resolve_secret(session, cred) == "sk-ant-oat-from-file"
    assert cred.access_token == "sk-ant-oat-from-file"
    assert cred.refreshed_at is not None


async def test_expired_file_triggers_a_refresh_and_caches_the_result(
    session, tmp_path, monkeypatch
):
    path = _write_claude_file(tmp_path, expires_in_seconds=10)
    cred = Credential(name="claude", kind="anthropic_oauth", source_path=path)
    session.add(cred)
    await session.commit()

    seen: dict[str, str] = {}

    async def fake_refresh(refresh_token: str):
        seen["refresh_token"] = refresh_token
        return "refreshed-token", creds._now() + timedelta(hours=1)

    monkeypatch.setattr(creds, "refresh_anthropic_token", fake_refresh)

    assert await resolve_secret(session, cred) == "refreshed-token"
    assert seen["refresh_token"] == "sk-ant-ort-refresh"
    assert cred.access_token == "refreshed-token"


async def test_oauth_credential_without_secret_or_source_path_raises(session):
    cred = Credential(name="claude", kind="anthropic_oauth")
    session.add(cred)
    await session.commit()
    with pytest.raises(CredentialError, match="no secret and no source_path"):
        await resolve_secret(session, cred)


async def test_anthropic_oauth_literal_token_is_returned_as_is(session):
    cred = Credential(name="ada", kind="anthropic_oauth", secret="sk-ant-oat01-yearlong")
    session.add(cred)
    await session.commit()
    assert await resolve_secret(session, cred) == "sk-ant-oat01-yearlong"


async def test_anthropic_oauth_literal_token_beats_source_path(session, tmp_path):
    # The stored token was placed deliberately; it must win even when a
    # credential file (with a different, fresher token) is also configured.
    path = _write_claude_file(tmp_path, expires_in_seconds=7200)
    cred = Credential(
        name="ada",
        kind="anthropic_oauth",
        secret="sk-ant-oat01-yearlong",
        source_path=path,
    )
    session.add(cred)
    await session.commit()
    assert await resolve_secret(session, cred) == "sk-ant-oat01-yearlong"


async def test_openai_oauth_credential_prefers_a_plain_api_key(session, tmp_path):
    path = _write_codex_file(tmp_path, expires_in_seconds=3600, api_key="sk-proj-xyz")
    cred = Credential(name="codex", kind="openai_oauth", source_path=str(path))
    session.add(cred)
    await session.commit()
    assert await resolve_secret(session, cred) == "sk-proj-xyz"


async def test_openai_oauth_adopts_a_fresh_file_token(session, tmp_path):
    path = _write_codex_file(tmp_path, expires_in_seconds=7200)
    cred = Credential(name="codex", kind="openai_oauth", source_path=path)
    session.add(cred)
    await session.commit()

    secret = await resolve_secret(session, cred)
    assert secret == json.loads(open(path).read())["tokens"]["access_token"]
    assert cred.expires_at is not None


async def test_openai_oauth_refreshes_stale_token_and_writes_back(
    session, tmp_path, monkeypatch
):
    path = _write_codex_file(tmp_path, expires_in_seconds=10)
    cred = Credential(name="codex", kind="openai_oauth", source_path=path)
    session.add(cred)
    await session.commit()

    fresh = _fake_jwt(expires_in_seconds=3 * 24 * 3600)

    async def fake_refresh(refresh_token: str):
        assert refresh_token == "codex-refresh"
        return fresh, "rotated-refresh", creds._now() + timedelta(days=3)

    monkeypatch.setattr(creds, "refresh_openai_token", fake_refresh)

    assert await resolve_secret(session, cred) == fresh
    # Rotated tokens must land back in the file so the CLI keeps working.
    blob = json.loads(open(path).read())
    assert blob["tokens"]["access_token"] == fresh
    assert blob["tokens"]["refresh_token"] == "rotated-refresh"
    assert blob["last_refresh"].endswith("Z")


async def test_openai_oauth_fresh_cache_skips_the_file(session, tmp_path):
    cred = Credential(
        name="codex",
        kind="openai_oauth",
        source_path=str(tmp_path / "auth.json"),
        access_token="cached-codex-token",
        expires_at=creds._now() + timedelta(hours=2),
    )
    session.add(cred)
    await session.commit()
    # source_path points at nothing; a cache hit must not read it.
    assert await resolve_secret(session, cred) == "cached-codex-token"
