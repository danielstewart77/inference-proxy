"""Where the proxy's data lives is one setting, and it resolves two ways.

`DATABASE_URL` wins and is handed through with the driver the caller asked
for. With it unset the proxy is on its own SQLite file, which is what makes
the revert one setting and a restart.
"""

from __future__ import annotations

import pytest

from app.config import REPO_ROOT, database_url, is_postgres, redacted


def test_postgres_url_gets_the_async_driver(monkeypatch):
    monkeypatch.setattr(
        "app.config.config.database_url_override",
        "postgresql://u:p@127.0.0.1:5433/hive",
    )
    assert database_url(driver="async") == (
        "postgresql+asyncpg://u:p@127.0.0.1:5433/hive"
    )


def test_postgres_url_gets_the_sync_driver_for_migrations(monkeypatch):
    monkeypatch.setattr(
        "app.config.config.database_url_override",
        "postgresql://u:p@127.0.0.1:5433/hive",
    )
    assert database_url(driver="sync") == (
        "postgresql+psycopg://u:p@127.0.0.1:5433/hive"
    )


def test_an_unset_override_resolves_to_the_sqlite_file(monkeypatch):
    monkeypatch.setattr("app.config.config.database_url_override", "")
    monkeypatch.setattr("app.config.config.db_path", "data/proxy.db")
    assert database_url(driver="async") == (
        f"sqlite+aiosqlite:///{REPO_ROOT / 'data' / 'proxy.db'}"
    )


def test_the_sqlite_file_gets_a_sync_driver_for_migrations(monkeypatch):
    monkeypatch.setattr("app.config.config.database_url_override", "")
    monkeypatch.setattr("app.config.config.db_path", "data/proxy.db")
    assert database_url(driver="sync") == (
        f"sqlite:///{REPO_ROOT / 'data' / 'proxy.db'}"
    )


def test_an_engine_this_proxy_does_not_speak_is_refused(monkeypatch):
    monkeypatch.setattr(
        "app.config.config.database_url_override", "mysql://u:p@h/d")
    with pytest.raises(ValueError):
        database_url(driver="async")


def test_a_password_is_never_rendered_back(monkeypatch):
    """The startup banner goes to a world-readable log.

    The location is a DSN now, so the thing that used to be a file path
    carries a credential.
    """
    assert redacted("postgresql+asyncpg://hive:s3cr3t-pw@127.0.0.1:5433/hive") == (
        "postgresql+asyncpg://hive:***@127.0.0.1:5433/hive")
    assert "s3cr3t-pw" not in redacted(
        "postgresql+asyncpg://hive:s3cr3t-pw@127.0.0.1:5433/hive")


def test_a_location_with_no_credential_is_left_alone(monkeypatch):
    path = "sqlite:////home/daniel/Storage/Dev/inference-proxy/data/proxy.db"
    assert redacted(path) == path


def test_is_postgres_follows_the_override(monkeypatch):
    monkeypatch.setattr("app.config.config.database_url_override", "")
    assert is_postgres() is False
    monkeypatch.setattr(
        "app.config.config.database_url_override", "postgresql://u:p@h/d"
    )
    assert is_postgres() is True
