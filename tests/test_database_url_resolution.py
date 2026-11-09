"""Where the proxy's data lives is one setting: a Postgres DSN.

`DATABASE_URL` is handed through with the driver the caller asked for. There
is no other engine, so an unset or foreign setting is refused rather than
resolved to somewhere else.
"""

from __future__ import annotations

import pytest

from app.config import database_url, redacted


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


def test_an_unset_database_url_is_refused_rather_than_falling_back_to_a_file(monkeypatch):
    monkeypatch.setattr("app.config.config.database_url_override", "")
    with pytest.raises(ValueError, match="DATABASE_URL is unset"):
        database_url(driver="async")


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
    dsn = "postgresql+asyncpg://hive@127.0.0.1:5433/hive"
    assert redacted(dsn) == dsn

