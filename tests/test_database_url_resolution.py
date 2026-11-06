"""Where the proxy's data lives is one setting, and it resolves two ways.

`DATABASE_URL` wins and is handed through with the driver the caller asked
for. With it unset the proxy is on its own SQLite file, which is what makes
the revert one setting and a restart.
"""

from __future__ import annotations

from app.config import REPO_ROOT, database_url, is_postgres


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


def test_is_postgres_follows_the_override(monkeypatch):
    monkeypatch.setattr("app.config.config.database_url_override", "")
    assert is_postgres() is False
    monkeypatch.setattr(
        "app.config.config.database_url_override", "postgresql://u:p@h/d"
    )
    assert is_postgres() is True


def test_a_dsn_is_never_treated_as_a_filesystem_path(monkeypatch, tmp_path):
    """The old code ran mkdir on whatever it was handed. A DSN carries a
    password, and that password became a world-readable directory name."""
    monkeypatch.setattr(
        "app.config.config.database_url_override",
        "postgresql://u:secret-pw@127.0.0.1:5433/hive",
    )
    before = set(p.name for p in REPO_ROOT.iterdir())
    database_url(driver="async")
    assert set(p.name for p in REPO_ROOT.iterdir()) == before
