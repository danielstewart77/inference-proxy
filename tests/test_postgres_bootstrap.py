"""The shape Alembic builds, and the branch the real cutover takes.

The nine migrations already ran clean against Postgres before this work --
what they built was the wrong shape, which no table-existence check can see.
And the branch production actually takes is not the empty-database one: the
schema already held the proxy's tables, one migration behind, with no
bookkeeping row at all.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from app.entrypoint import BootstrapRefused, bootstrap_schema
from tests.conftest import PG_DSN, requires_postgres

pytestmark = requires_postgres


def _sync_dsn(schema: str) -> str:
    for prefix in ("postgresql+psycopg://", "postgresql://", "postgres://"):
        if PG_DSN.startswith(prefix):
            body = PG_DSN[len(prefix):]
            break
    else:
        body = PG_DSN
    sep = "&" if "?" in body else "?"
    return f"postgresql+psycopg://{body}{sep}options=-csearch_path%3D{schema}"


def _columns(schema: str) -> dict[tuple[str, str], tuple[str, str]]:
    engine = create_engine(_sync_dsn(schema))
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT table_name, column_name, data_type, is_identity "
            "FROM information_schema.columns WHERE table_schema = :s"
        ), {"s": schema}).all()
    engine.dispose()
    return {(r[0], r[1]): (r[2], r[3]) for r in rows}


@requires_postgres
def test_migrations_build_timestamps_that_carry_a_timezone(pg_schema):
    """A timestamp without a timezone is the defect, not a style choice.

    It is the column type that lets a driver reinterpret a naive value as
    local time, and it is what made a fresh install and the migrated
    database two different schemas wearing the same table names.
    """
    bootstrap_schema(dsn=_sync_dsn(pg_schema), schema=pg_schema)

    columns = _columns(pg_schema)
    for table, column in (
        ("clients", "created_at"), ("credentials", "expires_at"),
        ("credentials", "refreshed_at"), ("clients", "last_seen_at"),
        ("usage_log", "created_at"),
    ):
        assert columns[(table, column)][0] == "timestamp with time zone", (
            f"{table}.{column}")


@requires_postgres
def test_migrations_build_identity_keys_not_serial_ones(pg_schema):
    bootstrap_schema(dsn=_sync_dsn(pg_schema), schema=pg_schema)

    columns = _columns(pg_schema)
    for table in ("clients", "keys", "credentials", "models", "providers",
                  "usage_log"):
        assert columns[(table, "id")][1] == "YES", table
        assert columns[(table, "id")][0] == "bigint", table


@requires_postgres
def test_a_schema_one_migration_behind_is_stamped_there_and_upgraded(pg_schema):
    """The branch the cutover actually takes.

    The loaded schema held all six tables and no bookkeeping row, and it was
    a migration behind the code. Stamping at head would have recorded the
    last migration as applied and never created its column -- a proxy that
    starts healthy and then refuses every request.
    """
    bootstrap_schema(dsn=_sync_dsn(pg_schema), schema=pg_schema)
    engine = create_engine(_sync_dsn(pg_schema))
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE providers DROP COLUMN systemone_path"))
        conn.execute(text("DROP TABLE alembic_version"))
    engine.dispose()

    bootstrap_schema(dsn=_sync_dsn(pg_schema), schema=pg_schema)

    assert ("providers", "systemone_path") in _columns(pg_schema)


@requires_postgres
def test_bootstrap_refuses_a_schema_that_is_not_the_one_it_was_told(pg_schema):
    """Pointed at the wrong schema, it must refuse rather than build a copy.

    A DSN with no search path resolves to the default schema, where the old
    code found no tables, declared the database empty, and built a complete
    empty copy of everything. The proxy then answered healthy and gave every
    mind a 401.
    """
    with pytest.raises(BootstrapRefused):
        bootstrap_schema(dsn=_sync_dsn(pg_schema), schema="proxy")
