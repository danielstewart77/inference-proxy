"""The shape Alembic builds, and the branch the real cutover takes.

The nine migrations already ran clean against Postgres before this work --
what they built was the wrong shape, which no table-existence check can see.
And the branch production actually takes is not the empty-database one: the
schema already held the proxy's tables, one migration behind, with no
bookkeeping row at all.

Every DSN here is plain. The search path is the thing under test: it has to
come from the configuration this code owns, because the alternative -- the
connection string carrying it -- satisfies psycopg and crashes asyncpg, which
has no such parameter.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from app.config import config
from app.entrypoint import REVISION_MARKERS, BootstrapRefused, bootstrap_schema
from tests.conftest import PG_DSN, sync_dsn

@pytest.fixture
def allow_create(monkeypatch):
    monkeypatch.setattr(config, "allow_schema_create", True)


def _conn(schema: str):
    return create_engine(
        sync_dsn(PG_DSN), connect_args={"options": f"-csearch_path={schema}"})


def _columns(schema: str) -> dict[tuple[str, str], tuple[str, str]]:
    engine = create_engine(sync_dsn(PG_DSN))
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT table_name, column_name, data_type, is_identity "
            "FROM information_schema.columns WHERE table_schema = :s"
        ), {"s": schema}).all()
    engine.dispose()
    return {(r[0], r[1]): (r[2], r[3]) for r in rows}


def _version_table_schema() -> list[str]:
    engine = create_engine(sync_dsn(PG_DSN))
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = 'alembic_version'")).all()
    engine.dispose()
    return [r[0] for r in rows]


def test_migrations_build_timestamps_that_carry_a_timezone(pg_schema, allow_create):
    """A timestamp without a timezone is the defect, not a style choice.

    It is the column type that lets a driver reinterpret a naive value as
    local time, and it is what made a fresh install and the migrated
    database two different schemas wearing the same table names.
    """
    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)

    columns = _columns(pg_schema)
    for table, column in (
        ("clients", "created_at"), ("credentials", "expires_at"),
        ("credentials", "refreshed_at"), ("clients", "last_seen_at"),
        ("usage_log", "created_at"),
    ):
        assert columns[(table, column)][0] == "timestamp with time zone", (
            f"{table}.{column}")


def test_migrations_build_identity_keys_not_serial_ones(pg_schema, allow_create):
    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)

    columns = _columns(pg_schema)
    for table in ("clients", "keys", "credentials", "models", "providers",
                  "usage_log"):
        assert columns[(table, "id")][1] == "YES", table
        assert columns[(table, "id")][0] == "bigint", table


def test_the_tables_and_the_bookkeeping_row_land_in_the_named_schema(
        pg_schema, allow_create):
    """Qualification comes from DB_SCHEMA, not from the connection string.

    A plain DSN resolves to the default schema. If the schema binding were an
    accident of the DSN, this would build nine tables and a version row in
    the schema the hive's other services share.
    """
    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)

    columns = _columns(pg_schema)
    assert ("clients", "id") in columns
    schemas = _version_table_schema()
    assert pg_schema in schemas
    assert "public" not in schemas


def test_a_schema_one_migration_behind_is_stamped_there_and_upgraded(
        pg_schema, allow_create):
    """The branch the cutover actually takes.

    The loaded schema held all six tables and no bookkeeping row, and it was
    a migration behind the code. Stamping at head would have recorded the
    last migration as applied and never created its column -- a proxy that
    starts healthy and then refuses every request.
    """
    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)
    engine = _conn(pg_schema)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE providers DROP COLUMN systemone_path"))
        conn.execute(text("DROP TABLE alembic_version"))
    engine.dispose()

    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)

    assert ("providers", "systemone_path") in _columns(pg_schema)


def test_every_revision_is_recognised_by_the_column_it_introduced(pg_schema):
    """Each rung of the ladder against the migration that built it.

    A schema created out of band carries no bookkeeping row, so the only way
    to know where it stands is to look at what it has. A marker naming a
    column its migration did not create -- or one that was already there --
    puts the schema at the wrong revision, and every migration above that
    point is recorded as applied without ever running. The ladder is walked
    here against the real migrations rather than against a hand-written set
    of column names, which would only assert what this test already said.
    """
    from alembic import command

    from app.entrypoint import _alembic_config, _schema_state, _revision_of

    cfg = _alembic_config(sync_dsn(PG_DSN), pg_schema)
    for revision, _, _ in reversed(REVISION_MARKERS):
        command.upgrade(cfg, revision)
        _, columns = _schema_state(sync_dsn(PG_DSN), pg_schema)
        assert _revision_of(columns) == revision, (
            f"a schema upgraded to {revision} is not recognised as being there")


def test_bootstrap_refuses_a_schema_that_does_not_exist(scratch_database):
    """Pointed at a schema that is not there, it must refuse.

    The old code read the default schema instead, found nothing, declared the
    database empty and built a complete empty copy of everything -- then
    reported healthy and gave every mind a 401.
    """
    with pytest.raises(BootstrapRefused):
        bootstrap_schema(dsn=sync_dsn(PG_DSN), schema="no_such_schema_here")


def test_a_fresh_install_creates_the_schema_it_was_given(scratch_database, allow_create):
    """An install against an empty database has to make its own schema.

    `current_schema()` is NULL for a schema that does not exist, so without
    this the install refuses instead of installing -- and nothing else in the
    proxy ever issues CREATE SCHEMA.
    """
    name = f"proxy_install_{uuid.uuid4().hex[:10]}"
    try:
        bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=name)
        assert ("clients", "id") in _columns(name)
    finally:
        engine = create_engine(sync_dsn(PG_DSN))
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
        engine.dispose()


def test_bootstrap_refuses_a_schema_holding_only_some_of_the_tables(
        pg_schema, allow_create):
    """A partial schema has no revision to stamp.

    Guessing one records migrations as applied that then fail on a table that
    is not there, and the bookkeeping row left behind makes every later start
    fail the same way -- forever, with a row asserting a revision that never
    ran.
    """
    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)
    engine = _conn(pg_schema)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE usage_log"))
        conn.execute(text("DROP TABLE alembic_version"))
    engine.dispose()

    with pytest.raises(BootstrapRefused):
        bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)


def test_the_orm_and_the_migrations_agree_on_every_column(pg_schema, allow_create):
    """One schema, not two wearing the same table names.

    This is what makes the next migration safe to autogenerate: if the ORM
    and the migrations disagree, autogenerate reads the difference as drift
    and ships a narrowing ALTER on the busiest table in the hive inside a
    migration that was supposed to add one column.
    """
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from app.orm import Base

    bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)

    engine = _conn(pg_schema)
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        diff = compare_metadata(context, Base.metadata)
    engine.dispose()

    assert diff == [], f"the ORM and the migrations disagree: {diff}"


def test_bootstrap_refuses_to_create_tables_unless_told_this_is_an_install(
        pg_schema, monkeypatch):
    """Creating the tables is a fresh install, not a start.

    Unset, this refusal is the thing that stops a mistyped DB_SCHEMA from
    quietly building a second, empty proxy beside the real one.
    """
    monkeypatch.setattr(config, "allow_schema_create", False)

    with pytest.raises(BootstrapRefused):
        bootstrap_schema(dsn=sync_dsn(PG_DSN), schema=pg_schema)
