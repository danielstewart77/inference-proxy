"""Shared fixtures: a Postgres database of this suite's own, made and dropped per run.

The proxy runs only on Postgres, so the suite does too. Never against the
live database: the proxy's real rows sit in the `proxy` schema of the hive's
one database, and a test that wrote keys, credentials or usage rows there
would be writing into production. So:

- `.env` is never read. `app.config` calls `load_dotenv()` at import, and on
  this host `.env` carries the production `DATABASE_URL`; it is disarmed
  below, before anything under `app` is imported.
- `PROXY_TEST_ADMIN_DSN`, from the process environment only, names a server
  this suite may create and drop a database on. There is deliberately no
  default: a default would be a credential in a tracked file.
- Every run creates `proxy_test_<random>` there and points `DATABASE_URL` at
  it. A DSN naming any other database is refused before a connection opens.
- Each test that wants tables gets a fresh schema inside that database, and
  the app's own engine is pointed at the same schema for the test.

The database is *claimed* for the length of the run with an advisory lock
held on the admin connection, and a run sweeps only the `proxy_test_`
databases nobody holds a claim on -- the leftovers of killed runs. A live run's
database often has no backends at all, so counting connections cannot tell a
leftover from one in use; the claim can. (The same scheme as
`hive_data.testing`, which owns the other suites' prefixes.)
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import dotenv

# Before any `app` import: nothing in this run may take its configuration from
# `.env`, which on this host is the production configuration.
dotenv.load_dotenv = lambda *args, **kwargs: False

#: Every database this suite creates starts with this, and nothing else does.
PREFIX = "proxy_test_"

#: A DSN with rights to create and drop a database.
ADMIN_DSN = os.environ.get("PROXY_TEST_ADMIN_DSN", "").strip()

NO_SERVER = (
    "PROXY_TEST_ADMIN_DSN is unset. It names a Postgres this suite may create "
    "and drop a database on, e.g. the hive's server on 127.0.0.1:5433. The "
    "proxy runs only on Postgres, so every database test needs it; there is "
    "deliberately no default, because a default would be a credential in a "
    "tracked file."
)


def with_database(dsn: str, name: str) -> str:
    parts = urlsplit(dsn)
    return urlunsplit(parts._replace(path=f"/{name}"))


def refuse_unless_scratch(dsn: str) -> None:
    """Refuse any DSN that does not name a database this suite created."""
    database = urlsplit(dsn).path.lstrip("/")
    if not database.startswith(PREFIX):
        raise RuntimeError(
            f"refusing to run tests against database {database!r}: only a "
            f"scratch database named {PREFIX}* is ever a test target")


SCRATCH_NAME = f"{PREFIX}{uuid.uuid4().hex[:12]}"
#: This run's own database. Empty when no server was named, so the app's
#: engine refuses to build rather than reaching anything at all.
PG_DSN = with_database(ADMIN_DSN, SCRATCH_NAME) if ADMIN_DSN else ""
if PG_DSN:
    refuse_unless_scratch(PG_DSN)

# The app reads these at import. A schema no test creates: an app engine
# that was not pointed at a test's own schema finds no tables, and says so.
os.environ["DATABASE_URL"] = PG_DSN
os.environ["DB_SCHEMA"] = "proxy_test_unbound"

import psycopg  # noqa: E402
import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

from app.orm import Base  # noqa: E402


def _claim_key(connection, name: str) -> int:
    return connection.execute("SELECT hashtextextended(%s, 0)", (name,)).fetchone()[0]


def _drop(connection, name: str) -> None:
    connection.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
    connection.execute(f'DROP DATABASE IF EXISTS "{name}"')


def _sweep_stale(admin) -> None:
    """Drop every leftover `proxy_test_` database no live run has claimed."""
    rows = admin.execute(
        "SELECT datname FROM pg_database WHERE datname LIKE %s",
        (PREFIX.replace("_", r"\_") + "%",)).fetchall()
    for (name,) in rows:
        key = _claim_key(admin, name)
        if not admin.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]:
            continue
        try:
            _drop(admin, name)
        except psycopg.Error:
            pass
        finally:
            admin.execute("SELECT pg_advisory_unlock(%s)", (key,))


@pytest.fixture(scope="session")
def scratch_database() -> str:
    """This run's database, created claimed and dropped at the end of the run."""
    if not ADMIN_DSN:
        pytest.fail(NO_SERVER, pytrace=False)
    keeper = psycopg.connect(ADMIN_DSN, autocommit=True)
    try:
        _sweep_stale(keeper)
        keeper.execute("SELECT pg_advisory_lock(%s)", (_claim_key(keeper, SCRATCH_NAME),))
        keeper.execute(f'CREATE DATABASE "{SCRATCH_NAME}"')
        yield PG_DSN
        _drop(keeper, SCRATCH_NAME)
    finally:
        keeper.close()


@pytest_asyncio.fixture
async def db_maker(scratch_database, monkeypatch) -> async_sessionmaker:
    """Sessions on a fresh schema holding the ORM shape, and the app on it too.

    `NullPool` because the app under `TestClient` runs on a loop of its own,
    and a pooled asyncpg connection belongs to the loop that opened it.
    The schema is left for the database drop at the end of the run.
    """
    import app.db

    refuse_unless_scratch(scratch_database)
    schema = f"t_{uuid.uuid4().hex[:12]}"
    engine = create_async_engine(
        _async_dsn(scratch_database),
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    monkeypatch.setattr(app.db, "_engine", engine)
    monkeypatch.setattr(app.db, "_maker", maker)
    yield maker
    await engine.dispose()


@pytest_asyncio.fixture
async def session(db_maker) -> AsyncSession:
    """A session against a fresh schema, torn down per test."""
    async with db_maker() as s:
        yield s


@pytest.fixture
def pg_schema(scratch_database) -> str:
    """An empty scratch schema, for tests that build the shape themselves."""
    name = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{name}"')
    return name


#: The driver prefixes a DSN may arrive carrying.
_PREFIXES = ("postgresql+asyncpg://", "postgresql+psycopg://",
             "postgresql://", "postgres://")


def _with_driver(dsn: str, driver: str) -> str:
    for prefix in _PREFIXES:
        if dsn.startswith(prefix):
            return f"{driver}://" + dsn[len(prefix):]
    return dsn


def _async_dsn(dsn: str) -> str:
    """The DSN with an asyncpg driver, whatever shape it arrived in."""
    return _with_driver(dsn, "postgresql+asyncpg")


def sync_dsn(dsn: str) -> str:
    """The DSN with a sync driver, for Alembic and the startup inspection."""
    return _with_driver(dsn, "postgresql+psycopg")
