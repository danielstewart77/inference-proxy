"""Shared fixtures.

Two engines, because the proxy supports two. SQLite in memory is the broad
suite and the revert path. Postgres is a scratch schema inside the hive's own
database, created and dropped per test, because the defects this project's
cutover turned up are all invisible on SQLite: a timezone a driver reinvents,
an identity counter, a date function that only one engine has.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.orm import Base

load_dotenv()

# The suite's own engine is SQLite in memory, always. Once this host sets
# DATABASE_URL -- which is the production configuration -- every test that
# builds the app's real engine would otherwise run against the live `proxy`
# schema and write into the hive's one database. The Postgres tests reach
# Postgres through their own fixture, never through the app's configuration.
os.environ["DATABASE_URL"] = ""

#: A DSN with rights to create and drop a schema. No default: a default here
#: would be a credential in a tracked file.
PG_DSN = os.getenv("PROXY_TEST_DSN", "")

requires_postgres = pytest.mark.skipif(
    not PG_DSN, reason="PROXY_TEST_DSN is unset; Postgres tests need a database"
)


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    """A session against a fresh in-memory SQLite database, torn down per test."""
    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s
    await engine.dispose()


@pytest_asyncio.fixture
async def pg_schema() -> str:
    """A scratch schema name, created and dropped around the test.

    A schema rather than a database: the published port reaches one database
    and creating another needs rights this role is not guaranteed to have.
    The name is random so two runs cannot collide, and the drop is CASCADE so
    a test that left a table behind still cleans up.
    """
    name = f"proxy_test_{uuid.uuid4().hex[:12]}"
    engine = create_async_engine(_async_dsn(PG_DSN), future=True)
    async with engine.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{name}"'))
    await engine.dispose()
    try:
        yield name
    finally:
        engine = create_async_engine(_async_dsn(PG_DSN), future=True)
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
        await engine.dispose()


@pytest_asyncio.fixture
async def pg_session(pg_schema: str) -> AsyncSession:
    """A session whose search path is a scratch schema holding the ORM shape."""
    engine = create_async_engine(
        _async_dsn(PG_DSN),
        future=True,
        connect_args={"server_settings": {"search_path": pg_schema}},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s
    await engine.dispose()


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
