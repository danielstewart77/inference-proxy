"""The engine and the session dependency, for whichever engine is configured.

Built on first use rather than at import. Two reasons, both learned the hard
way: an engine created at import time freezes the configuration before
anything can point it somewhere, and on a server engine it means the process
tries to reach the database while the module graph is still loading -- so a
proxy that starts before Postgres is ready dies at import instead of at a
place that can report why.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import config, database_url, is_postgres

_engine: Optional[AsyncEngine] = None
_maker: Optional[async_sessionmaker] = None


def _build_engine() -> AsyncEngine:
    url = database_url(driver="async")

    if not is_postgres():
        engine = create_async_engine(url, pool_pre_ping=True, future=True)

        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            """WAL for concurrent readers during streaming, and real FKs.

            Only ever registered on a SQLite engine. Registered on a Postgres
            one it fires inside pool checkout, so every connection -- and
            every reconnection after a server restart -- fails on a word the
            server does not know.
            """
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        return engine

    # The search path is the whole of the schema binding: the ORM declares no
    # schema on any table, and `credentials` exists three times in this
    # database meaning three different things.
    #
    # The timeouts are the difference between a failure and a mind that
    # appears to be thinking forever. The server's own `statement_timeout` and
    # `idle_in_transaction_session_timeout` are both 0, and with no keepalives
    # a connection that dies while checked out leaves its coroutine blocked in
    # recv with nothing to end it.
    return create_async_engine(
        url,
        future=True,
        pool_pre_ping=True,
        pool_size=config.pool_size,
        max_overflow=config.pool_max_overflow,
        pool_timeout=config.pool_timeout_seconds,
        pool_recycle=1800,
        connect_args={
            "server_settings": {
                "search_path": config.db_schema,
                "statement_timeout": str(
                    config.statement_timeout_seconds * 1000),
                # Deliberately off. A streamed turn legitimately holds its
                # session open for the whole stream, and an Opus turn runs
                # well past a minute -- a timeout here terminates the backend
                # mid-answer and the cause is sixty seconds in the past by
                # the time anything notices. `statement_timeout` bounds the
                # thing that can actually hang, which is one statement.
                "idle_in_transaction_session_timeout": "0",
                "application_name": "inference-proxy",
            },
            "timeout": config.connect_timeout_seconds,
        },
    )


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _sessionmaker() -> async_sessionmaker:
    global _maker
    if _maker is None:
        _maker = async_sessionmaker(
            get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _maker


class _LazySessionLocal:
    """`SessionLocal()` without binding the engine at import.

    Every module that writes outside a request does `from app.db import
    SessionLocal`, so the name has to survive. It resolves the real
    sessionmaker on first call.
    """

    def __call__(self, *args, **kwargs) -> AsyncSession:
        return _sessionmaker()(*args, **kwargs)


SessionLocal = _LazySessionLocal()


async def dispose_engine() -> None:
    """Drop the pool. Called at shutdown, and by tests between engines."""
    global _engine, _maker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _maker = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, rolls back on exception, closes after."""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
