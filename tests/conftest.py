"""Shared fixtures: an in-memory SQLite database built from the ORM metadata."""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.orm import Base


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    """A session against a fresh in-memory database, torn down per test."""
    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s
    await engine.dispose()
