"""Background task that prunes old usage_log rows."""

from __future__ import annotations

import asyncio
import time

from datetime import timedelta

from sqlalchemy import delete

from app.config import config
from app.db import SessionLocal
from app.orm import UsageLog
from app.timeutil import utcnow


async def prune_once(session) -> int:
    """Delete usage rows past the retention window; return how many went.

    The cutoff is computed here rather than by the database.
    `datetime('now', ...)` is a SQLite function, and against Postgres it
    raised once a day into a log nobody reads while `usage_log` grew without
    limit.
    """
    cutoff = utcnow() - timedelta(days=config.usage_prune_days)
    result = await session.execute(
        delete(UsageLog).where(UsageLog.created_at < cutoff))
    await session.commit()
    return int(getattr(result, "rowcount", -1))


async def _prune_once() -> None:
    if config.usage_prune_disabled:
        return
    started = time.time()
    async with SessionLocal() as session:
        rowcount = await prune_once(session)
    duration = time.time() - started
    print(
        f"[retention] pruned {rowcount} usage_log row(s) older than "
        f"{config.usage_prune_days} days in {duration:.2f}s",
        flush=True,
    )


async def usage_prune_loop() -> None:
    """Run forever: prune, sleep, repeat. Started at app startup."""
    if config.usage_prune_disabled:
        print("[retention] USAGE_PRUNE_DISABLE=true; not starting prune loop", flush=True)
        return
    while True:
        try:
            await _prune_once()
        except Exception as e:
            print(f"[retention] prune cycle failed: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(config.usage_prune_interval_seconds)
