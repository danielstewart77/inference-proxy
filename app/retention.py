"""Background task that prunes old usage_log rows."""

from __future__ import annotations

import asyncio
import time

from sqlalchemy import text

from app.config import config
from app.db import SessionLocal


async def _prune_once() -> None:
    if config.usage_prune_disabled:
        return
    started = time.time()
    async with SessionLocal() as session:
        result = await session.execute(
            text(
                "DELETE FROM usage_log "
                "WHERE created_at < datetime('now', '-' || :days || ' days')"
            ),
            {"days": config.usage_prune_days},
        )
        await session.commit()
        rowcount = getattr(result, "rowcount", -1)
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
