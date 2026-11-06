"""Public /status page: aggregate health, usage trends, upstream connectivity.

Anonymous endpoints — no auth. These expose only aggregate counts (request
volume, error rate, latency, upstream reachability counts). They deliberately
never surface usernames, client IPs, costs, or model/host names.

The connectivity snapshot is refreshed by a background loop (modeled on
`app.retention.usage_prune_loop`) and read from an in-memory cache, so an
anonymous page load never triggers a live upstream probe.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import Float, case, func, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionLocal, get_session
from app.orm import Model, UsageLog
from app.timeutil import utcnow
from app.templating import templates

router = APIRouter()

PROBE_INTERVAL_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 5.0

# Debounce: a "major outage" is only declared after this many CONSECUTIVE
# failing probe cycles (~N * PROBE_INTERVAL_SECONDS). A single transient blip —
# e.g. a momentary network hiccup upstream that trips both probes at
# once — surfaces as "degraded" for one cycle instead of painting the public
# page red. Until then the failure shows as degraded.
OUTAGE_FAIL_THRESHOLD = 3
# Warm-up grace: never escalate to outage until the prober has completed a few
# cycles, so a cold first probe right after a container restart (DNS/connections
# not yet warm) can't flip the page.
WARMUP_PROBES = 2

# ---- Cached health snapshot -------------------------------------------------
# Rebuilt every PROBE_INTERVAL_SECONDS by connectivity_probe_loop() and read by
# the (anonymous) /status endpoints. Replaced by atomic reference swap — safe
# under the GIL, no lock — same approach as auth._DB_KEY_HASHES.
_HEALTH: dict = {
    "checked_at": None,      # epoch seconds of last probe, None before first run
    "database": "unknown",   # "up" | "down" | "unknown"
    "upstreams_total": 0,
    "upstreams_up": 0,
    "consec_fail": 0,        # consecutive failing probe cycles (debounce counter)
    "probe_count": 0,        # completed probe cycles since startup (warm-up gate)
}

# Persist the debounce/warm-up counters across cycles (the snapshot dict itself
# is replaced wholesale each cycle, so these live alongside it).
_consec_fail = 0
_probe_count = 0


def _snapshot() -> dict:
    return dict(_HEALTH)


# ---- Connectivity probe -----------------------------------------------------


async def _probe_database() -> str:
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return "up"
    except Exception:
        return "down"


async def _distinct_origins() -> list[str]:
    """Distinct scheme://host[:port] origins from enabled model target URIs.

    Origins are used only to count reachability; they are never returned to the
    public endpoints.
    """
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(Model.target_uri)
                .where(Model.enabled.is_(True))
                .where(Model.target_uri.is_not(None))
                .distinct()
            )
        ).all()
    origins: set[str] = set()
    for (uri,) in rows:
        parts = urlsplit(uri)
        if parts.scheme and parts.netloc:
            origins.add(f"{parts.scheme}://{parts.netloc}")
    return sorted(origins)


async def _probe_origin(client: httpx.AsyncClient, origin: str) -> bool:
    # Any HTTP response (even 401/404) means the host is reachable; only a
    # connect/timeout/TLS failure counts as down.
    try:
        await client.get(origin)
        return True
    except Exception:
        return False


async def _probe_once() -> None:
    global _HEALTH, _consec_fail, _probe_count
    db_status = await _probe_database()
    origins = await _distinct_origins()
    up = 0
    if origins:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            results = await asyncio.gather(*[_probe_origin(client, o) for o in origins])
        up = sum(1 for r in results if r)

    # A cycle is "hard down" when the DB is unreachable or every configured
    # upstream failed this cycle. Track consecutive hard-down cycles so a single
    # blip doesn't escalate to a full outage (see _overall_status).
    hard_down = (db_status == "down") or (len(origins) > 0 and up == 0)
    _consec_fail = _consec_fail + 1 if hard_down else 0
    _probe_count += 1

    _HEALTH = {
        "checked_at": int(time.time()),
        "database": db_status,
        "upstreams_total": len(origins),
        "upstreams_up": up,
        "consec_fail": _consec_fail,
        "probe_count": _probe_count,
    }


async def connectivity_probe_loop() -> None:
    """Run forever: probe, sleep, repeat. Started at app startup."""
    while True:
        try:
            await _probe_once()
        except Exception as e:
            print(f"[status] probe cycle failed: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(PROBE_INTERVAL_SECONDS)


# ---- Aggregate usage queries (no per-client / per-model / IP / cost detail) ----


async def _summary(session: AsyncSession, hours: int = 24) -> dict:
    """Request volume, error count and mean latency over the last `hours`."""
    errored = case(
        (UsageLog.error_type.isnot(None), 1),
        (UsageLog.status_code >= 400, 1),
        else_=0,
    )
    row = (
        await session.execute(
            select(
                func.count().label("requests"),
                func.coalesce(func.sum(errored), 0).label("errors"),
                func.avg(func.cast(UsageLog.duration_ms, Float)).label("avg_ms"),
            ).where(UsageLog.created_at >= _cutoff(hours))
        )
    ).one()
    requests = int(row.requests or 0)
    errors = int(row.errors or 0)
    avg_ms = float(row.avg_ms) if row.avg_ms is not None else None
    success_rate = (100.0 * (requests - errors) / requests) if requests else None
    return {
        "requests": requests,
        "errors": errors,
        "avg_ms": avg_ms,
        "success_rate": success_rate,
    }


def _cutoff(hours: int):
    """The lower bound of the window, as an instant.

    Computed here rather than by the database. `datetime('now', ...)` is a
    SQLite function that Postgres does not have, and an aware value is the
    only shape both drivers agree on.
    """
    return utcnow() - timedelta(hours=hours)


def _hour_bucket(session: AsyncSession):
    """The created_at hour, as the engine this session is on can express it.

    Postgres has `date_trunc`; SQLite has `strftime`. There is no portable
    spelling, so this is the one place the dialect is named -- and it is read
    off the connection rather than the process configuration, because the
    session is what decides which database the query reaches.
    """
    if session.bind.dialect.name == "postgresql":
        return func.date_trunc(literal_column("'hour'"), UsageLog.created_at)
    return func.strftime("%Y-%m-%dT%H:00:00", UsageLog.created_at)


async def _timeseries(session: AsyncSession, hours: int = 24) -> dict:
    """The same counts as the summary, one row per hour."""
    bucket = _hour_bucket(session).label("bucket")
    errored = case(
        (UsageLog.error_type.isnot(None), 1),
        (UsageLog.status_code >= 400, 1),
        else_=0,
    )
    rows = (
        await session.execute(
            select(
                bucket,
                func.count().label("requests"),
                func.coalesce(func.sum(errored), 0).label("errors"),
                func.avg(func.cast(UsageLog.duration_ms, Float)).label("avg_ms"),
            )
            .where(UsageLog.created_at >= _cutoff(hours))
            .group_by(bucket)
            .order_by(bucket)
        )
    ).all()
    buckets: list[str | None] = []
    requests: list[int] = []
    errors: list[int] = []
    avg_ms: list[float | None] = []
    for r in rows:
        value = r.bucket
        buckets.append(
            value.strftime("%Y-%m-%dT%H:00:00")
            if hasattr(value, "strftime") else value
        )
        requests.append(int(r.requests or 0))
        errors.append(int(r.errors or 0))
        avg_ms.append(round(float(r.avg_ms), 1) if r.avg_ms is not None else None)
    return {"buckets": buckets, "requests": requests, "errors": errors, "avg_ms": avg_ms}


def _overall_status(health: dict, summary: dict) -> tuple[str, str]:
    """Derive (level, label); level in {ok, degraded, outage}.

    A hard-down condition (DB unreachable or all upstreams unreachable) only
    escalates to "outage" after it has persisted for OUTAGE_FAIL_THRESHOLD
    consecutive probe cycles, and not during the warm-up window. Below that it
    surfaces as "degraded" — so one transient blip or a cold first probe can't
    flip the public page to red.
    """
    db_down = health.get("database") == "down"
    total = health.get("upstreams_total", 0)
    up = health.get("upstreams_up", 0)
    upstreams_down = total > 0 and up == 0
    upstreams_partial = total > 0 and up < total
    hard_down_now = db_down or upstreams_down

    consec_fail = health.get("consec_fail", 0)
    probe_count = health.get("probe_count", 0)

    if hard_down_now and probe_count >= WARMUP_PROBES and consec_fail >= OUTAGE_FAIL_THRESHOLD:
        return ("outage", "Major outage")

    sr = summary.get("success_rate")
    high_errors = sr is not None and sr < 95.0

    # A current hard-down still below the outage threshold surfaces as degraded
    # rather than silently reading "operational".
    if hard_down_now or upstreams_partial or high_errors:
        return ("degraded", "Degraded performance")
    return ("ok", "All systems operational")


# ---- Routes (public, no auth) -----------------------------------------------


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request, session: AsyncSession = Depends(get_session)):
    # Anonymous-friendly: render the page whether or not the admin is signed
    # in. The shared nav keys off the session itself, so nothing extra needs
    # passing through here.
    health = _snapshot()
    summary = await _summary(session)
    level, label = _overall_status(health, summary)
    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "health": health,
            "summary": summary,
            "overall_level": level,
            "overall_label": label,
        },
    )


@router.get("/status/data")
async def status_data(session: AsyncSession = Depends(get_session)):
    health = _snapshot()
    summary = await _summary(session)
    series = await _timeseries(session)
    level, label = _overall_status(health, summary)
    return JSONResponse(
        {
            "health": health,
            "summary": summary,
            "series": series,
            "overall": {"level": level, "label": label},
        }
    )
