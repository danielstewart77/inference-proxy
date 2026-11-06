"""One instant, two engines.

Postgres' `timestamptz` columns decode to timezone-aware datetimes; SQLite
has no timezone at all and decodes to naive ones. A naive datetime is also
the one value a driver will silently reinterpret: `asyncpg` encodes it as the
*client process'* local wall clock, so a naive-UTC value written from this
host lands six hours in the future without raising anything.

So everything this package writes is aware UTC — correct on Postgres, and on
SQLite stored as the same UTC wall clock it has always stored — and
everything it reads is coerced before being compared, because a stored naive
value is UTC by every convention this codebase has ever followed.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Now, as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """`value` as aware UTC, treating a naive value as already UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
