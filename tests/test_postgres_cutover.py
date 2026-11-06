"""What only Postgres can prove.

Every test here failed before the cutover work and passes after it. They are
separated from the SQLite suite because each one guards a behaviour SQLite
cannot express: a timezone the driver reinterprets, a date function only one
engine has, an identity column, a search path.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import text

from app.admin.usage import _report
from app.config import config
from app.orm import Client, UsageLog
from app.retention import prune_once
from app.status import _summary, _timeseries
from app.timeutil import utcnow
from tests.conftest import requires_postgres

pytestmark = requires_postgres


async def _client(session) -> Client:
    client = Client(name="probe", kind="app")
    session.add(client)
    await session.commit()
    await session.refresh(client)
    return client


def _row(client_id: int, *, age: timedelta, **over) -> UsageLog:
    fields = dict(
        client_id=client_id, key_id=None, model_name="m", endpoint="/v1/messages",
        status_code=200, error_type=None, duration_ms=100, input_tokens=1,
        output_tokens=1, total_tokens=2,
    )
    fields.update(over)
    return UsageLog(created_at=utcnow() - age, **fields)


@requires_postgres
async def test_the_window_includes_a_row_just_inside_it_and_excludes_one_just_outside(
        pg_session):
    """The boundary is an instant, not a local wall clock.

    A naive-UTC boundary is encoded by asyncpg as the client process' local
    time, which on this host moves the cutoff six hours. Rows either side of
    a one-day boundary are what notices: a shifted cutoff drops the row that
    belongs in the report, and the page looks healthy and empty.
    """
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(hours=23)))
    pg_session.add(_row(client.id, age=timedelta(hours=25)))
    await pg_session.commit()

    report = await _report(pg_session, days=1)

    assert [r.endpoint for r in report["by_endpoint"]] == ["/v1/messages"]
    assert report["by_endpoint"][0].requests == 1


@requires_postgres
async def test_the_status_summary_counts_requests_and_errors(pg_session):
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(minutes=5)))
    pg_session.add(_row(client.id, age=timedelta(minutes=6), status_code=500,
                        error_type="upstream"))
    pg_session.add(_row(client.id, age=timedelta(days=5)))
    await pg_session.commit()

    summary = await _summary(pg_session, hours=24)

    assert summary["requests"] == 2
    assert summary["errors"] == 1


@requires_postgres
async def test_the_status_timeseries_separates_two_hours(pg_session):
    """Two rows an hour apart are two buckets, not one.

    Truncating to the day instead of the hour, or returning an empty series,
    both answer this wrong.
    """
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(minutes=10)))
    pg_session.add(_row(client.id, age=timedelta(minutes=70)))
    pg_session.add(_row(client.id, age=timedelta(minutes=75)))
    await pg_session.commit()

    series = await _timeseries(pg_session, hours=24)

    assert len(series["buckets"]) == 2
    assert sorted(series["requests"]) == [1, 2]


@requires_postgres
async def test_pruning_deletes_the_old_rows_and_keeps_the_rest(pg_session,
                                                               monkeypatch):
    monkeypatch.setattr(config, "usage_prune_days", 90)
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(days=config.usage_prune_days + 1)))
    pg_session.add(_row(client.id, age=timedelta(days=config.usage_prune_days + 2)))
    pg_session.add(_row(client.id, age=timedelta(days=1)))
    await pg_session.commit()

    deleted = await prune_once(pg_session)

    assert deleted == 2
    remaining = (await pg_session.execute(
        text("SELECT count(*) FROM usage_log"))).scalar()
    assert remaining == 1


@requires_postgres
async def test_a_generated_id_does_not_collide_with_a_loaded_one(pg_session):
    """A row carrying an explicit id does not advance the identity counter.

    This is the shape of the cutover load, and the counter has to be moved
    past the rows afterwards or the next insert collides on the primary key.
    """
    client = await _client(pg_session)
    await pg_session.execute(text(
        "INSERT INTO usage_log (id, client_id, model_name, endpoint, status_code)"
        " VALUES (5000, :c, 'm', '/v1/messages', 200)"), {"c": client.id})
    await pg_session.execute(text(
        "SELECT setval(pg_get_serial_sequence('usage_log', 'id'), "
        "(SELECT max(id) FROM usage_log), true)"))
    await pg_session.commit()

    pg_session.add(_row(client.id, age=timedelta(minutes=1)))
    await pg_session.commit()

    top = (await pg_session.execute(
        text("SELECT max(id) FROM usage_log"))).scalar()
    assert top == 5001


@requires_postgres
async def test_a_failed_usage_insert_is_reported_not_swallowed(pg_session):
    """A usage row that cannot be stored must say so.

    The old code caught every exception and printed, which turned a primary
    key collision into metering that silently stopped while every request
    kept answering normally.
    """
    client = await _client(pg_session)
    await pg_session.execute(text(
        "INSERT INTO usage_log (id, client_id, model_name, endpoint, status_code)"
        " VALUES (10, :c, 'm', '/v1/messages', 200)"), {"c": client.id})
    await pg_session.commit()
    # The counter is left one short of the row that already exists, which is
    # exactly the state a load that preserved ids leaves behind.
    await pg_session.execute(text(
        "SELECT setval(pg_get_serial_sequence('usage_log', 'id'), 10, false)"))
    await pg_session.commit()

    from app.usage import store_usage_row

    with pytest.raises(Exception):
        await store_usage_row(
            pg_session, client_id=client.id, key_id=None, model_name="m",
            endpoint="/v1/messages", status_code=200, error_type=None,
            duration_ms=1, client_ip_str=None, input_tokens=1,
            output_tokens=1, total_tokens=2,
            cache_creation_input_tokens=None, cache_read_input_tokens=None,
        )
