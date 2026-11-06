"""What only Postgres can prove.

Every test here failed before the cutover work and passes after it. They are
separated from the SQLite suite because each one guards a behaviour SQLite
cannot express: a timezone the driver reinterprets, a date function only one
engine has, an identity column, a search path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.admin.usage import _report
from app.config import config
from app.orm import Client, UsageLog
from app.retention import prune_once
from app.status import _summary, _timeseries
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
    return UsageLog(created_at=datetime.now(timezone.utc) - age, **fields)


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
    assert [r.model_name for r in report["by_model"]] == ["m"]
    assert report["by_model"][0].requests == 1
    assert [r.client_name for r in report["by_client"]] == ["probe"]
    assert report["by_client"][0].requests == 1


@requires_postgres
async def test_the_status_summary_counts_requests_and_errors(pg_session):
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(minutes=5)))
    pg_session.add(_row(client.id, age=timedelta(minutes=6), status_code=500))
    pg_session.add(_row(client.id, age=timedelta(minutes=7),
                        error_type="upstream"))
    pg_session.add(_row(client.id, age=timedelta(days=5)))
    await pg_session.commit()

    summary = await _summary(pg_session, hours=24)

    assert summary["requests"] == 3
    assert summary["errors"] == 2


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
    pg_session.add(_row(client.id, age=timedelta(days=5)))
    await pg_session.commit()

    series = await _timeseries(pg_session, hours=24)

    assert len(series["buckets"]) == 2
    assert sorted(series["requests"]) == [1, 2]


@requires_postgres
async def test_pruning_deletes_the_old_rows_and_keeps_the_rest(pg_session,
                                                               monkeypatch):
    monkeypatch.setattr(config, "usage_prune_days", 7)
    client = await _client(pg_session)
    pg_session.add(_row(client.id, age=timedelta(days=8)))
    pg_session.add(_row(client.id, age=timedelta(days=9)))
    pg_session.add(_row(client.id, age=timedelta(days=6)))
    pg_session.add(_row(client.id, age=timedelta(days=60)))
    await pg_session.commit()

    deleted = await prune_once(pg_session)

    assert deleted == 3
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


@requires_postgres
async def test_a_stored_usage_row_reads_back_with_its_own_fields(pg_session):
    """The row that can be stored is stored, with its values where they go.

    Guarding only the failure direction leaves the whole mapping free: input
    and output tokens could be swapped, or the caller's address dropped, and
    nothing would say so.
    """
    from app.usage import store_usage_row

    client = await _client(pg_session)
    await store_usage_row(
        pg_session, client_id=client.id, key_id=None, model_name="m",
        endpoint="/v1/messages", status_code=200, error_type=None,
        duration_ms=123, client_ip_str="10.1.2.3", input_tokens=11,
        output_tokens=22, total_tokens=33,
        cache_creation_input_tokens=44, cache_read_input_tokens=55,
    )

    row = (await pg_session.execute(text(
        "SELECT input_tokens, output_tokens, total_tokens, duration_ms, "
        "client_ip, cache_creation_input_tokens, cache_read_input_tokens "
        "FROM usage_log"))).one()
    assert row.input_tokens == 11
    assert row.output_tokens == 22
    assert row.total_tokens == 33
    assert row.duration_ms == 123
    assert row.client_ip == "10.1.2.3"
    assert row.cache_creation_input_tokens == 44
    assert row.cache_read_input_tokens == 55


@requires_postgres
async def test_the_orm_builds_identity_keys_on_postgres(pg_session):
    """The fixture's schema comes from the ORM, not from Alembic.

    Both have to produce the same shape, or a fresh install and the migrated
    database diverge again -- so the ORM's own identity declaration is
    checked where it is actually applied.
    """
    rows = (await pg_session.execute(text(
        "SELECT table_name, is_identity, data_type FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND column_name = 'id' "
        "ORDER BY table_name"))).all()

    assert rows, "the fixture built no tables"
    for row in rows:
        assert row.is_identity == "YES", row.table_name
        assert row.data_type == "bigint", row.table_name
