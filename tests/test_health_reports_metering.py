"""Metering that has stopped is invisible from a request.

Every answer is normal and the usage log simply stops growing, which is what
made a swallowed primary-key collision survivable for days. The count has to
be readable from outside the process or it is the same as no count.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import usage
from app.main import create_app


def test_health_reports_how_many_usage_rows_were_lost(db_maker, monkeypatch):
    monkeypatch.setattr(usage, "_failures", 4)

    with TestClient(create_app()) as client:
        body = client.get("/health").json()

    assert body["metering_failures"] == 4


async def test_a_lost_usage_row_increments_the_count(session, monkeypatch):
    """The background insert path is the one that loses rows.

    It cannot raise -- a request must not fail because its metering did -- so
    the only way the loss is ever visible is the count going up.
    """
    monkeypatch.setattr(usage, "_failures", 0)
    monkeypatch.setattr(usage, "SessionLocal", lambda: _Reusable(session))

    # A client id no row carries: the foreign key refuses it.
    await usage._insert(
        client_id=987654321, key_id=None, model_name="m",
        endpoint="/v1/messages", status_code=200, error_type=None,
        duration_ms=1, client_ip_str=None, input_tokens=1, output_tokens=1,
        total_tokens=2, cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )

    assert usage.metering_failures() == 1


class _Reusable:
    """Hands the fixture's session to code that opens its own."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        await self._session.rollback()
        return False
