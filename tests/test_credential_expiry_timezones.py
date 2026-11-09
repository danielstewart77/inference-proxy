"""Freshness is judged whatever shape the timestamp arrives in.

Postgres hands back an aware datetime for `timestamptz`; a row built in code
can carry a naive one. `_is_fresh` has to answer for both, and before this it
raised a TypeError on the aware one — which is every read on Postgres.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.credentials import REFRESH_SKEW_SECONDS, _is_fresh
from app.orm import Credential


def _cred(expires_at: datetime) -> Credential:
    return Credential(
        name="probe", kind="anthropic_oauth", access_token="tok",
        expires_at=expires_at,
    )


def test_aware_expiry_in_the_future_is_fresh():
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    assert _is_fresh(_cred(future)) is True


def test_aware_expiry_in_the_past_is_stale():
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    assert _is_fresh(_cred(past)) is False


def test_naive_expiry_in_the_future_is_fresh():
    """A naive value is UTC by convention. It has to keep meaning UTC, not local time."""
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(tzinfo=None)
    assert _is_fresh(_cred(future)) is True


def test_naive_expiry_in_the_past_is_stale():
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None)
    assert _is_fresh(_cred(past)) is False


def test_expiry_inside_the_refresh_skew_is_stale():
    """A token expiring sooner than the skew is refreshed, not used."""
    edge = datetime.now(timezone.utc) + timedelta(
        seconds=REFRESH_SKEW_SECONDS - 30
    )
    assert _is_fresh(_cred(edge)) is False
