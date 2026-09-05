"""The Anthropic-only listing path is retired in favour of one merged listing.

Requirement 3: a caller still asking on the old path is refused with an
error status. It matters that this is not the catch-all's 200 — a client
parsing that as an empty catalog shows a mind with no models, which reads
as a broken mind rather than as a stale URL somebody can fix.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_the_retired_anthropic_listing_path_is_refused_not_answered():
    with TestClient(create_app()) as client:
        response = client.get("/v1/anthropic/models")

    assert response.status_code == 410
    assert "/v1/models" in response.json()["detail"]
