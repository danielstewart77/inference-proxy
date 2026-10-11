"""`/v1/providers/limits` over HTTP, and the context window on the listing.

The parse is covered in `test_provider_limits.py`. What these pin is the
part a console actually consumes: that the reading is attributed to the
right provider row, that a provider nobody has called is reported as *not
reporting* rather than as idle at zero, that the route refuses an unknown
key, that a model's declared context window and image capability are
relayed beside the percentage of it a conversation rotates at, and that a
model with no window is offered to nobody.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app import provider_limits
from app.db import get_session
from app.main import create_app
from app.orm import Base, Credential, Model, Provider


ANTHROPIC_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.42",
    "anthropic-ratelimit-unified-5h-reset": "1789110000",
}
CODEX_HEADERS = {
    "x-codex-primary-used-percent": "13",
    "x-codex-primary-window-minutes": "300",
}


@pytest_asyncio.fixture
async def client(monkeypatch):
    """The app on a fresh in-memory db, with two providers and a valid key."""
    import app.main as main

    provider_limits.reset()
    engine = create_async_engine("sqlite+aiosqlite://", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with maker() as setup:
        credential = Credential(name="c", kind="static", secret="s")
        setup.add(credential)
        await setup.flush()
        anthropic = Provider(
            name="anthropic",
            label="Anthropic",
            base_url="https://api.anthropic.com",
            messages_path="/v1/messages",
            credential_id=credential.id,
        )
        openai = Provider(
            name="openai",
            label="Codex",
            base_url="https://chatgpt.com",
            responses_path="/backend-api/codex/responses",
            credential_id=credential.id,
        )
        setup.add_all([anthropic, openai])
        await setup.flush()
        setup.add_all(
            [
                Model(
                    deployment_name="claude-opus-5",
                    provider_id=anthropic.id,
                    enabled=True,
                    context_window=1_000_000,
                    accepts_images=True,
                ),
                Model(
                    deployment_name="gpt-5.6",
                    provider_id=openai.id,
                    enabled=True,
                    context_window=400_000,
                    accepts_images=False,
                ),
                Model(
                    deployment_name="claude-unsized",
                    provider_id=anthropic.id,
                    enabled=True,
                    context_window=None,
                ),
                Model(
                    deployment_name="claude-unsurveyed",
                    provider_id=anthropic.id,
                    enabled=True,
                    context_window=200_000,
                ),
            ]
        )
        await setup.commit()

    async def _override():
        async with maker() as s:
            yield s

    app = create_app()
    app.dependency_overrides[get_session] = _override
    monkeypatch.setattr(main, "validate_api_key", lambda *a, **k: True, raising=False)
    monkeypatch.setattr("app.main.validate_api_key", lambda *a, **k: True)
    monkeypatch.setattr("app.main.resolve_requester_role", lambda *a, **k: "admin")
    yield TestClient(app)
    provider_limits.reset()
    await engine.dispose()


def _by_name(payload: dict) -> dict:
    return {row["name"]: row for row in payload["providers"]}


def test_each_provider_carries_only_the_reading_from_its_own_host(client):
    """Two providers, two different numbers, neither borrowed nor summed."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)
    provider_limits.record("chatgpt.com", CODEX_HEADERS)

    providers = _by_name(client.get("/v1/providers/limits").json())

    assert providers["anthropic"]["windows"][0]["used_percent"] == 42.0
    assert providers["openai"]["windows"][0]["used_percent"] == 13.0


def test_a_provider_that_has_taken_no_traffic_says_so_rather_than_reading_zero(client):
    """Requirement 12. An empty bar and an unobserved provider are opposite
    claims, and only one of them means there is headroom."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)

    providers = _by_name(client.get("/v1/providers/limits").json())

    assert providers["openai"]["reporting"] is False
    assert providers["openai"]["windows"] == []
    assert providers["anthropic"]["reporting"] is True


def test_every_configured_provider_appears_even_when_silent(client):
    """Dropping a silent provider would read as 'not configured', which sends
    someone to add a row that already exists."""
    providers = _by_name(client.get("/v1/providers/limits").json())

    assert set(providers) == {"anthropic", "openai"}


def test_a_reading_reports_its_age(client):
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)

    row = _by_name(client.get("/v1/providers/limits").json())["anthropic"]

    assert row["age_seconds"] is not None
    assert row["observed_at"] is not None


def test_an_unknown_key_is_refused(client, monkeypatch):
    """The report names what the hive is spending and how close it is to
    being cut off; it answers on the same port the models listing does."""
    monkeypatch.setattr("app.main.validate_api_key", lambda *a, **k: False)

    assert client.get("/v1/providers/limits").status_code == 401


def test_the_listing_relays_a_declared_context_window(client):
    """Requirement 9's denominator. The proxy owns the model map, so this is
    where the number lives."""
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}

    assert models["claude-opus-5"]["context_window"] == 1_000_000


@pytest.mark.parametrize("harness", [None, "claude", "dsh"])
def test_a_model_with_no_context_window_is_offered_to_no_caller(client, harness):
    """R2 / T2. A conversation rotates at a percentage of its model's window,
    so a model nobody measured has no threshold and cannot be picked — not by
    the unfiltered union, and not by a harness that names itself."""
    url = "/v1/models" if harness is None else f"/v1/models?harness={harness}"
    offered = {m["id"] for m in client.get(url).json()["data"]}

    assert "claude-unsized" not in offered
    assert "claude-opus-5" in offered


def test_whether_a_model_reads_images_is_relayed_in_three_states(client):
    """T3. Unknown stays distinguishable from no: only a declared answer may
    decide that a conversation's images survive a switch."""
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}

    assert models["claude-opus-5"]["accepts_images"] is True
    assert models["gpt-5.6"]["accepts_images"] is False
    assert models["claude-unsurveyed"]["accepts_images"] is None


def test_a_stale_reading_is_reported_as_stale_over_the_wire(client, monkeypatch):
    """The module knows when a figure has aged out; the route has to say so.
    Hardcoding the flag false leaves the page presenting a figure from
    yesterday as this minute's."""
    import time as _time

    provider_limits.record(
        "api.anthropic.com",
        ANTHROPIC_HEADERS,
        now=_time.time() - provider_limits.STALE_AFTER_SECONDS - 10,
    )

    row = _by_name(client.get("/v1/providers/limits").json())["anthropic"]

    assert row["stale"] is True
    assert row["windows"][0]["used_percent"] == 42.0


def test_a_fresh_reading_is_not_reported_as_stale(client):
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS)

    assert _by_name(client.get("/v1/providers/limits").json())["anthropic"]["stale"] is False


def test_a_context_window_can_be_set_and_comes_back_on_the_listing(client, monkeypatch):
    """Without a write path the column, its migration and its relay are an
    elaborate way of returning null forever — every column on the dashboard
    reading "context window unknown" with no way to fix it."""
    # The admin console is session-authed; without this the POST answers 303
    # to the login page and the assertion below would be checking that a
    # redirect happened rather than that anything was written.
    async def _allow(request):
        return None

    monkeypatch.setattr("app.admin.models.require_html_admin", _allow)

    response = client.post(
        "/admin/models/1",
        data={
            "target_uri": "",
            "provider_id": "1",
            "harnesses": "",
            "credential_id": "",
            "api_version": "",
            "auth_scheme": "bearer",
            "label": "",
            "description": "",
            "cost_per_million_input": "",
            "cost_per_million_output": "",
            "cost_per_million_cache_write": "",
            "cost_per_million_cache_read": "",
            "context_window": "393216",
            "rotation_threshold_percent": "30",
            "enabled": "on",
        },
        follow_redirects=False,
    )

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["context_window"] == 393216


def test_an_edit_without_a_context_window_is_refused_and_keeps_the_old_one(
    client, monkeypatch,
):
    """R2. Blanking the window would make the model unpickable without anyone
    deciding it should be."""
    response = _post_model(client, monkeypatch, context_window="")

    assert response.status_code == 400
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["context_window"] == 1_000_000


def test_a_model_cannot_be_added_without_a_context_window(client, monkeypatch):
    """R2. The add form is where every new model gets its window."""
    response = _add_model(client, monkeypatch, deployment_name="claude-new",
                          context_window="")

    assert response.status_code == 400
    rows = client.get("/v1/models").json()["data"]
    assert "claude-new" not in {m["id"] for m in rows}


@pytest.mark.parametrize("answer, expected", [("yes", True), ("no", False), ("", None)])
def test_the_admin_form_sets_whether_a_model_reads_images(
    client, monkeypatch, answer, expected,
):
    """T3. Without a write path the column is null forever."""
    _post_model(client, monkeypatch, model_id=2, provider_id="2",
                context_window="400000", accepts_images=answer)

    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["gpt-5.6"]["accepts_images"] is expected


def test_a_new_model_comes_back_on_the_listing_with_its_image_answer(
    client, monkeypatch,
):
    response = _add_model(client, monkeypatch, deployment_name="claude-new",
                          context_window="200000", accepts_images="yes")

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-new"]["accepts_images"] is True


def _add_model(client, monkeypatch, **fields):
    async def _allow(request):
        return None

    monkeypatch.setattr("app.admin.models.require_html_admin", _allow)
    data = {
        "deployment_name": "claude-new", "target_uri": "", "provider_id": "1",
        "harnesses": "", "credential_id": "", "api_version": "",
        "auth_scheme": "bearer", "label": "", "description": "",
        "cost_per_million_input": "", "cost_per_million_output": "",
        "cost_per_million_cache_write": "", "cost_per_million_cache_read": "",
        "context_window": "200000", "rotation_threshold_percent": "30",
        "effort_levels": "", "enabled": "on",
    }
    data.update(fields)
    return client.post("/admin/models", data=data, follow_redirects=False)


def _post_model(client, monkeypatch, model_id=1, **fields):
    async def _allow(request):
        return None

    monkeypatch.setattr("app.admin.models.require_html_admin", _allow)
    data = {
        "target_uri": "", "provider_id": "1", "harnesses": "", "credential_id": "",
        "api_version": "", "auth_scheme": "bearer", "label": "", "description": "",
        "cost_per_million_input": "", "cost_per_million_output": "",
        "cost_per_million_cache_write": "", "cost_per_million_cache_read": "",
        "context_window": "1000000", "rotation_threshold_percent": "30",
        "effort_levels": "", "enabled": "on",
    }
    data.update(fields)
    return client.post(f"/admin/models/{model_id}", data=data, follow_redirects=False)


def test_a_models_effort_levels_come_back_on_the_listing_in_the_order_set(
    client, monkeypatch,
):
    """Every surface's effort picker draws from this list, in this order."""
    response = _post_model(client, monkeypatch, effort_levels="low, medium, high, MAX")

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["effort_levels"] == ["low", "medium", "high", "max"]


def test_a_model_with_no_effort_levels_lists_an_empty_list(client):
    """Empty is the whole of "this model takes no effort setting"."""
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}

    assert models["gpt-5.6"]["effort_levels"] == []


def test_an_unknown_effort_level_is_refused_rather_than_stored(client, monkeypatch):
    """A stored typo becomes a button that spawns a harness on a value it rejects."""
    response = _post_model(client, monkeypatch, effort_levels="high, turbo")

    assert response.status_code == 400
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["effort_levels"] == []


def test_each_harness_is_offered_only_the_effort_words_it_accepts(client, monkeypatch):
    """Claude takes max and no minimal; Codex the reverse. A word the CLI
    rejects, stored on a conversation, breaks every respawn after it."""
    _post_model(client, monkeypatch, effort_levels="minimal, low, high, max")
    _post_model(client, monkeypatch, model_id=2, provider_id="2",
                context_window="400000", effort_levels="minimal, low, high, max")

    def levels(harness, model):
        rows = client.get(f"/v1/models?harness={harness}").json()["data"]
        return {m["id"]: m for m in rows}[model]["effort_levels"]

    assert levels("claude", "claude-opus-5") == ["low", "high", "max"]
    assert levels("codex", "gpt-5.6") == ["minimal", "low", "high"]
    # dsh passes no effort to its model, so it is offered none.
    assert levels("dsh", "gpt-5.6") == []


def test_effort_levels_set_when_a_model_is_added_come_back_on_the_listing(
    client, monkeypatch,
):
    """The add form is the first place a model gets its levels; dropping
    them there lists a new model as taking no effort until someone edits it."""
    async def _allow(request):
        return None

    monkeypatch.setattr("app.admin.models.require_html_admin", _allow)
    response = client.post("/admin/models", data={
        "deployment_name": "claude-new", "target_uri": "", "provider_id": "1",
        "harnesses": "", "credential_id": "", "api_version": "",
        "auth_scheme": "bearer", "label": "", "description": "",
        "cost_per_million_input": "", "cost_per_million_output": "",
        "cost_per_million_cache_write": "", "cost_per_million_cache_read": "",
        "context_window": "200000", "rotation_threshold_percent": "30",
        "effort_levels": "low, high", "enabled": "on",
    }, follow_redirects=False)

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-new"]["effort_levels"] == ["low", "high"]


def test_a_rotation_threshold_set_on_the_edit_form_comes_back_on_the_harness_listing(
    client, monkeypatch,
):
    """The threshold is the model's, so the minds read it off the same row as
    the window it is a percentage of."""
    response = _post_model(client, monkeypatch, rotation_threshold_percent="45")

    assert response.status_code in (303, 302)
    rows = client.get("/v1/models?harness=claude").json()["data"]
    model = {m["id"]: m for m in rows}["claude-opus-5"]
    assert model["context_window"] == 1_000_000
    assert model["rotation_threshold_percent"] == 45


def test_a_rotation_threshold_set_when_a_model_is_added_comes_back_on_the_listing(
    client, monkeypatch,
):
    response = _add_model(client, monkeypatch, deployment_name="claude-new",
                          rotation_threshold_percent="60")

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-new"]["rotation_threshold_percent"] == 60


def test_a_model_cannot_be_added_without_a_rotation_threshold(client, monkeypatch):
    response = _add_model(client, monkeypatch, deployment_name="claude-new",
                          rotation_threshold_percent="")

    assert response.status_code == 400
    rows = client.get("/v1/models").json()["data"]
    assert "claude-new" not in {m["id"] for m in rows}


@pytest.mark.parametrize("raw", ["", "0", "100", "thirty"])
def test_an_edit_with_a_missing_or_out_of_range_threshold_is_refused_and_keeps_the_old_one(
    client, monkeypatch, raw,
):
    """Zero rotates every turn and a hundred never rotates before the overflow."""
    _post_model(client, monkeypatch, rotation_threshold_percent="45")

    response = _post_model(client, monkeypatch, rotation_threshold_percent=raw)

    assert response.status_code == 400
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["rotation_threshold_percent"] == 45


@pytest.mark.parametrize("raw", ["1", "99"])
def test_the_ends_of_the_threshold_range_are_accepted(client, monkeypatch, raw):
    response = _post_model(client, monkeypatch, rotation_threshold_percent=raw)

    assert response.status_code in (303, 302)
    models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert models["claude-opus-5"]["rotation_threshold_percent"] == int(raw)


def test_the_add_form_offers_thirty_percent_as_the_rotation_threshold(client, monkeypatch):
    async def _allow(request):
        return None

    monkeypatch.setattr("app.admin.models.require_html_admin", _allow)
    html = client.get("/admin/models/_add-form").text

    field = next(
        line for line in html.splitlines() if 'name="rotation_threshold_percent"' in line
    )
    assert 'value="30"' in field
    assert "required" in field
