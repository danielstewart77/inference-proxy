"""Subscription pressure, read off the responses we already proxy.

Requirement 11 asks how much of each provider's allowance is gone. Both
upstreams answer that on every ordinary call — Anthropic in
``anthropic-ratelimit-unified-*``, the Codex backend in ``x-codex-*`` — so
the figure is a byproduct of traffic rather than a poll against an endpoint
that is itself rate-limited (a probe of Anthropic's usage route returned
429). These tests pin the parse, the per-provider attribution, and the
three states a reader has to tell apart: reporting, never-seen, and stale.
"""

from __future__ import annotations

import pytest

from app import provider_limits


ANTHROPIC_HEADERS = {
    "anthropic-ratelimit-unified-5h-utilization": "0.07",
    "anthropic-ratelimit-unified-5h-reset": "1789110000",
    "anthropic-ratelimit-unified-7d-utilization": "0.28",
    "anthropic-ratelimit-unified-7d-reset": "1789344000",
    "anthropic-ratelimit-unified-status": "allowed",
}

CODEX_HEADERS = {
    "x-codex-primary-used-percent": "15",
    "x-codex-primary-window-minutes": "300",
    "x-codex-primary-reset-at": "1789106451",
    "x-codex-secondary-used-percent": "21",
    "x-codex-secondary-window-minutes": "10080",
    "x-codex-secondary-reset-at": "1789457768",
    "x-codex-plan-type": "plus",
}


@pytest.fixture(autouse=True)
def _clean_store():
    provider_limits.reset()
    yield
    provider_limits.reset()


def test_anthropic_utilisation_fraction_becomes_a_percentage():
    """Anthropic reports 0.07; a reader wants 7 percent, not 0.07 percent."""
    windows = provider_limits.parse_headers(ANTHROPIC_HEADERS)

    by_label = {w.label: w for w in windows}
    assert by_label["5h"].used_percent == 7.0
    assert by_label["7d"].used_percent == 28.0


def test_anthropic_windows_carry_their_reset_and_length():
    windows = {w.label: w for w in provider_limits.parse_headers(ANTHROPIC_HEADERS)}

    assert windows["5h"].resets_at == 1789110000
    assert windows["5h"].window_minutes == 300
    assert windows["7d"].resets_at == 1789344000
    assert windows["7d"].window_minutes == 10080


def test_codex_used_percent_is_already_a_percentage():
    """The Codex backend reports 15 meaning 15 percent. Scaling it would
    report a plan at 15% as one at 1500%, or at 0.15%."""
    windows = {w.label: w for w in provider_limits.parse_headers(CODEX_HEADERS)}

    assert windows["5h"].used_percent == 15.0
    assert windows["7d"].used_percent == 21.0


def test_codex_windows_are_named_by_their_length_not_their_rank():
    """``primary``/``secondary`` mean nothing to a reader. The window length
    is what both providers actually have in common, so the labels come from
    the minutes rather than from the vendor's ordering."""
    windows = {w.label: w for w in provider_limits.parse_headers(CODEX_HEADERS)}

    assert windows["5h"].window_minutes == 300
    assert windows["7d"].window_minutes == 10080
    assert windows["5h"].resets_at == 1789106451


def test_headers_with_no_limit_information_parse_to_nothing():
    """Absence has to be distinguishable from zero use — an empty parse is
    what lets the snapshot say 'never seen' instead of drawing an empty bar."""
    assert provider_limits.parse_headers({"content-type": "application/json"}) == []


def test_a_provider_reports_only_its_own_figures():
    """Requirement 11 says separately. Attribution by host is what stops one
    provider's pressure being read as the other's, or the two being summed."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)
    provider_limits.record("chatgpt.com", CODEX_HEADERS, now=1000.0)

    snapshot = provider_limits.snapshot(now=1000.0)

    anthropic = {w["label"]: w["used_percent"] for w in snapshot["api.anthropic.com"]["windows"]}
    codex = {w["label"]: w["used_percent"] for w in snapshot["chatgpt.com"]["windows"]}
    assert anthropic == {"5h": 7.0, "7d": 28.0}
    assert codex == {"5h": 15.0, "7d": 21.0}


def test_a_host_never_seen_is_absent_rather_than_zero():
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)

    assert "chatgpt.com" not in provider_limits.snapshot(now=1000.0)


def test_a_recorded_figure_carries_how_old_it_is():
    """Requirement 27: every figure says when it was gathered, so a page
    that stops refreshing ages visibly instead of looking current."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)

    entry = provider_limits.snapshot(now=1240.0)["api.anthropic.com"]

    assert entry["observed_at"] == 1000.0
    assert entry["age_seconds"] == 240.0
    assert entry["stale"] is False


def test_a_figure_past_the_staleness_window_is_marked_stale():
    """It keeps its number — the last known value with its age beats a blank
    panel — but it must not be presented as current."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)

    entry = provider_limits.snapshot(
        now=1000.0 + provider_limits.STALE_AFTER_SECONDS + 1
    )["api.anthropic.com"]

    assert entry["stale"] is True
    assert entry["windows"][0]["used_percent"] == 7.0


def test_a_later_response_replaces_the_earlier_figure():
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)
    provider_limits.record(
        "api.anthropic.com",
        {**ANTHROPIC_HEADERS, "anthropic-ratelimit-unified-5h-utilization": "0.91"},
        now=1100.0,
    )

    entry = provider_limits.snapshot(now=1100.0)["api.anthropic.com"]
    windows = {w["label"]: w["used_percent"] for w in entry["windows"]}

    assert windows["5h"] == 91.0
    assert entry["observed_at"] == 1100.0


def test_a_response_carrying_no_limit_headers_leaves_the_last_figure_intact():
    """An Ollama turn, or an upstream error, must not erase what Anthropic
    last told us — recording an empty parse would blank the panel on the
    next local call."""
    provider_limits.record("api.anthropic.com", ANTHROPIC_HEADERS, now=1000.0)
    provider_limits.record("api.anthropic.com", {"content-type": "application/json"}, now=1100.0)

    entry = provider_limits.snapshot(now=1100.0)["api.anthropic.com"]

    assert entry["observed_at"] == 1000.0
    assert entry["windows"][0]["used_percent"] == 7.0


def test_a_malformed_utilisation_value_is_ignored_rather_than_crashing():
    """Upstream header shapes are not ours to guarantee. A junk value costs
    one window, not the proxy."""
    windows = provider_limits.parse_headers(
        {
            "anthropic-ratelimit-unified-5h-utilization": "unknown",
            "anthropic-ratelimit-unified-7d-utilization": "0.28",
        }
    )

    assert [w.label for w in windows] == ["7d"]


def test_host_is_taken_from_a_url_so_callers_pass_what_they_have():
    assert provider_limits.host_of("https://api.anthropic.com/v1/messages") == "api.anthropic.com"
    assert provider_limits.host_of("https://chatgpt.com/backend-api/codex/responses") == "chatgpt.com"


# --- the capture is actually wired into the upstream call -------------------


class _FakeResponse:
    def __init__(self, status_code, headers):
        self.status_code = status_code
        self.headers = headers

    async def aread(self):
        return b""

    async def aclose(self):
        return None


class _FakeClient:
    """Answers whatever the test queued, in order."""

    def __init__(self, responses):
        self._responses = list(responses)

    def build_request(self, *args, **kwargs):
        return object()

    async def send(self, *args, **kwargs):
        return self._responses.pop(0)

    async def post(self, *args, **kwargs):
        return self._responses.pop(0)


def test_an_upstream_response_records_its_allowance(monkeypatch):
    """The one line that populates the whole feature. Deleting it leaves
    every provider reporting `reporting: false` forever, on a page whose
    entire job is showing allowance — and the rest of the suite stays green,
    because everything else tests the store rather than the writer."""
    import app.azure as azure

    monkeypatch.setattr(
        azure, "shared_client", lambda: _FakeClient([_FakeResponse(200, ANTHROPIC_HEADERS)])
    )

    import asyncio

    asyncio.run(
        azure.post_with_retries(
            "https://api.anthropic.com/v1/messages", {}, {}, stream=False, log_prefix="[t]"
        )
    )

    snapshot = provider_limits.snapshot()
    assert snapshot["api.anthropic.com"]["windows"][0]["used_percent"] == 7.0


def test_a_rate_limited_response_records_before_it_is_retried(monkeypatch):
    """A 429 carries the reading that matters most — the account nearly
    spent. Recording only the final attempt is how the near-exhausted
    figures are the ones that never reach the page."""
    import app.azure as azure

    hot = {**ANTHROPIC_HEADERS, "anthropic-ratelimit-unified-5h-utilization": "0.99"}
    monkeypatch.setattr(azure, "retry_delay", lambda *a, **k: 0)
    monkeypatch.setattr(
        azure,
        "shared_client",
        lambda: _FakeClient([_FakeResponse(429, hot), _FakeResponse(200, {})]),
    )

    import asyncio

    asyncio.run(
        azure.post_with_retries(
            "https://api.anthropic.com/v1/messages", {}, {}, stream=False, log_prefix="[t]"
        )
    )

    assert provider_limits.snapshot()["api.anthropic.com"]["windows"][0]["used_percent"] == 99.0
