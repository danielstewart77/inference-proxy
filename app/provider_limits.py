"""How much of each provider's subscription allowance is gone.

Both upstreams say so on every ordinary call. Anthropic returns
``anthropic-ratelimit-unified-5h-utilization`` (a fraction) alongside its
7-day sibling and their reset epochs; the Codex backend returns
``x-codex-primary-used-percent`` (already a percentage) with the window's
length in minutes. So the figure is a byproduct of traffic the proxy is
already carrying, which is the whole reason it is read here rather than
polled: the credentials are subscription OAuth — the same ones the minds
spend — and Anthropic's own usage endpoint answered 429 on a single probe.
A monitor that hammered it would degrade the allowance it exists to report.

The store is in-memory and per-process. That is honest for what it holds:
the last thing this proxy observed. It is not a usage ledger — ``usage_log``
is — and a figure that did not survive a restart is one nobody should have
been reading as current anyway, which is why every entry carries its age.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit

#: Past this, a figure keeps its number but stops claiming to be current.
#: Anthropic's short window is five hours, so a quarter-hour-old reading is
#: still meaningful; a day-old one is a different conversation entirely.
STALE_AFTER_SECONDS = 900.0

#: Window lengths, in minutes, for the two rolling limits both vendors run.
#: Anthropic names its windows in the header; Codex names its *rank*, so the
#: length is what the two have in common and what a reader can compare.
_FIVE_HOUR_MINUTES = 300
_SEVEN_DAY_MINUTES = 10080


@dataclass(frozen=True)
class LimitWindow:
    """One rolling limit: how much of it is spent, and when it resets."""

    label: str
    used_percent: float
    resets_at: Optional[int]
    window_minutes: Optional[int]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "used_percent": self.used_percent,
            "resets_at": self.resets_at,
            "window_minutes": self.window_minutes,
        }


_lock = threading.Lock()
_store: dict[str, dict] = {}


def host_of(url: str) -> str:
    """The host a call went to, which is how a figure is attributed.

    Callers hold a URL, not a provider row — ``post_with_retries`` is one
    choke point for every upstream shape and knows nothing about the
    database. Matching the host back to a provider is the reader's job.
    """
    return urlsplit(url).hostname or ""


def _number(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _epoch(raw: Optional[str]) -> Optional[int]:
    value = _number(raw)
    return int(value) if value is not None else None


def _anthropic_windows(headers: Mapping[str, str]) -> list[LimitWindow]:
    """Anthropic reports a *fraction*: 0.07 is seven percent of the window."""
    found: list[LimitWindow] = []
    for label, minutes in (("5h", _FIVE_HOUR_MINUTES), ("7d", _SEVEN_DAY_MINUTES)):
        utilization = _number(headers.get(f"anthropic-ratelimit-unified-{label}-utilization"))
        if utilization is None:
            continue
        found.append(
            LimitWindow(
                label=label,
                used_percent=round(utilization * 100, 4),
                resets_at=_epoch(headers.get(f"anthropic-ratelimit-unified-{label}-reset")),
                window_minutes=minutes,
            )
        )
    return found


def _codex_windows(headers: Mapping[str, str]) -> list[LimitWindow]:
    """Codex reports a percentage already, and names windows by rank.

    The rank is relabelled by length so the two providers can be read side
    by side: "primary" against "5h" is not a comparison anybody can make.
    """
    found: list[LimitWindow] = []
    for rank in ("primary", "secondary"):
        used = _number(headers.get(f"x-codex-{rank}-used-percent"))
        if used is None:
            continue
        minutes = _number(headers.get(f"x-codex-{rank}-window-minutes"))
        window_minutes = int(minutes) if minutes is not None else None
        label = "7d" if window_minutes == _SEVEN_DAY_MINUTES else "5h"
        found.append(
            LimitWindow(
                label=label,
                used_percent=round(used, 4),
                resets_at=_epoch(headers.get(f"x-codex-{rank}-reset-at")),
                window_minutes=window_minutes,
            )
        )
    return found


def parse_headers(headers: Mapping[str, str]) -> list[LimitWindow]:
    """Every rolling limit this response reported. Empty means it said none.

    Empty is load-bearing: it is what distinguishes an upstream that does not
    publish limits (Ollama, or an error response) from one reporting zero use.
    """
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return _anthropic_windows(lowered) + _codex_windows(lowered)


def record(host: str, headers: Mapping[str, str], *, now: Optional[float] = None) -> None:
    """Remember what ``host`` last said about its allowance.

    A response carrying no limit headers is not news and must not overwrite
    what the host last reported — otherwise one local Ollama turn blanks the
    Anthropic panel until the next Anthropic call happens to land.
    """
    windows = parse_headers(headers)
    if not windows or not host:
        return
    with _lock:
        _store[host] = {
            "windows": [w.as_dict() for w in windows],
            "observed_at": time.time() if now is None else now,
        }


def snapshot(*, now: Optional[float] = None) -> dict[str, dict]:
    """The latest figure per host, each with its age.

    A host absent from this mapping has never reported, which the reader must
    render as unknown rather than as an empty bar — "no pressure" and "no
    information" are opposite answers.
    """
    moment = time.time() if now is None else now
    with _lock:
        entries = {host: dict(entry) for host, entry in _store.items()}
    for entry in entries.values():
        age = moment - entry["observed_at"]
        entry["age_seconds"] = age
        entry["stale"] = age > STALE_AFTER_SECONDS
    return entries


def reset() -> None:
    """Drop everything. For tests, which must not inherit each other's state."""
    with _lock:
        _store.clear()
