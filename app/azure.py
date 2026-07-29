"""Azure-side helpers shared across all proxy clients.

URLs and credentials are no longer configured here — each deployment row
in the `models` table carries its own target URI, API key, and api_version.
Use `app.deployments.resolve_deployment(...)` to look one up.
"""

from __future__ import annotations

import asyncio

import httpx

from app.utils import log

_shared_client: httpx.AsyncClient | None = None


def shared_client() -> httpx.AsyncClient:
    """Process-wide upstream HTTP client.

    Keep-alive connection reuse means most requests skip the DNS lookup and
    TLS handshake entirely. The host's corporate DNS resolvers fail
    intermittently, so every avoided lookup is an avoided connect error.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=300.0,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=120.0,
            ),
        )
    return _shared_client


async def close_shared_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


def retry_delay(attempt: int, *, is_rate_limit: bool = False) -> float:
    """Exponential backoff. Longer waits on 429s to let Azure free capacity."""
    if is_rate_limit:
        return min(3.0 * (2 ** (attempt - 1)), 30.0)
    return min(1.0 * (2 ** (attempt - 1)), 10.0)


# 529 is Anthropic's overloaded_error; the rest are transient upstream/infra.
RETRYABLE_UPSTREAM_STATUS = frozenset({429, 500, 502, 503, 504, 529})
MAX_UPSTREAM_ATTEMPTS = 5


async def post_with_retries(
    url: str, headers: dict, body: dict, *, stream: bool, log_prefix: str
) -> httpx.Response:
    """Open an upstream POST on the shared client, retrying connect failures
    and retryable HTTP statuses with backoff.

    Retrying here is safe because nothing has been sent to the client yet.
    Returns the successful response, or the final failed response for the
    caller's error passthrough. Raises the final connect/timeout error.
    """
    client = shared_client()
    for attempt in range(1, MAX_UPSTREAM_ATTEMPTS + 1):
        try:
            if stream:
                response = await client.send(
                    client.build_request("POST", url, headers=headers, json=body),
                    stream=True,
                )
            else:
                response = await client.post(url, headers=headers, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt >= MAX_UPSTREAM_ATTEMPTS:
                log(
                    f"{log_prefix} Upstream connect failed on all "
                    f"{MAX_UPSTREAM_ATTEMPTS} attempts: {exc!r}"
                )
                raise
            delay = retry_delay(attempt)
            log(
                f"{log_prefix} Upstream connect failed "
                f"(attempt {attempt}/{MAX_UPSTREAM_ATTEMPTS}): {exc!r} — "
                f"retrying in {delay}s"
            )
            await asyncio.sleep(delay)
            continue

        if (
            response.status_code in RETRYABLE_UPSTREAM_STATUS
            and attempt < MAX_UPSTREAM_ATTEMPTS
        ):
            error_body = await response.aread()
            await response.aclose()
            delay = retry_delay(
                attempt, is_rate_limit=response.status_code in (429, 529)
            )
            log(
                f"{log_prefix} Upstream {response.status_code} "
                f"(attempt {attempt}/{MAX_UPSTREAM_ATTEMPTS}): "
                f"{error_body[:300]} — retrying in {delay}s"
            )
            await asyncio.sleep(delay)
            continue

        return response
    raise AssertionError("unreachable")


def is_invalid_encrypted_content(payload: bytes | str) -> bool:
    """Azure 400 fires when a turn carries encrypted reasoning blobs that the
    current deployment instance can't decrypt — typically because the
    deployment was swapped (admin edit, key rotation) since the blob was
    minted. Detected by Azure's error code, which is stable in the body.
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="ignore")
    return "invalid_encrypted_content" in payload


def strip_encrypted_reasoning(body: dict) -> tuple[dict, int]:
    """Return a copy of `body` with all reasoning items removed from `input`.

    The Responses API requires that *all* reasoning items between two
    function calls be present or none, so we drop the whole class rather
    than a subset. The model just restarts its reasoning chain on retry.
    """
    new_body = dict(body)
    input_items = new_body.get("input")
    if not isinstance(input_items, list):
        return new_body, 0
    kept = [
        it for it in input_items
        if not (isinstance(it, dict) and it.get("type") == "reasoning")
    ]
    removed = len(input_items) - len(kept)
    if removed:
        new_body["input"] = kept
    return new_body, removed


# Per-item fields that newer clients attach to `input` items but Azure's
# Responses API rejects with an `unknown_parameter` 400. They carry no meaning
# upstream, so we drop them before forwarding. Extend this set as new clients
# introduce their own passthrough metadata fields.
UNSUPPORTED_INPUT_ITEM_FIELDS = frozenset(
    {"internal_chat_message_metadata_passthrough"}
)


def strip_unsupported_input_fields(body: dict) -> tuple[dict, int]:
    """Return a copy of `body` with Azure-unsupported per-item fields removed
    from every `input` item.

    Clients such as recent Claude Code / Codex builds tack metadata onto each
    input item (e.g. `internal_chat_message_metadata_passthrough`). Azure's
    Responses API does not recognise these and rejects the whole request with
    a 400 `unknown_parameter`. We strip them up front so the request succeeds.

    Returns the (possibly rewritten) body and the number of items modified.
    """
    new_body = dict(body)
    input_items = new_body.get("input")
    if not isinstance(input_items, list):
        return new_body, 0
    modified = 0
    new_items: list = []
    for it in input_items:
        if isinstance(it, dict) and UNSUPPORTED_INPUT_ITEM_FIELDS & it.keys():
            it = {k: v for k, v in it.items() if k not in UNSUPPORTED_INPUT_ITEM_FIELDS}
            modified += 1
        new_items.append(it)
    if modified:
        new_body["input"] = new_items
    return new_body, modified


# stream_options values newer clients send but Azure's Responses API rejects
# with an `invalid_value` 400. The ChatGPT desktop app (July 2026) sends
# reasoning_summary_delivery="sequential_cutoff"; Azure only accepts
# sequential/concurrent/concurrent_cutoff. Dropping the key falls back to
# Azure's default delivery, which every client build also understands.
UNSUPPORTED_STREAM_OPTION_VALUES: dict[str, frozenset[str]] = {
    "reasoning_summary_delivery": frozenset({"sequential_cutoff"}),
}


def sanitize_stream_options(body: dict) -> tuple[dict, list[str]]:
    """Return a copy of `body` with Azure-rejected stream_options keys removed.

    Only keys whose value is in UNSUPPORTED_STREAM_OPTION_VALUES are dropped;
    supported values pass through untouched. Returns the (possibly rewritten)
    body and the list of removed keys.
    """
    opts = body.get("stream_options")
    if not isinstance(opts, dict):
        return body, []
    removed = [
        k for k, bad in UNSUPPORTED_STREAM_OPTION_VALUES.items()
        if opts.get(k) in bad
    ]
    if not removed:
        return body, []
    new_opts = {k: v for k, v in opts.items() if k not in removed}
    new_body = dict(body)
    if new_opts:
        new_body["stream_options"] = new_opts
    else:
        new_body.pop("stream_options", None)
    return new_body, removed


COMPACT_SYSTEM_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. "
    "Create a handoff summary for another LLM that will resume the task. "
    "Include: Current progress and key decisions made, "
    "Important context/constraints/user preferences, "
    "What remains to be done, "
    "Any critical data/examples/references needed to continue. "
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
)
