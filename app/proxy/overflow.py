"""A turn that did not fit, named the same way on every route.

Every upstream says "too big" in its own words. OpenAI and the Codex backend
set `code: context_length_exceeded`; Anthropic answers `invalid_request_error`
with "prompt is too long: N tokens > M maximum", or ends a served turn with
`stop_reason: model_context_window_exceeded`; Ollama's cloud models send an
OpenAI envelope whose `code` is null and whose only signal is the sentence
("The prompt is too long: 131621, model maximum context length: 131072").

A caller that has to rotate a conversation on overflow cannot read four
dialects, and one that is handed a bare "Azure API error: 400" or a `length`
finish cannot tell overflow from anything else at all. So every route hands
the same shape back: an OpenAI error whose `code` (and, on the Responses
wire's failure event, `error.code`) is `context_length_exceeded`, carrying the
upstream's own sentence so the numbers in it survive.
"""

from __future__ import annotations

import json
from typing import Optional

OVERFLOW_CODE = "context_length_exceeded"

#: Codes and types an upstream uses for this failure and nothing else.
_OVERFLOW_CODES = frozenset(
    {"context_length_exceeded", "model_context_window_exceeded", "context_window_exceeded"}
)

#: The sentences that say it when no code does. Each names the context or the
#: prompt's length explicitly, so a rate limit ("would exceed your account's
#: rate limit") or a usage cap is never read as an overflow.
_OVERFLOW_PHRASES = (
    "prompt is too long",
    "context length",
    "context window",
    "context limit",
    "exceeds the context",
)


def _as_payload(raw: object) -> object:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return raw
    return raw


def _error_object(payload: object) -> object:
    """The part of a refusal that describes the failure, in any envelope."""
    if not isinstance(payload, dict):
        return payload
    # Responses-wire failure event: {"type": "response.failed", "response": {"error": {...}}}
    response = payload.get("response")
    if isinstance(response, dict) and response.get("error") is not None:
        return response["error"]
    if payload.get("error") is not None:
        return payload["error"]
    return payload


def overflow_message(raw: object) -> Optional[str]:
    """The upstream's own sentence if `raw` reports a context overflow, else None.

    `raw` may be a response body (bytes or text), a parsed envelope, or a
    stream event — every shape the routes see a refusal in.
    """
    error = _error_object(_as_payload(raw))
    if isinstance(error, str):
        message, codes = error, ()
    elif isinstance(error, dict):
        message = error.get("message")
        codes = (error.get("code"), error.get("type"))
    else:
        return None
    if any(isinstance(c, str) and c.lower() in _OVERFLOW_CODES for c in codes):
        return message if isinstance(message, str) and message else OVERFLOW_CODE
    if isinstance(message, str):
        lowered = message.lower()
        if any(phrase in lowered for phrase in _OVERFLOW_PHRASES):
            return message
    return None


def overflow_error(message: str) -> dict:
    """The OpenAI error object a caller reads an overflow out of."""
    return {
        "message": message,
        "type": "invalid_request_error",
        "param": None,
        "code": OVERFLOW_CODE,
    }


def overflow_body(message: str) -> dict:
    """An overflow as a whole OpenAI error response body."""
    return {"error": overflow_error(message)}


def overflow_failed_event(message: str) -> dict:
    """An overflow as the Responses wire's own failure event.

    This is the shape the Codex CLI recognises as a full context window
    (`response.failed` with `error.code == "context_length_exceeded"`), so a
    Responses caller reads it the way it would read the Codex backend's own.
    """
    return {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {"code": OVERFLOW_CODE, "message": message},
        },
    }
