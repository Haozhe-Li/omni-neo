"""The `{"type": "error", ...}` SSE event contract.

Every stream-terminating error the backend sends to the frontend — a
safety cutoff, an unhandled exception mid-generation, an empty result —
must go through `error_payload()` below rather than being hand-assembled.
Two things this buys:

1. The client never sees a raw exception string. Before this module
   existed, both `core/stream.py` and `core/routers/chat.py` sent
   `{"type": "error", "content": str(exc)}` straight from a bare `except`.
   `str(exc)` can carry DB connection details, internal hostnames, stack
   fragments — or, per the prompt-leakage guard this module was added
   alongside, a chunk of the system prompt the guard just caught. Callers
   pass a `code` from the closed `ErrorCode` enum plus an optional
   override `message`; the real exception/detail goes to `logger.error`
   only, tagged with a `request_id` the client also receives so a user
   report can be matched back to the log line without the log line's
   contents ever reaching the client.

2. A stable, closed vocabulary the frontend switches on to decide how to
   render the failure (retryable toast vs. a terminal "conversation
   ended" state) instead of pattern-matching on message text, which is
   copy and can change without notice.

`SAFETY_TERMINATED` deliberately covers BOTH "query flagged as harmful
before generation started" and "answer caught leaking the system prompt
mid-stream" — giving each its own wire-visible code would hand an
attacker doing automated probing a free oracle for which category their
attempt hit (a leak-specific code tells them "your injection worked,
just got caught after the fact" versus "didn't even get through input
screening" — exactly the signal that lets probing converge). The client
renders one "this conversation was ended" state either way; the specific
cause is only in the server log, keyed by `request_id`.

This module intentionally does NOT cover the existing `HTTPException`
paths (403 thread access, 404 rewind target, 429 usage limit in
`core/routers/chat.py`'s `_usage_limit_detail`) — those are ordinary REST
responses with their own already-structured bodies on a different
channel (HTTP status + JSON body on the initial POST, not a mid-stream
SSE event), not `error` events over an open stream.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Closed vocabulary for the `error` SSE event's `code` field.

    The frontend must switch on `code`, never on `message` — message is
    display copy and can change without notice. Add new members here (and
    to `_MESSAGES` below) rather than inventing an ad hoc string at a call
    site; an ungoverned code defeats the point of a closed contract.
    """

    # A turn was cut short for a safety reason: the query was flagged
    # before generation started, or the model's own output tripped the
    # prompt-leakage guard mid-stream. Deliberately undifferentiated to
    # the client — see module docstring.
    SAFETY_TERMINATED = "safety_terminated"
    # An unhandled exception during generation. Retryable.
    GENERATION_FAILED = "generation_failed"
    # The agent's turn ended with nothing to show (no text, no artifact) —
    # distinct from an exception; nothing *broke*, there's just no output.
    NO_OUTPUT = "no_output"


_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.SAFETY_TERMINATED: "This conversation was ended for safety reasons.",
    ErrorCode.GENERATION_FAILED: "Something went wrong while generating a response. Please try again.",
    ErrorCode.NO_OUTPUT: "The assistant didn't produce a response. Please try again.",
}


def error_payload(
    code: ErrorCode,
    *,
    detail: str | None = None,
    message: str | None = None,
) -> dict:
    """Build the dict for an `{"type": "error", ...}` SSE event.

    Args:
        code: the closed-vocabulary reason, drives frontend rendering.
        detail: backend-internal context (exception text, guard trigger
            reason) — logged server-side only, NEVER included in the
            returned dict. This is the one parameter every call site must
            get right: pass the exception/detail here, not into `message`.
        message: override for the default copy in `_MESSAGES`. Rarely
            needed — prefer adding a new `ErrorCode` over ad hoc copy.

    Returns:
        ``{"type": "error", "code": ..., "message": ..., "request_id": ...}``.
        Callers serialize this the same way as any other SSE payload (see
        `core/stream.py`'s `_sse()`).
    """
    request_id = uuid.uuid4().hex[:12]
    if detail:
        logger.error("[error:%s] code=%s detail=%s", request_id, code.value, detail)
    else:
        logger.error("[error:%s] code=%s", request_id, code.value)
    return {
        "type": "error",
        "code": code.value,
        "message": message or _MESSAGES[code],
        "request_id": request_id,
    }
