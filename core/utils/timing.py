"""Lightweight structured-timing logs for the non-LLM request path.

LangSmith traces start at `agent.astream`, so everything before it — the charge
gate, ownership check, stream setup, citation load, and the time it takes the
first event to reach the buffer — is invisible there. This module emits one
JSON line per instrumented span to stdout, which Cloud Run forwards to Cloud
Logging and parses into `jsonPayload.*`. From there you can build log-based
distribution metrics (p50/p95/p99) and alerts on any field.

Usage:
    t = Timing("chat_prelude", thread_id=tid, mode=mode)
    with t.stage("stream_begin"):
        await stream_begin(tid)
    charge, owner = await asyncio.gather(t.atimed("charge", ...), t.atimed("owner", ...))
    t.emit(outcome="started")   # -> {"event":"chat_prelude","total_ms":..,"stream_begin_ms":..,..}

Each numeric stage is emitted as ``<name>_ms``; string/bool fields (cache
hits, outcome) pass through as-is. Emit exactly once per Timing.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager

# Dedicated logger: a stdout handler that emits ONLY the message (the JSON
# object), with propagation off so the app's root formatter doesn't wrap it —
# Cloud Logging only parses a line into jsonPayload when the whole line is JSON.
_logger = logging.getLogger("omni.timing")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False


class Timing:
    def __init__(self, event: str, **fields):
        self.event = event
        self.fields: dict = dict(fields)
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    def _ms_since_start(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def set(self, **fields) -> "Timing":
        """Attach non-timing fields (cache hits, outcome, counts, …)."""
        self.fields.update(fields)
        return self

    def record(self, name: str, ms: float) -> None:
        """Record a stage duration measured elsewhere (milliseconds)."""
        self.stages[name] = round(ms, 1)

    def mark(self, name: str) -> None:
        """Record elapsed-from-start at this point, as ``<name>_ms``."""
        self.stages[name] = self._ms_since_start()

    @contextmanager
    def stage(self, name: str):
        """Time a synchronous block into ``<name>_ms``."""
        s = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - s) * 1000, 1)

    async def atimed(self, name: str, coro):
        """Time a coroutine into ``<name>_ms`` (safe inside asyncio.gather)."""
        s = time.perf_counter()
        try:
            return await coro
        finally:
            self.stages[name] = round((time.perf_counter() - s) * 1000, 1)

    def emit(self, **fields) -> None:
        """Write the single structured log line for this span."""
        self.fields.update(fields)
        payload = {
            "severity": "INFO",
            "event": self.event,
            "total_ms": self._ms_since_start(),
        }
        payload.update({f"{k}_ms": v for k, v in self.stages.items()})
        payload.update(self.fields)
        try:
            _logger.info(json.dumps(payload, default=str))
        except Exception:
            _logger.info(str(payload))
