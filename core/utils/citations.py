"""Per-thread citation numbering, shared across every tool call in one agent run.

Scoped with contextvars (not a module-level dict) so concurrent `/chat`
requests don't share state. `register_citation` is called from inside the
retrieval tools themselves — before their result reaches the model — so the
model sees the same numbers it's expected to cite with `[n]`.

Numbering is persisted per-thread in Redis (see `redis_sources.py`), not reset
every run: `reset_citation_registry` hydrates the in-memory registry from
Redis at the top of each turn, so a source fetched two turns ago keeps its
`[n]` and the model can cite it again without re-fetching. Threads used
outside a persisted thread (thread_id=None, e.g. the direct-stream fallback
in main.py) keep the old run-scoped-only behavior.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar

from core.utils import redis_sources

_registry: ContextVar[list[dict] | None] = ContextVar("citation_registry", default=None)
_thread_id: ContextVar[str | None] = ContextVar("citation_thread_id", default=None)
# Coarse global lock: guards only the in-memory list dedupe/append below, not
# the Redis persist call (that happens after the lock is released) — so a
# network round-trip never serializes concurrent chats against each other.
_lock = threading.Lock()


def reset_citation_registry(thread_id: str | None = None) -> None:
    """Start the registry for a new turn.

    If `thread_id` is given, hydrates from every citation this thread has
    ever produced (so numbering and de-dupe carry across turns); otherwise
    starts empty, matching the old run-scoped-only behavior.
    """
    _thread_id.set(thread_id)
    if thread_id:
        try:
            _registry.set(redis_sources.load_citations(thread_id))
            return
        except Exception:
            # Redis hiccup shouldn't break the chat — fall back to a fresh,
            # run-scoped registry for this turn.
            pass
    _registry.set([])


def register_citation(title: str, url: str, content: str) -> int | None:
    """Assign (or reuse) the 1-based citation number for `url` in this thread.

    Returns None if `url` is empty — there's nothing for the frontend to link
    to, so no number is worth giving the model.
    """
    if not url:
        return None
    reg = _registry.get()
    if reg is None:
        # Defensive: register_citation called without a preceding
        # reset_citation_registry (e.g. a tool invoked outside _stream_agent).
        reg = []
        _registry.set(reg)

    is_new = False
    with _lock:
        for item in reg:
            if item["url"] == url:
                n = item["n"]
                break
        else:
            n = len(reg) + 1
            record = {"n": n, "title": title, "url": url, "content": content}
            reg.append(record)
            is_new = True

    if is_new:
        thread_id = _thread_id.get()
        if thread_id:
            try:
                redis_sources.persist_citation(thread_id, record)
            except Exception:
                # Persistence is best-effort: the model still gets its number
                # for this run even if Redis is briefly unavailable.
                pass
    return n


def all_citations() -> list[dict]:
    return list(_registry.get() or [])
