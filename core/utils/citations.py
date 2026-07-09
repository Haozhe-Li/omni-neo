"""Per-turn citation numbering, shared across every tool call in one agent run.

Scoped with a contextvar (not a module-level dict) so concurrent `/chat`
requests don't share state. `register_citation` is called from inside the
retrieval tools themselves — before their result reaches the model — so the
model sees the same numbers it's expected to cite with `[n]`.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar

_registry: ContextVar[list[dict] | None] = ContextVar("citation_registry", default=None)
# Coarse global lock: registration is a few list operations, never a
# bottleneck, and a per-registry lock would be more bookkeeping for no
# practical benefit.
_lock = threading.Lock()


def reset_citation_registry() -> None:
    """Start a fresh, empty registry. Call once at the top of each agent run."""
    _registry.set([])


def register_citation(title: str, url: str, content: str) -> int | None:
    """Assign (or reuse) the 1-based citation number for `url` this turn.

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
    with _lock:
        for item in reg:
            if item["url"] == url:
                return item["n"]
        n = len(reg) + 1
        reg.append({"n": n, "title": title, "url": url, "content": content})
        return n


def all_citations() -> list[dict]:
    return list(_registry.get() or [])
