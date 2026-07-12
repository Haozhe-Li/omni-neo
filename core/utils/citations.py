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

Every new citation also carries the `turn` it was first introduced in (given
by the caller, ultimately the frontend — see `reset_citation_registry`), so
`/check_source` can later confine a claim from turn N to sources visible by
turn N. Reused citations (same dedup key cited/mounted again in a later turn)
keep their original `turn`, which is exactly the semantics we want: a source
is valid for every turn from the one it first appeared in onward.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

from core.utils import redis_sources, vector_sources

# redis_sources.persist_citation is a synchronous (blocking) Redis call, but
# _register runs inline inside `google_search`/`load_web_page` — both `async
# def` tools that the agent awaits directly on the event loop, unlike sync
# tools, which LangChain itself already dispatches to a worker thread. A
# blocking call here would freeze that request's event loop for its
# duration, stalling every other concurrent thread's SSE stream (writes/reads
# in redis_stream.py) on the same loop — exactly the same class of bug
# vector_sources.enqueue_source_indexing already works around below with its
# own executor. Same fix, same reasoning.
_persist_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="citation-persist")

_registry: ContextVar[list[dict] | None] = ContextVar("citation_registry", default=None)
_thread_id: ContextVar[str | None] = ContextVar("citation_thread_id", default=None)
_turn: ContextVar[int | None] = ContextVar("citation_turn", default=None)
# Coarse global lock: guards only the in-memory list dedupe/append below, not
# the Redis persist call (that happens after the lock is released) — so a
# network round-trip never serializes concurrent chats against each other.
_lock = threading.Lock()


def reset_citation_registry(thread_id: str | None = None, turn: int | None = None) -> None:
    """Start the registry for a new turn.

    If `thread_id` is given, hydrates from every citation this thread has
    ever produced (so numbering and de-dupe carry across turns); otherwise
    starts empty, matching the old run-scoped-only behavior. `turn` is
    stamped onto any newly-registered citation this run.
    """
    _thread_id.set(thread_id)
    _turn.set(turn)
    if thread_id:
        try:
            _registry.set(redis_sources.load_citations(thread_id))
            return
        except Exception:
            # Redis hiccup shouldn't break the chat — fall back to a fresh,
            # run-scoped registry for this turn.
            pass
    _registry.set([])


def _register(
    key_field: str,
    key_value: str,
    title: str,
    url: str,
    content: str,
    index_content: str | None = None,
    credibility: dict | None = None,
) -> int:
    """Shared implementation behind `register_citation` and
    `register_document_citation`. Dedupes on `record[key_field] == key_value`
    — `url` for web sources, `file_id` for uploaded documents (which all
    share `url == ""`, so deduping those on `url` would collapse every
    document into one entry).

    `index_content` lets the vector index get a different (typically longer)
    slice of the source than what's persisted to Redis / shown to the
    frontend — used for documents, where the citation's display `content` is
    capped much shorter than what's worth chunking for search.

    `credibility` is the `{"label", "reason"}` dict from
    `source_credibility.classify_sources` (None for uploaded documents, which
    aren't classified). Like `turn`, a reused citation keeps whatever
    credibility it was first registered with — it isn't recomputed on repeat
    cites.
    """
    reg = _registry.get()
    if reg is None:
        # Defensive: called without a preceding reset_citation_registry
        # (e.g. a tool invoked outside _stream_agent).
        reg = []
        _registry.set(reg)

    is_new = False
    with _lock:
        for item in reg:
            if item.get(key_field) == key_value:
                n = item["n"]
                break
        else:
            n = len(reg) + 1
            record = {
                "n": n,
                "title": title,
                "url": url,
                "content": content,
                "turn": _turn.get(),
                "credibility": credibility,
            }
            if key_field != "url":
                record[key_field] = key_value
            reg.append(record)
            is_new = True

    if is_new:
        thread_id = _thread_id.get()
        if thread_id:
            def _persist(thread_id=thread_id, record=record) -> None:
                try:
                    redis_sources.persist_citation(thread_id, record)
                except Exception:
                    # Persistence is best-effort: the model still gets its number
                    # for this run even if Redis is briefly unavailable.
                    pass
            _persist_executor.submit(_persist)
            # Junk is never indexed: the agent never sees junk content (see
            # google_search/load_web_page), so no claim in the answer can
            # legitimately be "supported by" it — indexing it anyway would
            # just be wasted storage, and worst case lets `/check_source`
            # spuriously match a claim against junk text that merely looks
            # similar, letting a low-quality source get presented as backing
            # a claim it never actually influenced.
            if (credibility.get("label") if credibility else None) != "junk":
                try:
                    index_record = (
                        record if index_content is None else {**record, "content": index_content}
                    )
                    vector_sources.enqueue_source_indexing(thread_id, index_record)
                except Exception:
                    # Same best-effort contract: indexing must never break citing.
                    pass
    return n


def register_citation(
    title: str, url: str, content: str, credibility: dict | None = None
) -> int | None:
    """Assign (or reuse) the 1-based citation number for `url` in this thread.

    Returns None if `url` is empty — there's nothing for the frontend to link
    to, so no number is worth giving the model.
    """
    if not url:
        return None
    return _register("url", url, title, url, content, credibility=credibility)


def register_document_citation(
    title: str, file_id: str, content: str, index_content: str | None = None
) -> int:
    """Assign (or reuse) the citation number for an uploaded document.

    Deduped by `file_id` (not `url` — documents don't have one; the frontend
    treats `url == ""` as "uploaded document, not a link"). `content` is
    what's persisted to Redis and shown to the user; pass `index_content` to
    chunk a longer slice into the vector index than what's displayed.
    """
    return _register("file_id", file_id, title, "", content, index_content)


def all_citations() -> list[dict]:
    return list(_registry.get() or [])
