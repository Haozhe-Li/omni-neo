"""Cross-turn citation persistence, keyed by thread_id.

Citations used to live only in the per-run contextvar (see `citations.py`),
which meant a source surfaced two turns ago was invisible to both the model
(prompted not to reuse old `[n]`s) and the frontend (nothing to look it up
by). This module persists every citation a thread has ever produced to Redis,
verbatim, so numbering survives across turns.

Semantic lookup for `/check_source` (chunking + similarity search) now lives
in `vector_sources.py` (Upstash Search) — this module is pure storage of the
full, unchunked source record.
"""
from __future__ import annotations

import json

from upstash_redis import Redis

# 90 days matches the longest thread retention window (logged-in users, see
# db_threads_control.py). Explicit deletion on thread delete handles the rest;
# this TTL is only a safety net against orphaned keys.
_SOURCE_TTL = 3600 * 24 * 90

_client: Redis | None = None


def _get_redis() -> Redis:
    global _client
    if _client is None:
        # HTTP REST client (UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN).
        _client = Redis.from_env()
    return _client


def _citation_key(thread_id: str) -> str:
    return f"omni:citation:{thread_id}"


def _index_key(thread_id: str) -> str:
    return f"omni:citation:{thread_id}:index"


def load_citations(thread_id: str) -> list[dict]:
    """Load every citation this thread has ever produced, oldest first."""
    r = _get_redis()
    raw = r.lrange(_citation_key(thread_id), 0, -1)
    out = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except Exception:
            continue
    return out


def persist_citation(thread_id: str, record: dict) -> None:
    """Append a newly-numbered citation to Redis, verbatim (no chunking).

    `record` must already carry the `n` assigned by the in-memory registry
    (see `citations.register_citation`) — this call only persists it, it
    doesn't decide the number.
    """
    r = _get_redis()

    pipe = r.pipeline()
    pipe.rpush(_citation_key(thread_id), json.dumps(record, ensure_ascii=False))
    pipe.expire(_citation_key(thread_id), _SOURCE_TTL)
    if record.get("url"):
        pipe.hset(_index_key(thread_id), record["url"], record["n"])
        pipe.expire(_index_key(thread_id), _SOURCE_TTL)
    pipe.exec()


def delete_thread_sources(thread_id: str) -> None:
    """Remove all persisted citations for a hard-deleted thread."""
    r = _get_redis()
    r.delete(_citation_key(thread_id), _index_key(thread_id))


def delete_threads_sources_bulk(thread_ids: list[str]) -> None:
    for thread_id in thread_ids:
        delete_thread_sources(thread_id)
