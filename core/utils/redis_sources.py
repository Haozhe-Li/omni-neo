"""Cross-turn citation persistence, keyed by thread_id, backing `/check_source`.

Citations used to live only in the per-run contextvar (see `citations.py`),
which meant a source surfaced two turns ago was invisible to both the model
(prompted not to reuse old `[n]`s) and the frontend (nothing to look it up
by). This module persists every citation a thread has ever produced to Redis
so numbering survives across turns, and additionally splits long page
content into overlapping chunks so `/check_source` can point the frontend at
the specific passage a claim came from instead of a whole page.

Search is RediSearch (`FT.SEARCH`) when the Redis instance supports it. Not
every managed Redis plan ships that module, so `setup_source_search_index`
probes for it at startup and `search_chunks` transparently falls back to a
SCAN + rapidfuzz scan over the same chunk hashes when it isn't available —
callers never need to know which path served a given request.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import redis
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100
# 90 days matches the longest thread retention window (logged-in users, see
# db_threads_control.py). Explicit deletion on thread delete handles the rest;
# this TTL is only a safety net against orphaned keys.
_SOURCE_TTL = 3600 * 24 * 90

_INDEX_NAME = "idx:omni_chunks"
_CHUNK_PREFIX = "omni:chunk:"

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
)

_client: redis.Redis | None = None
_ft_available = False

_MIN_FALLBACK_SCORE = 60.0


def _get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _client


def _citation_key(thread_id: str) -> str:
    return f"omni:citation:{thread_id}"


def _index_key(thread_id: str) -> str:
    return f"omni:citation:{thread_id}:index"


def _chunk_key(thread_id: str, n: int, chunk_index: int) -> str:
    return f"{_CHUNK_PREFIX}{thread_id}:{n}:{chunk_index}"


def _chunk_pattern(thread_id: str) -> str:
    return f"{_CHUNK_PREFIX}{thread_id}:*"


# RediSearch special characters that need escaping inside a TAG/TEXT query.
_QUERY_SPECIAL_RE = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\ ])")


def _escape_query_token(value: str) -> str:
    return _QUERY_SPECIAL_RE.sub(r"\\\1", value)


def setup_source_search_index() -> None:
    """Idempotently create the RediSearch index over source chunks.

    Safe to call on every startup. If the Redis instance doesn't support the
    module at all, this logs a warning and leaves `_ft_available` False —
    `search_chunks` then always uses the SCAN + fuzzy-match fallback.
    """
    global _ft_available
    r = _get_redis()
    try:
        r.ft(_INDEX_NAME).info()
        _ft_available = True
        return
    except Exception:
        pass

    try:
        from redis.commands.search.field import TagField, TextField, NumericField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType

        r.ft(_INDEX_NAME).create_index(
            fields=[
                TagField("thread_id"),
                TextField("text"),
                TextField("title"),
                TextField("url"),
                NumericField("n"),
            ],
            definition=IndexDefinition(
                prefix=[_CHUNK_PREFIX], index_type=IndexType.HASH
            ),
        )
        _ft_available = True
    except Exception as exc:
        logger.warning(
            f"[redis_sources] RediSearch unavailable, falling back to SCAN+fuzzy search: {exc}"
        )
        _ft_available = False


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
    """Append a newly-numbered citation and its search chunks to Redis.

    `record` must already carry the `n` assigned by the in-memory registry
    (see `citations.register_citation`) — this call only persists it, it
    doesn't decide the number.
    """
    r = _get_redis()

    pipe = r.pipeline(transaction=False)
    pipe.rpush(_citation_key(thread_id), json.dumps(record, ensure_ascii=False))
    pipe.expire(_citation_key(thread_id), _SOURCE_TTL)
    if record.get("url"):
        pipe.hset(_index_key(thread_id), record["url"], record["n"])
        pipe.expire(_index_key(thread_id), _SOURCE_TTL)
    pipe.execute()

    content = (record.get("content") or "").strip()
    if not content:
        return

    chunks = _splitter.split_text(content)
    pipe = r.pipeline(transaction=False)
    for i, chunk_text in enumerate(chunks):
        key = _chunk_key(thread_id, record["n"], i)
        pipe.hset(
            key,
            mapping={
                "thread_id": thread_id,
                "n": record["n"],
                "url": record.get("url", ""),
                "title": record.get("title", ""),
                "chunk_index": i,
                "text": chunk_text,
            },
        )
        pipe.expire(key, _SOURCE_TTL)
    pipe.execute()


def delete_thread_sources(thread_id: str) -> None:
    """Remove all persisted citations + chunks for a hard-deleted thread."""
    r = _get_redis()
    keys = [_citation_key(thread_id), _index_key(thread_id)]
    cursor = 0
    pattern = _chunk_pattern(thread_id)
    while True:
        cursor, found = r.scan(cursor=cursor, match=pattern, count=200)
        keys.extend(found)
        if cursor == 0:
            break
    if keys:
        r.delete(*keys)


def delete_threads_sources_bulk(thread_ids: list[str]) -> None:
    for thread_id in thread_ids:
        delete_thread_sources(thread_id)


def _fallback_search(thread_id: str, text_selection: str) -> dict[str, Any] | None:
    """SCAN + rapidfuzz fuzzy match — used when RediSearch isn't available,
    or when a FT.SEARCH query fails for any reason (e.g. an escaping edge
    case), so a single query-string bug never turns into an outage."""
    from rapidfuzz import fuzz

    r = _get_redis()
    pattern = _chunk_pattern(thread_id)
    best: dict[str, str] | None = None
    best_score = -1.0
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
        for key in keys:
            data = r.hgetall(key)
            if not data:
                continue
            score = fuzz.partial_ratio(text_selection, data.get("text", ""))
            if score > best_score:
                best_score = score
                best = data
        if cursor == 0:
            break

    if best is None or best_score < _MIN_FALLBACK_SCORE:
        return None
    return {
        "n": int(best["n"]),
        "title": best.get("title", ""),
        "url": best.get("url", ""),
        "chunk": best.get("text", ""),
        "score": round(best_score, 2),
    }


def search_chunks(thread_id: str, text_selection: str) -> dict[str, Any] | None:
    """Return the single best-matching chunk for `text_selection` among this
    thread's persisted sources, or None if nothing scores high enough."""
    if not _ft_available:
        return _fallback_search(thread_id, text_selection)

    try:
        from redis.commands.search.query import Query

        tag = _escape_query_token(thread_id)
        terms = _escape_query_token(text_selection)
        query_str = f"@thread_id:{{{tag}}} @text:({terms})"
        q = Query(query_str).with_scores().paging(0, 1)
        res = _get_redis().ft(_INDEX_NAME).search(q)
        if not res.docs:
            return None
        doc = res.docs[0]
        return {
            "n": int(getattr(doc, "n")),
            "title": getattr(doc, "title", ""),
            "url": getattr(doc, "url", ""),
            "chunk": getattr(doc, "text", ""),
            "score": round(float(getattr(doc, "score", 0.0)), 4),
        }
    except Exception as exc:
        logger.warning(f"[redis_sources] FT.SEARCH failed, falling back: {exc}")
        return _fallback_search(thread_id, text_selection)
