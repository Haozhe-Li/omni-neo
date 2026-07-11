"""Chunked semantic index of every citation a thread has produced, backing
`/check_source`.

Indexing is fire-and-forget: `enqueue_source_indexing` is called from inside
`citations.register_citation`, which itself runs synchronously deep inside an
agent tool call — it must never block the token stream, so the actual chunk
+ upsert work runs on a small background thread pool and the caller never
waits on it.

Documents live in a single Upstash Search index shared by every thread;
isolation between threads is enforced two ways: the document id is prefixed
with `{thread_id}:`, and every document carries `thread_id` in its `content`
(Upstash Search's `filter` only matches `content` fields, not `metadata` —
confirmed empirically, undocumented in the SDK) so a query can filter to
just this thread's sources. Chunk volume per thread
is small (a couple hundred at most), so the `turn` cutoff (see
`citations.py`) is applied client-side in `search_similar_chunks` rather than
pushed into the Upstash filter — one extra Python pass over ~100 rows is
cheaper than getting an untested filter expression wrong.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter
from upstash_search import AsyncSearch, Search

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 256
_CHUNK_OVERLAP = 32
_INDEX_NAME = "omni_chunks"
_MIN_SCORE = 0.6
_QUERY_LIMIT = 100

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vector-index")

_sync_client: Search | None = None
_async_client: AsyncSearch | None = None


def _get_sync_client() -> Search:
    global _sync_client
    if _sync_client is None:
        _sync_client = Search.from_env()
    return _sync_client


def _get_async_client() -> AsyncSearch:
    global _async_client
    if _async_client is None:
        _async_client = AsyncSearch.from_env()
    return _async_client


def _doc_id(thread_id: str, n: int, chunk_index: int) -> str:
    return f"{thread_id}:{n}:{chunk_index}"


def _escape_filter_value(value: str) -> str:
    return value.replace("'", "''")


def _index_source_sync(thread_id: str, record: dict) -> None:
    content = (record.get("content") or "").strip()
    if not content:
        return
    n = record["n"]
    chunks = _splitter.split_text(content)
    docs = [
        {
            "id": _doc_id(thread_id, n, i),
            # Upstash Search's `filter` only matches fields inside `content`
            # (confirmed empirically — a filter on a `metadata`-only field
            # always returns zero hits, undocumented in the SDK). thread_id
            # HAS to live here for query-time isolation to work at all.
            "content": {"text": chunk_text, "thread_id": thread_id},
            "metadata": {
                "n": n,
                "url": record.get("url", ""),
                "title": record.get("title", ""),
                "chunk_index": i,
                "turn": record.get("turn"),
                "credibility": record.get("credibility"),
            },
        }
        for i, chunk_text in enumerate(chunks)
    ]
    if not docs:
        return
    _get_sync_client().index(_INDEX_NAME).upsert(docs)
    print(
        f"[vector_sources] indexed {len(docs)} chunk(s) for thread={thread_id} "
        f"n={n} turn={record.get('turn')} title={record.get('title', '')!r}"
    )


def enqueue_source_indexing(thread_id: str, record: dict) -> None:
    """Fire-and-forget: chunk `record["content"]` and upsert to Upstash Search.

    Never raises — indexing is best-effort and must not affect the calling
    tool or the token stream.
    """
    def _run() -> None:
        try:
            _index_source_sync(thread_id, record)
        except Exception as exc:
            logger.warning(f"[vector_sources] indexing failed for thread {thread_id}: {exc}")

    _executor.submit(_run)


async def search_similar_chunks(
    thread_id: str, text_selection: str, turn: int | None = None
) -> list[dict]:
    """Return every chunk above `_MIN_SCORE` similarity for this thread,
    restricted to sources introduced at or before `turn` (None = no cutoff).

    Sorted by score, descending.
    """
    tag = _escape_filter_value(thread_id)
    try:
        results = await _get_async_client().index(_INDEX_NAME).search(
            text_selection,
            limit=_QUERY_LIMIT,
            filter=f"thread_id = '{tag}'",
            semantic_weight=1.0,
        )
    except Exception as exc:
        logger.warning(f"[vector_sources] search failed for thread {thread_id}: {exc}")
        return []

    print(
        f"[vector_sources] search thread={thread_id} turn_cutoff={turn} "
        f"query={text_selection[:80]!r} -> {len(results)} raw candidate(s)"
    )

    matches = []
    for doc in results:
        meta = doc.metadata or {}
        doc_turn = meta.get("turn")
        kept = doc.score > _MIN_SCORE and (turn is None or doc_turn is None or doc_turn <= turn)
        print(
            f"[vector_sources]   score={doc.score:.4f} turn={doc_turn} n={meta.get('n')} "
            f"kept={kept} text={doc.content.get('text', '')[:60]!r}"
        )
        if not kept:
            continue
        matches.append(
            {
                "n": meta.get("n"),
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "chunk": doc.content.get("text", ""),
                "score": round(float(doc.score), 4),
                "turn": doc_turn,
                "credibility": meta.get("credibility"),
            }
        )
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches


def delete_thread_vectors(thread_id: str) -> None:
    """Remove every indexed chunk for a hard-deleted thread."""
    _get_sync_client().index(_INDEX_NAME).delete(prefix=f"{thread_id}:")
