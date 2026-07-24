"""Cross-turn widget dedup, keyed by thread_id.

`widget_predictor.predict_widgets` runs per-query with no notion of what a
thread has already shown, so the same widget (e.g. the Tokyo weather card, or
the AAPL quote) could be re-emitted on a later turn that happens to trigger
the same classification. This module persists the set of widget dedup keys a
thread has already surfaced, so `predict_widgets` can skip re-fetching and
re-emitting them.

Mirrors `redis_sources.py`'s storage pattern (same TTL, same thread-delete
hooks) but uses a Redis SET instead of a list — we only need membership, not
order or the full record.
"""
from __future__ import annotations

from upstash_redis import Redis
from upstash_redis.asyncio import Redis as AsyncRedis

# Matches the citation TTL (redis_sources.py) — the longest thread retention
# window. Explicit deletion on thread delete handles the rest; this TTL is
# only a safety net against orphaned keys.
_WIDGET_TTL = 3600 * 24 * 90

_client: Redis | None = None
_async_client: AsyncRedis | None = None


def _get_redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_env()
    return _client


def _get_async_redis() -> AsyncRedis:
    global _async_client
    if _async_client is None:
        _async_client = AsyncRedis.from_env()
    return _async_client


def _shown_key(thread_id: str) -> str:
    return f"omni:widget:{thread_id}"


async def load_shown_async(thread_id: str) -> set[str]:
    """Return every widget dedup key this thread has already shown."""
    r = _get_async_redis()
    return set(await r.smembers(_shown_key(thread_id)))


def mark_shown(thread_id: str, keys: list[str]) -> None:
    """Record `keys` as shown for this thread (sync — for worker threads)."""
    if not keys:
        return
    r = _get_redis()
    pipe = r.pipeline()
    pipe.sadd(_shown_key(thread_id), *keys)
    pipe.expire(_shown_key(thread_id), _WIDGET_TTL)
    pipe.exec()


async def mark_shown_async(thread_id: str, keys: list[str]) -> None:
    """Record `keys` as shown for this thread (async — hot chat path)."""
    if not keys:
        return
    r = _get_async_redis()
    pipe = r.pipeline()
    pipe.sadd(_shown_key(thread_id), *keys)
    pipe.expire(_shown_key(thread_id), _WIDGET_TTL)
    await pipe.exec()


def delete_thread_widgets(thread_id: str) -> None:
    """Remove the persisted widget dedup set for a hard-deleted thread."""
    r = _get_redis()
    r.delete(_shown_key(thread_id))


def delete_threads_widgets_bulk(thread_ids: list[str]) -> None:
    for thread_id in thread_ids:
        delete_thread_widgets(thread_id)
