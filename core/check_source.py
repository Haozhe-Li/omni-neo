from __future__ import annotations

import hashlib
import json
import logging

from core.utils import source_rerank, vector_sources
from core.utils.redis_cache import l1cache

logger = logging.getLogger(__name__)

_CHECK_SOURCE_CACHE_TTL = 60 * 10


def _build_check_source_cache_key(
    thread_id: str, text_selection: str, turn: int | None
) -> str:
    payload = json.dumps(
        {
            "thread_id": thread_id,
            "text_selection": text_selection,
            "turn": turn,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{l1cache.prefix}check_source:{hashlib.md5(payload.encode('utf-8')).hexdigest()}"


async def check_source_matches(
    thread_id: str, text_selection: str, turn: int | None = None
) -> dict:
    text_selection = text_selection.strip()
    if not text_selection:
        return {"error": "Text selection is empty"}

    cache_key = _build_check_source_cache_key(thread_id, text_selection, turn)
    cached = l1cache.redis.get(cache_key)
    if cached is not None:
        try:
            return {"matches": json.loads(cached)}
        except json.JSONDecodeError:
            l1cache.redis.delete(cache_key)
            logger.warning("[check_source] invalid cache payload, fallback to recompute")

    candidates = await vector_sources.search_similar_chunks(
        thread_id, text_selection, turn
    )
    matches = await source_rerank.rerank_candidates(text_selection, candidates)
    l1cache.redis.setex(
        cache_key,
        _CHECK_SOURCE_CACHE_TTL,
        json.dumps(matches, default=str, ensure_ascii=False),
    )
    return {"matches": matches}
