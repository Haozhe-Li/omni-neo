from __future__ import annotations

from core.utils import source_rerank, vector_sources
from core.utils.redis_cache import l1cache

_CHECK_SOURCE_CACHE_TTL = 60 * 60 * 24 * 10


@l1cache(ttl=_CHECK_SOURCE_CACHE_TTL)
async def _check_source_matches_cached(
    thread_id: str, text_selection: str, turn: int | None = None
) -> list[dict]:
    candidates = await vector_sources.search_similar_chunks(
        thread_id, text_selection, turn
    )
    return await source_rerank.rerank_candidates(text_selection, candidates)


async def check_source_matches(
    thread_id: str, text_selection: str, turn: int | None = None
) -> dict:
    text_selection = text_selection.strip()
    if not text_selection:
        return {"error": "Text selection is empty"}

    matches = await _check_source_matches_cached(thread_id, text_selection, turn)
    return {"matches": matches}
