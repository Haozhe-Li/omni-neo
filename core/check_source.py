from __future__ import annotations

from core.utils import source_rerank, vector_sources
from core.utils.redis_cache import l1cache

_CHECK_SOURCE_CACHE_TTL = 60 * 60 * 24 * 10

# rerank_candidates hands every candidate to gpt-oss-20b in one prompt — past
# a handful it starts misjudging keep/drop and mangling excerpts. Candidates
# are already score-sorted descending by vector_sources.search_similar_chunks,
# so capping here just keeps the top-scored ones.
_MAX_RERANK_CANDIDATES = 5


@l1cache(ttl=_CHECK_SOURCE_CACHE_TTL)
async def _check_source_matches_cached(
    thread_id: str, text_selection: str, turn: int | None = None
) -> list[dict]:
    candidates = await vector_sources.search_similar_chunks(
        thread_id, text_selection, turn
    )
    candidates = candidates[:_MAX_RERANK_CANDIDATES]
    return await source_rerank.rerank_candidates(text_selection, candidates)


async def check_source_matches(
    thread_id: str, text_selection: str, turn: int | None = None
) -> dict:
    text_selection = text_selection.strip()
    if not text_selection:
        return {"error": "Text selection is empty"}

    matches = await _check_source_matches_cached(thread_id, text_selection, turn)
    return {"matches": matches}
