from dotenv import load_dotenv

load_dotenv()
from core.utils.redis_cache import l1cache
from core.utils.citations import register_citation
from core.utils.source_credibility import classify_sources
from langchain_community.utilities import GoogleSerperAPIWrapper
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tavily import TavilyClient
from typing import Literal
import asyncio
import json
import os
import arxiv

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _normalize_tavily_result_item(item: dict) -> dict:
    title = str(item.get("title", "") or "").strip()
    url = str(item.get("url", "") or "").strip()
    content = item.get("content")
    if content is None:
        content = item.get("raw_content", "")
    content = str(content or "").strip()
    return {
        "title": title,
        "url": url,
        "content": content,
    }


def _normalize_tavily_response(raw: dict, query: str) -> dict:
    if not isinstance(raw, dict):
        return {
            "query": query,
            "results": [],
            "error": "Search failed: invalid Tavily response format.",
        }

    normalized_results = []
    for item in raw.get("results", []):
        if not isinstance(item, dict):
            continue
        normalized_item = _normalize_tavily_result_item(item)
        if (
            normalized_item["title"]
            or normalized_item["url"]
            or normalized_item["content"]
        ):
            normalized_results.append(normalized_item)

    return {
        "query": raw.get("query", query),
        "follow_up_questions": raw.get("follow_up_questions"),
        "answer": raw.get("answer"),
        "images": raw.get("images", []),
        "results": normalized_results,
        "response_time": raw.get("response_time"),
        "request_id": raw.get("request_id"),
        "error": raw.get("error"),
    }


@l1cache(ttl=3600 * 24 * 3)
def tavily_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
):
    """
    Perform a web search using Tavily API.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return. Default is 5. Max is 10.
        topic (Literal["general", "news", "finance"]): The topic of the search. Default is "general".

    Returns:
        dict: A dictionary containing the search results.
    """
    TIMEOUT_SECONDS = 10

    def _search():
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=False,
            topic=topic,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_search)
        try:
            raw_res = future.result(timeout=TIMEOUT_SECONDS)
            return _normalize_tavily_response(raw_res, query)
        except TimeoutError:
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "response_time": None,
                "request_id": None,
                "error": f"Search timed out after {TIMEOUT_SECONDS} seconds.",
            }
        except Exception as e:
            return {
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "response_time": None,
                "request_id": None,
                "error": f"Search failed: {str(e)}",
            }


# tbs -> how long a cached result stays valid. A narrow window like "past
# hour" is only correct while it's fresh; caching it for days would keep
# serving the same stale snapshot as if it were still "the last hour".
_TBS_CACHE_TTL = {
    "qdr:h": 60 * 5,
    "qdr:d": 60 * 30,
    "qdr:w": 3600 * 3,
    "qdr:m": 3600 * 24,
    "qdr:y": 3600 * 24 * 3,
}
_DEFAULT_SEARCH_TTL = 3600 * 24 * 3


def _google_search_uncached(
    query: str, k: int = 5, tbs: str | None = None
) -> list[dict]:
    search = GoogleSerperAPIWrapper(k=k, type="search", tbs=tbs)
    res = search.results(query)
    normalized_results = []
    if res.get("answerBox"):
        answer_box = res["answerBox"]
        normalized_results.append(
            {
                "title": answer_box.get("title", ""),
                "url": "https://www.google.com/search?q=" + query.replace(" ", "+"),
                "content": "Answer from Google Answer Box: "
                + answer_box.get("snippet", ""),
            }
        )

    if res.get("knowledgeGraph"):
        kg = res["knowledgeGraph"]
        normalized_results.append(
            {
                "title": kg.get("title", ""),
                "url": kg.get("descriptionLink", ""),
                "content": "Knowledge Graph: " + kg.get("description", ""),
            }
        )

    if res.get("organic"):
        for item in res["organic"]:
            title = item.get("title", "").strip()
            url = item.get("link", "").strip()
            content = item.get("snippet", "").strip()
            if title or url or content:
                normalized_results.append(
                    {
                        "title": title,
                        "url": url,
                        "content": content,
                    }
                )
    res = normalized_results[:k]
    if not res:
        return [
            {
                "title": "No results found, please change your query",
                "url": "",
                "content": "",
            }
        ]
    return res


def _google_search_cached(query: str, k: int = 5, tbs: str | None = None) -> list[dict]:
    """Same Redis-backed cache as the rest of this module (see `l1cache`), but
    with a TTL that depends on `tbs` — kept separate from citation numbering
    below, which must be assigned fresh on every call even on a cache hit."""
    cache_key = l1cache._build_cache_key(
        _google_search_uncached, (query, k, tbs), {}
    )
    cached = l1cache.redis.get(cache_key)
    if cached is not None:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            l1cache.redis.delete(cache_key)

    result = _google_search_uncached(query, k, tbs)
    ttl = _TBS_CACHE_TTL.get(tbs, _DEFAULT_SEARCH_TTL)
    l1cache.redis.setex(cache_key, ttl, json.dumps(result, default=str, ensure_ascii=False))
    return result


# Rank used to sort search results by credibility before they reach the
# agent: official/trusted/first_party first (equally — all three are "the
# reader can lean on this"), unknown in the middle, social_media last. Junk
# never appears here — it's dropped from the agent-facing list entirely
# (still registered as a citation, see below, so it's not lost to the
# frontend's source list, just kept out of what grounds the answer).
_CREDIBILITY_RANK = {
    "official": 0,
    "trusted": 0,
    "first_party": 0,
    "unknown": 1,
    "social_media": 2,
}


async def google_search(
    query: str, k: int = 5, tbs: str | None = None
) -> list[dict]:
    """
    Perform an google search using Google Serper API.

    Args:
        query (str): The search query.
        k (int): The number of results to return. Default is 5. Max is 10.
        tbs (str, optional): Google's time-range filter. Use this whenever the
            question is about something recent or time-sensitive (breaking
            news, live scores, "just happened", latest price/status) instead
            of adding words like "today"/"latest" to the query. One of:
            - "qdr:h" — past hour
            - "qdr:d" — past day
            - "qdr:w" — past week
            - "qdr:m" — past month
            - "qdr:y" — past year
            Leave unset for queries with no time constraint.

    Returns:
        list[dict]: A list of search result dictionaries. Each carries a `n`
        field — cite it inline as [n] when you use that result in your answer.
    """
    k = min(k, 10)
    # _google_search_cached is synchronous (blocking Redis cache lookup, and
    # a blocking Serper HTTP call on a cache miss) — this function is itself
    # `async def`, so the agent awaits it directly on the event loop instead
    # of LangChain dispatching it to a worker thread the way it does for
    # plain sync tools. Calling it inline would freeze that loop — and every
    # other concurrent thread's SSE stream on it — for the call's duration.
    results = await asyncio.to_thread(_google_search_cached, query, k, tbs)
    results = await classify_sources(results, query)
    out = []
    for item in results:
        item = dict(item)
        credibility = item.get("credibility")  # {"label": ..., "reason": ...} | None
        # Registered regardless of tier — junk still gets an `n` and a citation
        # record (so it's not lost to the frontend's source list) — it's just
        # excluded from the list handed back to the agent below.
        n = register_citation(
            item.get("title", ""),
            item.get("url", ""),
            item.get("content", ""),
            credibility=credibility,
        )
        if credibility and credibility.get("label") == "junk":
            continue
        if n is not None:
            item["n"] = n
        out.append(item)
    out.sort(
        key=lambda item: _CREDIBILITY_RANK.get(
            (item.get("credibility") or {}).get("label"), 1
        )
    )
    return out


# @l1cache(ttl=3600 * 24 * 90)
def google_search_places(query: str, k: int = 5) -> list[dict]:
    """
    Use Google Search for places, restaurants, etc.

    query (str): The place to search for, must include "near <location>"
    k (int, optional): The number of results to return. Defaults to 5. Max to be 10.
    """
    k = min(k, 5)
    search = GoogleSerperAPIWrapper(k=k, type="places")
    res = search.results(query)
    return res.get("places", [])[:k]


# @l1cache(ttl=3600 * 24 * 90)
def arxiv_search(query: str, k: int = 5) -> list[dict]:
    """
    Perform an arxiv search using Arxiv API.

    Args:
        query (str): The search query.
        k (int): The number of results to return. Default is 3. Max is 5.

    Returns:
        list[dict]: A list of search result dictionaries.

    """
    k = min(k, 5)
    search = arxiv.Search(
        query=query, max_results=k, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for result in search.results():
        results.append(
            {
                "title": result.title,
                "url": result.pdf_url,
                "content": result.summary,
            }
        )
    if not results:
        return [
            {
                "title": "No results found, please change your query",
                "url": "",
                "content": "",
            }
        ]
    return results
