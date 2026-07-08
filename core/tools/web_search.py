from dotenv import load_dotenv

load_dotenv()
from core.utils.redis_cache import l1cache
from core.utils.citations import register_citation
from langchain_community.utilities import GoogleSerperAPIWrapper
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tavily import TavilyClient
from typing import Literal
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


@l1cache(ttl=3600 * 24 * 3)
def _google_search_cached(query: str, k: int) -> list[dict]:
    """Cached Serper call, kept separate from citation numbering below —
    numbers must be assigned fresh on every call (they're scoped to the
    current agent turn), even when the search itself is served from cache."""
    search = GoogleSerperAPIWrapper(k=k, type="search")
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


def google_search(query: str, k: int = 3) -> list[dict]:
    """
    Perform an google search using Google Serper API.

    Args:
        query (str): The search query.
        k (int): The number of results to return. Default is 3. Max is 10.
        k (int): The number of results to return per query. Default is 5. Max is 10.

    Returns:
        list[dict]: A list of search result dictionaries. Each carries a `n`
        field — cite it inline as [n] when you use that result in your answer.
    """
    k = min(k, 10)
    results = _google_search_cached(query, k)
    out = []
    for item in results:
        item = dict(item)
        n = register_citation(item.get("title", ""), item.get("url", ""), item.get("content", ""))
        if n is not None:
            item["n"] = n
        out.append(item)
    return out


@l1cache(ttl=3600 * 24 * 90)
def google_search_places(query: str, k: int = 3) -> list[dict]:
    """
    Use Google Search for places, restaurants, etc.

    query (str): The place to search for, must include "near <location>"
    k (int, optional): The number of results to return. Defaults to 5. Max to be 10.
    """
    k = min(k, 5)
    search = GoogleSerperAPIWrapper(k=k, type="places")
    res = search.results(query)
    return res.get("places", [])[:k]


@l1cache(ttl=3600 * 24 * 90)
def arxiv_search(query: str, k: int = 3) -> list[dict]:
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
