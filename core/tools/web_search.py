from langchain_community.utilities import GoogleSerperAPIWrapper
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tavily import TavilyClient
from typing import Literal
import os

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
        if normalized_item["title"] or normalized_item["url"] or normalized_item["content"]:
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


def google_search(querys: list[str], k: int = 5, tbs: str = "") -> list[dict]:
    """
    Perform an google search using Google Serper API.

    Args:
        querys (list[str]): A list of search queries.
        k (int): The number of results to return per query. Default is 5. Max is 10.
        tbs (str): Time-based search filter. Default is an empty string.

    Returns:
        list[dict]: A list of search result dictionaries.

    """
    print(f"Searching google: {querys}, k: {k}, tbs: {tbs}")
    k = min(k, 10)
    if not tbs:
        search = GoogleSerperAPIWrapper(k=k)
    else:
        search = GoogleSerperAPIWrapper(k=k, tbs=tbs)
    results = []
    for query in querys:
        result = search.results(query)
        results.append(result)
    return results
