from langchain_community.utilities import GoogleSerperAPIWrapper
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from tavily import TavilyClient
from typing import Literal
import os

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


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
            return future.result(timeout=TIMEOUT_SECONDS)
        except TimeoutError:
            return {
                "results": [],
                "error": f"Search timed out after {TIMEOUT_SECONDS} seconds.",
            }
        except Exception as e:
            return {"results": [], "error": f"Search failed: {str(e)}"}


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
