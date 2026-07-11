from langchain_community.document_loaders import SpiderLoader
import concurrent.futures
import os

from core.utils.citations import register_citation
from core.utils.source_credibility import classify_sources

# from core.utils.redis_cache import l1cache

_TIMEOUT_SECONDS = 5
_TIMEOUT_RESULT = {
    "url": "",
    "content": "Failed to load the web page: request timed out. Do not try the same URL again.",
    "title": "Timeout",
}


def _load_spider(url: str):
    loader = SpiderLoader(
        api_key=os.getenv("SPIDER_API_KEY"),
        url=url,
        mode="scrape",
        params={"request_timeout": _TIMEOUT_SECONDS, "return_format":"markdown"},
    )
    return loader.load()


# @l1cache(
#     ttl=3600 * 24 * 90
# )  # Cache for 90 days since historical web page content doesn't change
def load_web_page_spider(url: str) -> dict:
    """Load a web page and return its content.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        dict: The loaded web page content as a dictionary with URL and content keys.
    """
    # A worker-thread-safe timeout. `signal.SIGALRM` only works on the main
    # thread, but agent tools run inside `asyncio.to_thread` worker threads, so
    # we enforce the wall-clock limit with a future instead.
    # Note: avoid using ThreadPoolExecutor as a context manager — its __exit__
    # calls shutdown(wait=True), which blocks until the worker finishes and
    # silently defeats the timeout. We call shutdown(wait=False) explicitly so
    # the function returns on time while the leaked thread drains in the background.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_load_spider, url)
    try:
        documents = future.result(timeout=_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        return {**_TIMEOUT_RESULT, "url": url}
    except Exception:
        pool.shutdown(wait=False, cancel_futures=True)
        return {"url": url, "content": "Failed to load the web page.", "title": "Error"}
    pool.shutdown(wait=False)

    if not documents:
        return {
            "url": url,
            "content": "No content found on the web page. This could happen if the firewall blocks the request or the page is empty.",
        }
    # print(f"length of content: {len(documents[0].page_content)}")
    return {
        "url": url,
        "content": documents[0].page_content,
        "title": documents[0].metadata.get("title", "No title found"),
    }


async def load_web_page(
    url: str,
):
    """Get the full text of a web page.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        dict: A dictionary with the URL, title, content, and a `n` field —
        cite it inline as [n] when you use this page's content in your answer.
    """
    result = load_web_page_spider(url)
    # No query/topic available for a direct page load, so the LLM layer
    # can't judge "first_party" here — it'll fall back to domain-only signal.
    classified = await classify_sources([result], None)
    result = classified[0] if classified else result
    n = register_citation(
        result.get("title", ""),
        result.get("url", "") or url,
        result.get("content", ""),
        credibility=result.get("credibility"),
    )
    if n is not None:
        result["n"] = n
    return result
