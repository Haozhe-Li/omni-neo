from langchain_community.document_loaders import SpiderLoader
import asyncio
import concurrent.futures
import json
import os
import re
from urllib.parse import urlsplit

from core.utils.citations import register_citation
from core.utils.redis_cache import r as _redis
from core.utils.source_credibility import classify_sources

# from core.utils.redis_cache import l1cache

_TIMEOUT_SECONDS = 5
_TIMEOUT_RESULT = {
    "url": "",
    "content": "Failed to load the web page: request timed out. Do not try the same URL again.",
    "title": "Timeout",
}

# Cloudflare sits in front of omniknows.xyz and blocks Spider outright — a
# direct fetch from here would fare no better, since it gates the inbound
# request itself, not which client makes it. So first-party content is never
# fetched over the network at all: the frontend pushes it straight into the
# same Upstash Redis database this backend already talks to (same
# UPSTASH_REDIS_REST_URL/TOKEN on both sides — see core/utils/redis_cache.py),
# and we just read it back out here.
_FIRST_PARTY_HOST = "omniknows.xyz"
# Written by the frontend's lib/llms-txt.ts, refreshed on demand (the
# benchmark page's "Ask Omni" link and its Refresh button) rather than on a
# schedule — see that file for why. Fixed key, not derived from the request
# URL: there is exactly one production llms.txt.
_LLMS_TXT_REDIS_KEY = "agent_page:https://omniknows.xyz/benchmark/llms.txt"
# Omni Pages are already Redis-native — `publish:{id}` is written directly by
# the frontend's /api/publish route (see app/api/publish/route.ts there), with
# `answer` holding the page's markdown body. No separate mirror needed.
_PAGE_ID_RE = re.compile(r"^/pages/([0-9a-f]{12})$")


def first_party_redis_shortcut(url: str) -> dict | None:
    """Read known first-party omniknows.xyz content straight out of Redis
    instead of fetching it. Returns None for anything not covered — the
    caller falls back to the normal Spider path unchanged."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host != _FIRST_PARTY_HOST and not host.endswith("." + _FIRST_PARTY_HOST):
        return None
    path = parsed.path.rstrip("/") or "/"

    if path == "/benchmark/llms.txt":
        raw = _redis.get(_LLMS_TXT_REDIS_KEY)
        data = _decode_redis_json(raw)
        if data is None:
            return None
        return {
            "url": url,
            "title": data.get("title") or "Omni Benchmarks",
            "content": data.get("content") or "",
        }

    match = _PAGE_ID_RE.match(path)
    if match:
        raw = _redis.get(f"publish:{match.group(1)}")
        data = _decode_redis_json(raw)
        if data is None:
            return None
        return {
            "url": url,
            "title": data.get("title") or "AI Research Report",
            "content": data.get("answer") or "",
        }

    return None


def _decode_redis_json(raw) -> dict | None:
    if not raw:
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


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
    shortcut = first_party_redis_shortcut(url)
    if shortcut is not None:
        return shortcut

    # A worker-thread-safe timeout. `signal.SIGALRM` only works on the main
    # thread, and this function is itself blocking (see load_web_page below,
    # which must call it via asyncio.to_thread rather than inline — it can't
    # assume it's already off the event loop), so we enforce the wall-clock
    # limit with a future instead.
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
    # load_web_page is itself `async def`, so the agent awaits it directly on
    # the event loop rather than LangChain dispatching it to a worker thread
    # the way it does for plain sync tools. load_web_page_spider is fully
    # synchronous and blocks for up to _TIMEOUT_SECONDS — calling it inline
    # would freeze that loop, and every other concurrent thread's SSE stream
    # riding on it, for the fetch's duration.
    result = await asyncio.to_thread(load_web_page_spider, url)
    # No query/topic available for a direct page load, so the LLM layer
    # can't judge "first_party" here — it'll fall back to domain-only signal.
    classified = await classify_sources([result], None)
    result = classified[0] if classified else result
    resolved_url = result.get("url", "") or url
    credibility = result.get("credibility")  # {"label": ..., "reason": ...} | None
    # Registered regardless of tier — junk still gets an `n` and a citation
    # record (so it's not lost to the frontend's source list) — the agent
    # just never sees the `n` or the actual page content for it below.
    n = register_citation(
        result.get("title", ""),
        resolved_url,
        result.get("content", ""),
        credibility=credibility,
    )
    if credibility and credibility.get("label") == "junk":
        return {
            "url": resolved_url,
            "title": result.get("title", ""),
            "content": "This page was flagged as low-quality/unreliable and its content has been withheld. Do not cite it — try a different source.",
            "credibility": credibility,
        }
    if n is not None:
        result["n"] = n
    return result
