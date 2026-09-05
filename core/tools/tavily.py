"""Tavily provider for the `web_search` capability.

Kept alongside the SearXNG provider as the fallback worth reaching for when
the self-hosted instance is the thing that broke: it is a paid API with its
own crawl, so it fails independently. Select it with `WEB_SEARCH_PROVIDER=tavily`.
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
_TIMEOUT_SECONDS = 10


def search_web(query: str, k: int = 5, time_range: str | None = None) -> list[dict]:
    """Canonical `web_search` provider signature — see core/tools/adapters.py."""
    # The SDK call is blocking with no timeout of its own, so it gets one here.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _client.search,
            query,
            max_results=k,
            include_raw_content=False,
            time_range=time_range,
        )
        try:
            raw = future.result(timeout=_TIMEOUT_SECONDS)
        except TimeoutError:
            raise TimeoutError(f"Tavily search timed out after {_TIMEOUT_SECONDS}s")

    results = []
    for item in (raw or {}).get("results", []):
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
        )
    return [r for r in results if any(r.values())][:k]
