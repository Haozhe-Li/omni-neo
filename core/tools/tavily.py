"""Tavily provider for the `web_search` capability — last in the fallback chain.

Reached only when both Exa and the self-hosted SearXNG have failed
(core/tools/adapters.py::_provider_chain). It is a paid API with its own crawl,
so it fails independently of the other two, which is the whole reason it is
still here.

Two settings, both chosen for that last-resort role:

- `search_depth="ultra-fast"` — 0.10-0.22s, against 0.55s for `fast`, 1.19s for
  `basic` and 2.02s for `advanced`. The valid set is exactly
  `ultra-fast | fast | basic | advanced`; the API rejects anything else, which
  is how it was established rather than guessed.
- `include_answer=False` — Tavily can synthesise an answer from the results,
  and this tool must not. The agent writes the answer, and a second model's
  prose arriving inside a search result is both a citation with no source and
  an invitation to copy it.

Quality is a real trade at this depth and worth knowing before changing the
default back: on "上海静安区日料店推荐", `ultra-fast` returned a Douyin search
page and two job-listing sites, while `basic` returned a Zhihu top-ten list and
two local restaurant guides. `TAVILY_SEARCH_DEPTH` overrides it without a
deploy if the degraded-mode results ever matter more than the seconds.
"""

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
_TIMEOUT_SECONDS = 10

# One of: ultra-fast | fast | basic | advanced. Anything else is a 400 from the
# API, which is where this list came from.
SEARCH_DEPTH = os.environ.get("TAVILY_SEARCH_DEPTH", "ultra-fast").strip() or "ultra-fast"


def search_web(query: str, k: int = 5, time_range: str | None = None) -> list[dict]:
    """Canonical `web_search` provider signature — see core/tools/adapters.py."""
    # The SDK call is blocking with no timeout of its own, so it gets one here.
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _client.search,
            query,
            max_results=k,
            search_depth=SEARCH_DEPTH,
            # Explicit rather than left to the API default: this tool returns
            # sources, never a synthesised answer.
            include_answer=False,
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
