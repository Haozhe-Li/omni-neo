"""Exa provider for the `web_search` capability — the default.

Exa is a neural search API with its own index, so it fails independently of
both the self-hosted SearXNG instance and Tavily. It became the default after
2026-09-07, when every engine SearXNG fans out to was suspended at once and
the product had no working search for hours.

Two properties decided it over the alternatives:

- **`instant` mode is fast enough to sit in front of an agent turn.** Measured
  0.4-0.6s against the public API, versus ~1s for `auto` and 4-15s for the
  `deep` modes. Search is on the critical path of every TOOL_NEEDED turn, so
  latency here is latency the user watches.
- **It answers Chinese queries with real sources.** That is not a given: the
  SearXNG instance's engines returned SEO-farm domains with word-salad titles
  for the same queries, and a Chinese query is roughly 60% of this product's
  traffic. Exa returned Shangri-La, Ctrip and tradingeconomics for the same
  ones.

`highlights` rather than `text` for the snippet: `text` is the full page as
markdown, which is both slower and far more than the agent needs to decide
whether to `fetch_url` the page. Highlights are the query-relevant excerpts,
which is exactly what a search result's `content` field is for, and they are
capped at `_HIGHLIGHT_MAX_CHARS` — uncapped they came back at 1,780 and 3,609
characters for a two-result query, and a five-result search would spend most
of its context on pages the agent then decides not to read.
"""

import os
import time

import httpx

_ENDPOINT = "https://api.exa.ai/search"
_TIMEOUT_SECONDS = 15.0

# Search mode. `instant` is the fastest tier; see the module docstring for the
# measured latencies and why that matters here.
SEARCH_TYPE = os.environ.get("EXA_SEARCH_TYPE", "instant").strip() or "instant"

# `web_search`'s time filter, in days. Exa has no equivalent parameter — it
# filters on the document's own publication date instead — so the window is
# converted to a `startPublishedDate` below. A page with no publication date
# Exa can read is excluded by that filter, which is the right trade for a
# query that asked for recency.
_TIME_RANGE_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}

# Cap on each result's excerpt. Verified by effect rather than by the request
# being accepted: Exa silently ignores request keys it does not know (a
# deliberately bogus one alongside this changed nothing), so the only proof
# that a content option works is the returned length actually changing —
# 3,609 chars became 990 with this set.
_HIGHLIGHT_MAX_CHARS = int(os.environ.get("EXA_HIGHLIGHT_MAX_CHARS", "1000"))


def _start_published_date(time_range: str | None) -> str | None:
    days = _TIME_RANGE_DAYS.get((time_range or "").strip().lower())
    if not days:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - days * 86400))


def _snippet(item: dict) -> str:
    """The result's readable body, best-effort across the shapes Exa returns.

    `highlights` is what we ask for, but a result can come back without them
    (Exa omits the field rather than returning an empty list), and `summary` /
    `text` appear when a caller asks for those. Falling through the three keeps
    this working if `EXA_SEARCH_TYPE` or the contents request is ever changed
    without anyone remembering to update the parser.
    """
    highlights = item.get("highlights")
    if isinstance(highlights, list) and highlights:
        return " … ".join(str(h).strip() for h in highlights if str(h).strip())
    for key in ("summary", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def search_web(query: str, k: int = 5, time_range: str | None = None) -> list[dict]:
    """Canonical `web_search` provider signature — see core/tools/adapters.py.

    The key is read per call rather than at import. `core/tools/adapters.py`
    resolves every provider at import time to fail fast on a misconfigured
    capability, and reading the key there would make an unset `EXA_API_KEY`
    stop the whole app from booting — where the fallback chain is designed to
    treat exactly that as "try the next provider".
    """
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set")

    payload: dict = {
        "query": query,
        "type": SEARCH_TYPE,
        "numResults": max(1, k),
        "contents": {"highlights": {"maxCharacters": _HIGHLIGHT_MAX_CHARS}},
    }
    start = _start_published_date(time_range)
    if start:
        payload["startPublishedDate"] = start

    resp = httpx.post(
        _ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()

    results = []
    for item in (body or {}).get("results") or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "content": _snippet(item),
            }
        )
    return [r for r in results if any(r.values())][:k]
