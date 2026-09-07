"""Tool adapter layer — the only tools the agent ever sees.

Each capability is one stable tool: a fixed name, a fixed schema, a fixed
return shape. Behind it sits a *provider* — the function that does the real
work — chosen from `PROVIDERS` below. Pointing `web_search` at Tavily instead
of SearXNG is one env var; the agent, the prompts, the skills, the evals and
every stored thread stay exactly as they are.

Behaviour that every provider of a capability should share — result caching,
credibility classification, citation numbering, failing soft — lives in the
adapter rather than the provider. A new provider only has to return the
canonical shape to inherit all of it.

Adding a provider:
    1. Write a function matching the capability's signature below.
    2. Add it to `PROVIDERS[capability]`.
    3. Select it with `<CAPABILITY>_PROVIDER=<name>` (e.g. WEB_SEARCH_PROVIDER=tavily).
"""

import asyncio
import json
import os
from typing import Any, Callable

from core.tools import coding_sandbox, currency_tool, searxng, tavily
from core.tools import stock_data_retriever, weather_tool, web_page_reader
from core.utils.citations import register_citation
from core.utils.redis_cache import l1cache
from core.utils.source_credibility import classify_sources

# capability -> {provider name: implementation}. The first entry of each is
# the default, so the common case needs no configuration at all.
PROVIDERS: dict[str, dict[str, Callable]] = {
    "web_search": {
        "searxng": searxng.search_web,
        "tavily": tavily.search_web,
    },
    "fetch_url": {
        "spider": web_page_reader.load_web_page_spider,
    },
    "python_exec": {
        "e2b": coding_sandbox.run_python_e2b,
    },
    "stock_search": {
        "yfinance": stock_data_retriever.get_stock_data,
    },
    "weather_current": {
        "owm": weather_tool.get_weather,
    },
    "weather_forecast": {
        "owm": weather_tool.get_weather_forecast,
    },
    "currency_convert": {
        "frankfurter": currency_tool.get_realtime_currency_rate,
    },
}


def provider_name(capability: str) -> str:
    """Which provider `capability` is currently configured to use."""
    configured = os.environ.get(f"{capability.upper()}_PROVIDER")
    available = PROVIDERS[capability]
    if configured and configured not in available:
        raise ValueError(
            f"{capability.upper()}_PROVIDER={configured!r} is not registered; "
            f"available: {', '.join(available)}"
        )
    return configured or next(iter(available))


def _provider(capability: str) -> Callable:
    return PROVIDERS[capability][provider_name(capability)]


# Resolve every capability once at import. A typo in a *_PROVIDER env var
# should fail at boot, not degrade quietly inside a request — `web_search`
# catches provider errors and answers "search unavailable", which would
# otherwise hide a misconfiguration behind what looks like an outage.
for _capability in PROVIDERS:
    provider_name(_capability)


# ── Shared attribution ──────────────────────────────────────────────────────
# Everything the agent retrieves from the open web goes through here, so
# credibility and citation numbering behave identically no matter which
# provider fetched it.

# Order results by credibility before the agent reads them: official/trusted/
# first_party first (equally — all three mean "the reader can lean on this"),
# unknown next, social_media last. Junk never appears: it is dropped from the
# agent-facing list entirely, though still registered as a citation below so
# the frontend's source list doesn't lose it.
_CREDIBILITY_RANK = {
    "official": 0,
    "trusted": 0,
    "first_party": 0,
    "unknown": 1,
    "social_media": 2,
}


async def _attribute(results: list[dict], query: str | None) -> list[dict]:
    """Classify, cite and rank retrieved sources."""
    out = []
    for item in await classify_sources(results, query):
        item = dict(item)
        credibility = item.get("credibility")  # {"label": ..., "reason": ...} | None
        # Registered regardless of tier — junk still gets an `n` and a citation
        # record so it isn't lost to the frontend's source list; it's only
        # excluded from what the agent reads.
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


# ── web_search ──────────────────────────────────────────────────────────────

# How long a cached result stays valid. A narrow window like "past day" is only
# correct while it's fresh; caching it for days would keep serving the same
# stale snapshot as if it were still "the last day".
_SEARCH_TTL = {"day": 60 * 30, "week": 3600 * 3, "month": 3600 * 24, "year": 3600 * 24 * 3}
_DEFAULT_SEARCH_TTL = 3600 * 24 * 3

_NO_RESULTS = [{"title": "No results found, please change your query", "url": "", "content": ""}]
_UNAVAILABLE = [
    {
        "title": "Search unavailable",
        "url": "",
        "content": (
            "The web search backend could not be reached. Answer from what you "
            "already have and say explicitly that you could not verify it."
        ),
    }
]


def _cached_search(query: str, k: int, time_range: str | None) -> list[dict]:
    """Run the configured search provider, Redis-cached with a TTL that follows
    `time_range`. Kept out of `web_search` below because citation numbers must
    be assigned fresh on every call, including on a cache hit."""
    name = provider_name("web_search")
    # Cache under the provider's own name: two providers answering the same
    # query are two different answers, and switching between them should not
    # serve the other one's results.
    cache_key = l1cache._build_cache_key(_cached_search, (name, query, k, time_range), {})
    cached = l1cache.redis.get(cache_key)
    if cached is not None:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            l1cache.redis.delete(cache_key)

    try:
        results = PROVIDERS["web_search"][name](query, k=k, time_range=time_range) or _NO_RESULTS
    except Exception as exc:
        # One provider is one point of failure for every search in the turn.
        # Degrade to something the agent can read and work around rather than
        # raising mid-answer — and never cache the failure.
        print(f"[web_search] provider {name!r} failed for {query!r}: {exc}")
        return _UNAVAILABLE

    ttl = _SEARCH_TTL.get(time_range, _DEFAULT_SEARCH_TTL)
    l1cache.redis.setex(cache_key, ttl, json.dumps(results, default=str, ensure_ascii=False))
    return results


async def web_search(query: str, k: int = 5, time_range: str | None = None) -> list[dict]:
    """
    Search the web. Your default tool for any public, factual, domain,
    practical, or real-world knowledge.

    Use it for definitions, explanations, advice, "what is", "how to", benefits
    or risks, troubleshooting, recommendations, comparisons, short or ambiguous
    questions, and anything touching health, medicine, law, finance, safety, or
    policy — even when the subject feels simple, familiar, or low-stakes.

    Do NOT search for work that can be done from the conversation alone:
    translation, rewriting, editing, summarising or classifying text the user
    supplied, creative writing, brainstorming, small talk, personal
    preferences, or questions about your own behaviour.

    Writing the query:
    - Keep it short and keyword-shaped. Drop filler words and question phrasing.
    - One subject per call. Split separate entities or aspects into separate
      calls rather than cramming them into one query; issuing several calls in
      the same turn is fine and usually better than one broad one.
    - Use the conversation to resolve a short or ambiguous follow-up before
      searching.
    - Never add an identifier the user, the conversation, or an earlier tool
      result did not give you — no invented years, versions, model names,
      titles, venues, hosts, or candidates.
    - For a recurring or time-sensitive subject, use neutral recency words
      ("latest", "next", "upcoming") instead of guessing a specific date, event,
      or outcome.

    Coverage, not a quota: search enough to support every substantive part of
    your answer. If the results come back incomplete, conflicting, snippet-only,
    one-sided, or silent on part of what was asked, search again with a sharper
    query or read the most relevant hits with `fetch_url` before you answer.
    Prefer primary sources and established outlets over aggregators, and when
    sources genuinely disagree, say so rather than quietly picking one.

    Args:
        query (str): The search query.
        k (int): The number of results to return. Default is 5. Max is 10.
        time_range (str, optional): Restrict results to a recent time window.
            Use this whenever the question is about something recent or
            time-sensitive (breaking news, live scores, "just happened",
            latest price/status) instead of adding words like "today" or
            "latest" to the query. One of "day", "week", "month", "year".
            Leave unset for queries with no time constraint.

    Returns:
        list[dict]: Results with `title`, `url`, `content`, and an `n` field —
        cite it inline as [n] when you use that result in your answer.
    """
    # This function is `async def`, so the agent awaits it directly on the
    # event loop instead of LangChain dispatching it to a worker thread the way
    # it does for plain sync tools. `_cached_search` blocks (Redis, then an
    # HTTP call), so running it inline would freeze that loop — and every other
    # concurrent thread's SSE stream on it — for the call's duration.
    results = await asyncio.to_thread(_cached_search, query, min(k, 10), time_range)
    return await _attribute(results, query)


# ── fetch_url ───────────────────────────────────────────────────────────────


async def fetch_url(url: str) -> dict:
    """
    Get the full text of a web page. Use it whenever page-level detail would
    make the answer better than a search snippet would.

    Prefer this over answering from snippets when the question turns on a
    specific source, list, policy, price or product page, article, table, exact
    wording, or any detail likely to sit below the fold of a snippet. Fetch
    several pages in the same turn when the answer depends on more than one.

    When the user pastes or names a URL, call this on that exact URL — do not
    `web_search` for it first and do not substitute a different source; a URL
    the user named outranks anything you would find yourself. The one exception
    is a page already delivered to you in `<context_enrichment>`: that content
    is the fetch, so do not fetch it again.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        dict: The URL, title, content, and an `n` field — cite it inline as
        [n] when you use this page's content in your answer.
    """
    # Blocking provider awaited off the event loop, for the reason in `web_search`.
    result = await asyncio.to_thread(_provider("fetch_url"), url)
    # No query available for a direct page load, so the LLM layer can't judge
    # "first_party" here — it falls back to domain-only signal.
    attributed = await _attribute([result], None)
    if attributed:
        return attributed[0]

    # Empty means `_attribute` dropped it as junk. Say so, rather than
    # returning content the agent would go on to cite.
    return {
        "url": result.get("url", "") or url,
        "title": result.get("title", ""),
        "content": (
            "This page was flagged as low-quality/unreliable and its content has "
            "been withheld. Do not cite it — try a different source."
        ),
    }


# ── Everything else ─────────────────────────────────────────────────────────
# These providers return their final payload directly: their sources are
# single, known and self-describing, so there is no credibility judgement to
# make and they register their own citations where they have one.


def python_exec(filename: str, code: str) -> str:
    """Execute Python code and return stdout, the final expression value, and any errors.

    You MUST use this — never work it out in your head, never approximate,
    never invent a number — for arithmetic beyond trivial mental math,
    statistics or probability, data analysis, unit conversions that apply a
    formula, numerical algorithms (sorting, search, optimisation, simulation),
    and anything the user asks you to calculate, compute, run, simulate, or
    verify with code.

    Do not use it for work that is not computation: explaining a concept,
    translating text, printing data you already have, or displaying a value you
    could simply write in the answer. No dry-run or test calls.

    Its output is visible to YOU only — the user never sees stdout. `print` is
    for values you need to read back and reason about, not a way to show the
    user anything. Anything the user should see has to be written into your
    answer.

    Write one complete, self-contained, immediately runnable script per call
    with minimal comments; each call is isolated, so nothing carries over from a
    previous one. When several results build on the same data, compute them in
    one script rather than alternating between preparation and use.

    The sandbox has no internet access — external requests and downloads fail.
    numpy, pandas, scipy, scikit-learn, sympy, and requests are pre-installed;
    for anything else, prepend:
        import subprocess; subprocess.run(["pip", "install", "pkg"], check=True)

    This tool is text-only. It cannot produce charts, plots, images, or any visual
    output — do NOT attempt to use matplotlib, PIL, or similar. For visualisations
    use the charting skill instead.

    Args:
        filename (str): A short name for the snippet, e.g. "compound_interest.py".
        code (str): The Python source to execute.
    """
    return _provider("python_exec")(filename, code)


def stock_search(symbol: str) -> dict[str, Any]:
    """Get the latest stock snapshot for a ticker.

    Use it whenever the question is about a specific publicly-traded company's
    share price, market performance, or earnings — "how is AMD doing" counts.
    For an index, use its ETF proxy ("SPY" for the S&P 500, "QQQ" for the Nasdaq
    100). Private companies, crypto, and general investing questions are
    `web_search` instead.

    Args:
        symbol (str): Stock ticker symbol, e.g. "TSLA". Prefer the US listing
            when one exists, ADRs included ("BABA", "TSM", "TM"); only a company
            with no US listing takes an exchange suffix ("0700.HK", "005930.KS").

    Returns:
        dict: Snapshot payload including key metrics.
    """
    return _provider("stock_search")(symbol)


def weather_current(location: str) -> dict:
    """Get the current weather for a location — conditions right now, nothing else.

    Use `weather_forecast` instead for tomorrow, the weekend, next week, later
    today, or any upcoming conditions. Neither weather tool answers questions
    about climate in general, historical weather, or what a place is typically
    like in a named month ("Hokkaido in January" is a travel question for
    `web_search`).

    Args:
        location (str): The location to get the weather for. MUST be in English.
    """
    return _provider("weather_current")(location)


def weather_forecast(location: str) -> dict:
    """Get the weather forecast for a location: current conditions, today's
    hourly slots, and a daily outlook out to about a week.

    Use it for anything upcoming — tomorrow, the weekend, next week, later
    today, "should I bring an umbrella". If the user asks about a day beyond
    what `daily_forecast` covers, say the forecast does not reach that far
    rather than guessing.

    Args:
        location (str): The location to forecast. MUST be in English.
    """
    return _provider("weather_forecast")(location)


def currency_convert(base_currency: str, target_currency: str) -> dict:
    """Get the real-time exchange rate between two national currencies.

    Use it whenever the question converts or compares two currencies. Crypto
    pairs and open-ended questions ("which currency is strongest") are
    `web_search` instead.

    Args:
        base_currency (str): ISO code to convert from, e.g. "USD".
        target_currency (str): ISO code to convert to, e.g. "CNY".
    """
    return _provider("currency_convert")(base_currency, target_currency)


# The tools handed to the agent, in the order they're offered.
AGENT_TOOLS = [
    web_search,
    fetch_url,
    weather_current,
    weather_forecast,
    stock_search,
    currency_convert,
    python_exec,
]
