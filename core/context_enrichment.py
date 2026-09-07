"""Pre-flight context enrichment — the "scout" that runs before the agent.

One fast structured-output call decides whether a single cheap retrieval would
give the main agent a running start on this turn, runs that one tool, and folds
the result into the ``<context_enrichment>`` block of the user message (see
``build_message_content`` in core/stream.py).

Design constraints, all deliberate:

- **Single call, no ReAct.** The model emits one JSON object and nothing else;
  *this* module matches it to a tool and calls it. The tool result is never fed
  back to the scout — there is no loop to run away.
- **Exactly one action per turn.** The scout is a starting point, not a
  research pass; fanning out here would just move the agent's job earlier and
  pay for it on the critical path.
- **Whitelist, not validation.** Anything that isn't one of the four retrieval
  actions — including a malformed or refused response — collapses to
  ``direct_response``, which enriches nothing and emits nothing.
- **Blocking, on purpose.** Its output has to be inside the user message, so
  the agent cannot start until it finishes. Everything here is therefore
  timeout-bounded, and every failure degrades to "no enrichment" rather than
  to an error.
- **First turn only**, and the caller enforces that (`_stream_agent` in
  core/stream.py, which is also where the exemption for user-named URLs
  lives). Nothing here sees the conversation, so there is nothing useful for
  it to say about a follow-up.
- **Citations are the real thing.** ``web_search`` here is the same
  ``core.tools.adapters.web_search`` the agent calls, so its results are
  credibility-classified and registered in the citation registry exactly like
  an agent-issued search. The ``[n]`` markers the agent sees in the injected
  block are live citation numbers it can cite directly, and the frontend
  receives them through the normal ``sources`` event.

The wire events this produces (``tool_call``, ``widget``) are byte-identical in
shape to the ones the agent's own tool loop produces — deliberately
indistinguishable to the frontend, which renders a scout search as just another
step in the timeline.

Replaces the old widget predictor: same "fast model in front of the agent"
idea, but its output now feeds the agent as well as the UI, instead of being a
decorative side channel the agent never saw.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import context_enrich_llm
from core.tools.adapters import (
    currency_convert,
    stock_search,
    weather_forecast,
    web_search,
)
from core.utils.citations import all_citations

logger = logging.getLogger(__name__)

# LangChain's `with_structured_output` hands the parsed pydantic object back on
# a response field that is typed as `None`, so pydantic emits a serializer
# warning on every single structured call. It is cosmetic and not fixable from
# here, and this module runs on every turn — without this filter it is one
# stderr paragraph per chat request. Matched on the exact message prefix so
# nothing else pydantic complains about is swallowed with it.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning, module="pydantic.main"
)


# ── Query length gate ───────────────────────────────────────────────────────
# A long query is usually a task (write this, refactor that, here is my essay),
# not something one cheap lookup helps with — and it is exactly the case where
# the scout's latency is least likely to be repaid. CJK has no spaces, so each
# character counts as a word.

_CJK_RE = re.compile(
    r'[一-鿿㐀-䶿'   # CJK Unified Ideographs (+ Extension A)
    r'぀-ゟ゠-ヿ'    # Hiragana, Katakana
    r'가-힯]'                # Hangul
)
_WORD_LIMIT = 50


def word_count(text: str) -> int:
    """Count words: each CJK character = 1 word, plus whitespace-split tokens."""
    cjk_chars = len(_CJK_RE.findall(text))
    latin_words = len(_CJK_RE.sub(' ', text).split())
    return cjk_chars + latin_words


# ── Decision schema ─────────────────────────────────────────────────────────
# One flat object rather than a tagged union: `with_structured_output` on a
# union produces a nested `anyOf` that small models fill in inconsistently,
# and a flat shape with empty-string defaults is trivially strict-JSON-schema
# compatible. Unused fields come back as "" and are ignored.

class EnrichmentDecision(BaseModel):
    """The scout's single decision for this turn."""

    action: Literal[
        "web_search",
        "weather_current",
        "weather_forecast",
        "stock",
        "currency",
        "direct_response",
    ] = Field(description="Which single retrieval to run, or direct_response for none.")
    search_query: str = Field(
        default="",
        description="For action=web_search: a short, broad English-or-native search query.",
    )
    location: str = Field(
        default="",
        description="For action=weather_*: city or place name in English, e.g. 'Tokyo'.",
    )
    ticker: str = Field(
        default="",
        description="For action=stock: Yahoo Finance ticker, e.g. 'AAPL' or '0700.HK'.",
    )
    base_currency: str = Field(
        default="",
        description="For action=currency: ISO 4217 code converted FROM, e.g. 'USD'.",
    )
    target_currency: str = Field(
        default="",
        description="For action=currency: ISO 4217 code converted TO, e.g. 'JPY'.",
    )


_SCOUT_PROMPT = """\
You are the pre-flight scout for a research assistant. You see ONE user query \
and decide the single most useful piece of context to fetch BEFORE the main \
agent starts working. The main agent will do the real research itself — your \
job is only to give it a running start.

# Output contract
Return one JSON object with an "action" field and only the fields that action \
needs. Exactly one action, never several. When in doubt, choose \
"direct_response".

# Actions

1. "web_search" — the default for anything factual, current, technical, local, \
or otherwise worth looking up.
   Field: "search_query".
   The query must be BROAD, not deep: it is a first sweep to orient the agent, \
not the finished research. Keep it SHORT (2-6 words), drop qualifiers, and aim \
at the general subject rather than the exact sub-question. Do not chain \
several questions into one query, do not add "2026", "latest" or "best" \
padding, and do not try to answer the question yourself.
   Write it in the language the answer will most likely be found in — English \
for international topics, the user's own language for local ones.
   Examples: "特斯拉2026年固态电池进展如何" -> "Tesla solid state battery"; \
"how do I set up a langgraph checkpointer with postgres" -> "langgraph \
postgres checkpointer".

2. "weather_current" — the query is about weather RIGHT NOW ("is it raining", \
"今天多少度", "current temperature").
   Field: "location" — a plain city or place name in English, shortest common \
form ("New York", not "New York City"). Resolve "here"/"outside"/"今天天气" \
against the user's location when one is given below.

3. "weather_forecast" — the query is about upcoming weather: tomorrow, this \
weekend, next week, later today, "should I bring an umbrella".
   Field: "location", same rules as above.

   Neither weather action applies to climate in general, historical weather, \
what a place is like in a named month ("hokkaido weather in january" is a \
travel question -> web_search), or a place mentioned for a non-weather reason.

4. "stock" — the query is about a specific publicly-traded company's stock, \
share price, earnings, or market performance. "how is AMD doing" and "nvidia \
earnings" both count.
   Field: "ticker" — uppercase Yahoo Finance symbol. Prefer the US listing when \
one exists, ADRs included: "BABA" Alibaba, "TSM" TSMC, "TM" Toyota, "SONY" \
Sony. Only a company with no US listing takes an exchange suffix: "0700.HK" \
Tencent, "005930.KS" Samsung, "1211.HK" BYD. For an index use its ETF proxy \
("SPY" for the S&P 500, "QQQ" for the Nasdaq 100).
   Not for private companies (SpaceX, ByteDance), crypto, or general investing \
advice -> web_search.

5. "currency" — the query asks to convert or compare two national currencies.
   Fields: "base_currency" (converted FROM) and "target_currency" (converted \
TO), both uppercase ISO 4217. "100 dollars in yen" is USD -> JPY; "日元汇率" \
from a US user is JPY -> USD.
   Not for crypto pairs or "which currency is strongest" -> web_search.

6. "direct_response" — no retrieval would help. Use it for translation, \
rewriting, editing, summarizing text the user supplied, classification, \
creative writing, brainstorming, small talk, personal preferences, and \
questions about Omni itself or how it behaves. Also use it when the query is \
too vague to search for.
   It is NOT for questions about the world. "who is X", "what is X", "is X \
any good" and anything else with a checkable answer are lookups -> \
web_search, however famous the subject.

# Rules
- Exactly one action. Never invent an action name or a field.
- Fill in only the fields the chosen action needs; leave the rest as "".
- A retrieval action must be clearly warranted. "direct_response" is a common \
and correct answer.
- Do not reason out loud and do not explain your choice.\
"""


def build_scout_messages(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> list[tuple[str, str]]:
    """Render the exact (system, user) messages the scout is asked with."""
    context_lines: list[str] = []
    if user_local_datetime:
        context_lines.append(f"User's current local date/time: {user_local_datetime}")
    if user_location:
        context_lines.append(f"User's current location: {user_location}")

    system_prompt = _SCOUT_PROMPT
    if context_lines:
        system_prompt += (
            "\n\n# Context about the user\n"
            "Use this to resolve relative or implicit references such as "
            "'here', 'nearby', 'now', 'today':\n"
            + "\n".join(context_lines)
        )
    return [("system", system_prompt), ("user", query)]


# ── Tool execution ──────────────────────────────────────────────────────────

_CLASSIFY_TIMEOUT_S = 6.0
_FETCH_TIMEOUT_S = 15.0
_SEARCH_K = 5
# Per-result snippet cap for the injected search block, and an overall cap for
# a raw JSON payload (a week of hourly weather is a lot of tokens for context
# the agent may not even use).
_MAX_RESULT_CHARS = 1500
_MAX_JSON_CHARS = 6000


async def _run_web_search(d: EnrichmentDecision) -> tuple[str, dict | None]:
    results = await web_search(d.search_query, k=_SEARCH_K)
    blocks: list[str] = []
    for r in results:
        n = r.get("n")
        marker = f"[{n}] " if n is not None else ""
        title = r.get("title", "") or ""
        url = r.get("url", "") or ""
        content = (r.get("content", "") or "")[:_MAX_RESULT_CHARS]
        blocks.append(f"{marker}{title} — {url}\n{content}".strip())
    return "\n\n".join(blocks), None


async def _run_weather(d: EnrichmentDecision) -> tuple[str, dict | None]:
    """Both weather actions resolve to `weather_forecast`.

    Its payload is a strict superset of `weather_current`'s (it carries the
    current conditions under `current`) and it is the shape the frontend's
    weather card is built against, so serving both intents from it means one
    upstream call, one citation, and a widget that renders either way. The
    scout still distinguishes the two intents because that is what tells us
    whether a weather lookup is warranted at all.
    """
    payload = await asyncio.to_thread(weather_forecast, d.location)
    return _dump(payload), {"widget": "weather", "data": payload}


async def _run_stock(d: EnrichmentDecision) -> tuple[str, dict | None]:
    payload = await asyncio.to_thread(stock_search, d.ticker)
    return _dump(payload), {"widget": "stock", "data": payload}


async def _run_currency(d: EnrichmentDecision) -> tuple[str, dict | None]:
    payload = await asyncio.to_thread(
        currency_convert, d.base_currency, d.target_currency
    )
    return _dump(payload), {"widget": "currency", "data": payload}


def _dump(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > _MAX_JSON_CHARS:
        text = text[:_MAX_JSON_CHARS] + " …(truncated)"
    return text


@dataclass(frozen=True)
class _Action:
    """One whitelisted action: the tool it maps to, and how to run it.

    ``tool`` and ``args`` are what goes out on the ``tool_call`` event, so they
    are the *agent's* tool name and argument keys (core/tools/adapters.py) —
    not the scout's action name — which is what makes a scout step
    indistinguishable from an agent step in the UI.
    """

    tool: str
    args: Callable[[EnrichmentDecision], dict]
    run: Callable[[EnrichmentDecision], Any]
    ready: Callable[[EnrichmentDecision], bool]


_ACTIONS: dict[str, _Action] = {
    "web_search": _Action(
        tool="web_search",
        args=lambda d: {"query": d.search_query},
        run=_run_web_search,
        ready=lambda d: bool(d.search_query.strip()),
    ),
    "weather_current": _Action(
        tool="weather_forecast",
        args=lambda d: {"location": d.location},
        run=_run_weather,
        ready=lambda d: bool(d.location.strip()),
    ),
    "weather_forecast": _Action(
        tool="weather_forecast",
        args=lambda d: {"location": d.location},
        run=_run_weather,
        ready=lambda d: bool(d.location.strip()),
    ),
    "stock": _Action(
        tool="stock_search",
        args=lambda d: {"symbol": d.ticker},
        run=_run_stock,
        ready=lambda d: bool(d.ticker.strip()),
    ),
    "currency": _Action(
        tool="currency_convert",
        args=lambda d: {
            "base_currency": d.base_currency,
            "target_currency": d.target_currency,
        },
        run=_run_currency,
        ready=lambda d: bool(d.base_currency.strip() and d.target_currency.strip()),
    ),
}


# ── Result ──────────────────────────────────────────────────────────────────

@dataclass
class Enrichment:
    """What the scout produced for one turn.

    ``text`` is the body of the ``<context_enrichment>`` block ("" when nothing
    was fetched). ``events`` are SSE payload dicts to emit *before* the agent
    starts, in order. ``sources`` are the citation records the run registered,
    to be folded into the turn's ``sources`` event the same way an uploaded
    document's are.
    """

    action: str = "direct_response"
    text: str = ""
    events: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)


_PREAMBLE = (
    "A fast pre-flight scout ran ONE retrieval for this turn before you "
    "started, to save you a round trip. It is a broad first sweep, not "
    "research: it may be shallow, partial, or beside the point. Read it "
    "first, then go deeper with your own tools whenever the question needs "
    "more — and ignore it entirely if it missed. You do not need to run this "
    "same lookup again."
)

# Appended to the preamble depending on what the run actually registered. The
# distinction is load-bearing: told unconditionally that "every [n] below is a
# real citation number", the model would emit a [1] for a result that carried
# no citation at all, and the frontend would render a marker linking nowhere.
_PREAMBLE_CITED = (
    " Every [n] below is a real citation number you can cite exactly as if you "
    "had run the tool yourself."
)
_PREAMBLE_UNCITED = (
    " This result carries no citation number — use the figures in it, but do "
    "not attach any [n] marker to them."
)


_llm = context_enrich_llm.with_structured_output(EnrichmentDecision, method="json_schema")


async def classify(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
    timeout: float = _CLASSIFY_TIMEOUT_S,
) -> EnrichmentDecision | None:
    """Run the scout model only — no fetching. None on any failure."""
    messages = build_scout_messages(query, user_location, user_local_datetime)
    try:
        with tracing_context(project_name="context-enrichment"):
            return await asyncio.wait_for(_llm.ainvoke(messages), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[context_enrichment] classification timed out after %ss", timeout)
    except Exception as exc:
        logger.warning("[context_enrichment] classification failed: %s", exc)
    return None


async def enrich_context(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> Enrichment:
    """Scout one retrieval for ``query`` and package it for the turn.

    Never raises: every failure path (long query, dead model, unparseable
    decision, unknown action, missing argument, failing tool) returns an empty
    `Enrichment`, which reads downstream as "this turn had no enrichment".
    """
    if not query.strip() or word_count(query) > _WORD_LIMIT:
        return Enrichment()

    t0 = time.monotonic()
    decision = await classify(query, user_location, user_local_datetime)
    if decision is None:
        return Enrichment()
    t_classify = time.monotonic() - t0

    # Whitelist: direct_response, an action we don't implement, and an action
    # whose required argument came back empty are all the same outcome — no
    # enrichment, nothing emitted, block left out of the prompt entirely.
    action = _ACTIONS.get(decision.action)
    if action is None:
        if decision.action != "direct_response":
            logger.warning("[context_enrichment] unknown action %r — ignored", decision.action)
        logger.info("[context_enrichment] %.2fs action=direct_response", t_classify)
        return Enrichment()
    if not action.ready(decision):
        logger.warning(
            "[context_enrichment] action %r missing its argument — ignored", decision.action
        )
        return Enrichment()

    args = action.args(decision)
    before = {c.get("n") for c in all_citations()}
    try:
        body, widget = await asyncio.wait_for(action.run(decision), timeout=_FETCH_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            "[context_enrichment] %s%s timed out after %ss", action.tool, args, _FETCH_TIMEOUT_S
        )
        return Enrichment()
    except Exception as exc:
        logger.warning("[context_enrichment] %s%s failed: %s", action.tool, args, exc)
        return Enrichment()

    if not (body or "").strip():
        logger.warning("[context_enrichment] %s%s returned nothing — ignored", action.tool, args)
        return Enrichment()

    # Citations registered by the tool we just ran (web_search registers one
    # per result including junk; the weather, stock and currency tools each
    # register their own). Read from the registry rather than the tool's return
    # value for the same reason core/stream.py does: the two can differ.
    sources = [
        {
            "n": c["n"],
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "content": c.get("content", ""),
            **({"credibility": c["credibility"]} if c.get("credibility") is not None else {}),
        }
        for c in all_citations()
        if c.get("n") is not None and c["n"] not in before
    ]

    call_line = ", ".join(f'{k}="{v}"' for k, v in args.items())
    preamble = _PREAMBLE + (_PREAMBLE_CITED if sources else _PREAMBLE_UNCITED)
    text = f"{preamble}\n\nTool called: {action.tool}({call_line})\n\n{body}"

    events: list[dict] = [{"type": "tool_call", "tool": action.tool, "args": args}]
    if widget:
        events.append({"type": "widget", **widget})

    logger.info(
        "[context_enrichment] %.2fs (classify %.2fs) action=%s tool=%s args=%s sources=%d",
        time.monotonic() - t0, t_classify, decision.action, action.tool, args, len(sources),
    )
    return Enrichment(action=decision.action, text=text, events=events, sources=sources)
