"""Pre-flight widget predictor.

Before (and in parallel with) the main agent, a very fast model classifies the
user's query and decides which live-data *widgets* to surface — weather, a stock
quote, an FX rate, or an entity knowledge card. Each hit's data is then fetched
and pushed to the frontend as a ``widget`` SSE event *before* the answer streams.

The model returns a **single line of JSON** (``{"widgets":[{"tool":…,"args":…}]}``)
rather than native tool calls. Plain text is the format an SFT'd small model can
be trained on directly — no chat-template tool-call tokens to match byte for
byte, no dependency on the serving stack exposing ``tools``/guided decoding — so
the same code path works for a hosted LoRA and for the current Groq model. The
legacy ``bind_tools`` path is still selectable via ``WIDGET_PREDICTOR_BACKEND``
for A/B comparison.

This path is intentionally fully decoupled from the agent's own tool loop: it
never feeds back into the agent, and an occasional duplicate fetch is acceptable.

Word-count gate: if the query contains more than 10 words (CJK characters are
counted individually), the predictor skips classification entirely and returns
an empty list. Long queries are almost never single-widget requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from langchain_community.utilities import GoogleSerperAPIWrapper
from langsmith import tracing_context
from pydantic import BaseModel, Field, ValidationError

from langchain_groq import ChatGroq

from core.llm import gpt_oss_20b, widget_predictor_llm
from core.tools.weather_tool import get_weather_forecast
from core.tools.stock_data_retriever import get_stock_data
from core.tools.currency_tool import get_realtime_currency_rate


# ── Word-count gate ─────────────────────────────────────────────────────────

_CJK_RE = re.compile(
    r'[一-鿿㐀-䶿'    # CJK Unified Ideographs (+ Extension A)
    r'぀-ゟ゠-ヿ'     # Hiragana, Katakana
    r'가-힯]'                  # Hangul
)
_WORD_LIMIT = 10


def _word_count(text: str) -> int:
    """Count words: each CJK character = 1 word, plus whitespace-split tokens."""
    cjk_chars = len(_CJK_RE.findall(text))
    without_cjk = _CJK_RE.sub(' ', text)
    latin_words = len(without_cjk.split())
    return cjk_chars + latin_words


# ── Widget argument schemas ─────────────────────────────────────────────────
# One pydantic model per widget, keyed by the short tool name the LLM emits.
# These are the *contract* for the JSON path: anything the model returns that
# does not validate against them is dropped, so a malformed prediction degrades
# to "no widget" instead of a fetch crash.

class WeatherArgs(BaseModel):
    location: str = Field(description="City or place name in English, e.g. 'Tokyo'.")


class StockArgs(BaseModel):
    ticker: str = Field(description="Yahoo Finance ticker symbol, e.g. 'AAPL' or '0700.HK'.")


class CurrencyArgs(BaseModel):
    base_currency: str = Field(description="ISO 4217 code converted FROM, e.g. 'USD'.")
    target_currency: str = Field(description="ISO 4217 code converted TO, e.g. 'JPY'.")


class EntityArgs(BaseModel):
    entity_name: str = Field(description="Canonical name of one entity, e.g. 'Donald Trump'.")


_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "weather": WeatherArgs,
    "stock": StockArgs,
    "currency": CurrencyArgs,
    "entity": EntityArgs,
}

# Cap on how many widgets one query may produce. A model that starts listing
# every entity it can think of is malfunctioning, and each hit costs a network
# fetch on the critical path before the answer streams.
_MAX_WIDGETS = 4

# The predictor sits in front of the answer: a slow classification is worse
# than no widget at all.
_PREDICT_TIMEOUT_S = 8.0


# ── Prompt ──────────────────────────────────────────────────────────────────
# Kept in one place and built by `build_predictor_messages` so that training
# data, offline evals and production all render byte-identical prompts. Changing
# this string invalidates any LoRA trained against it.

_PREDICTOR_PROMPT = """\
You are the widget router for a search assistant. You read ONE user query and \
decide which live-data cards ("widgets") to show above the answer.

# Output contract
Reply with a single line of JSON and nothing else — no prose, no explanation, \
no markdown fence.
Shape: {"widgets":[{"tool":"<name>","args":{...}}]}
When no widget clearly applies, reply exactly: {"widgets":[]}
That is the correct answer for most queries.
Never invent a tool name or an argument key. Never repeat the same tool with \
the same arguments.

# Widgets

1. weather — args: {"location": string}
   Use when the query asks for current conditions or a forecast for a place.
   "location" is a plain city or place name in English, in its shortest common \
form: "New York", not "New York City".
   Resolve "here", "outside", "今天天气" against the user's location when one is \
given in the context below.
   Do NOT use for: climate in general, historical weather, what a place is like \
in a named month ("hokkaido weather in january" is a travel question), or a \
place mentioned for any non-weather reason.

2. stock — args: {"ticker": string}
   Use when the query is about a specific publicly-traded company's stock, \
share price, earnings, or market performance. "how is AMD doing" and "nvidia \
earnings" are stock queries.
   "ticker" is the uppercase Yahoo Finance symbol. Prefer the company's listing \
on a major US exchange whenever it has one, ADRs included: "BABA" Alibaba, \
"TSM" TSMC, "TM" Toyota, "SONY" Sony, "SAP" SAP. Only a company with no US \
listing takes an exchange suffix: "0700.HK" Tencent, "005930.KS" Samsung, \
"1211.HK" BYD, "1810.HK" Xiaomi.
   For an index, use its ETF proxy ("SPY" for the S&P 500, "QQQ" for the \
Nasdaq 100) — raw index symbols carry no quote data to render.
   Do NOT use for: private companies (DJI, SpaceX, ByteDance), \
cryptocurrencies, currencies, or how-to questions about investing in general.

3. currency — args: {"base_currency": string, "target_currency": string}
   Use when the query asks to convert or compare two national currencies.
   Both values are uppercase ISO 4217 codes. "base_currency" is the currency \
converted FROM, "target_currency" the one converted TO: "100 dollars in yen" \
is base USD, target JPY; "日元汇率" from a US user is base JPY, target USD.
   Do NOT use for: crypto pairs, stock prices, or open-ended questions like \
"which currency is strongest".

4. entity — args: {"entity_name": string}
   Use when the query ASKS ABOUT one specific named entity — a person, company, \
product, cryptocurrency, country, or landmark. A question about a property of \
that entity counts: "who founded Tesla" and "is the iPhone waterproof" fire \
entity cards for Tesla and the iPhone.
   "entity_name" is always the entity the query is ABOUT, never the answer the \
query is looking for: "who is the ceo of starbucks" is Starbucks, not the \
person; "法国首都是哪" is France, not Paris.
   Write it as the canonical, widely-used English name with no legal suffix: \
"Apple" not "Apple Inc.", "Jensen Huang" not "黄仁勋", "Mount Fuji" not "富士山".
   Do NOT use when the query wants a PLAN, a LIST, a RECOMMENDATION or \
DIRECTIONS involving the entity rather than facts about the entity itself: \
"京都三日游安排", "best sushi in osaka" and "how do i get to heathrow" all get \
nothing.
   Also skip when: the query has two or more subjects ("iphone vs android", \
"中美关系"); it asks how to do something ("how to learn python"); or there is no \
one specific named entity ("best laptops", "what causes inflation", "explain \
transformers").

# Rules
- Fire a widget only when the query clearly and directly calls for that data. \
When nothing fits, {"widgets":[]} is the right answer and it is a common one.
- Emit several widgets only when the query genuinely needs them: "weather in \
Tokyo and Osaka" is two weather widgets.
- weather, stock and currency win over entity for the same subject: "Tesla \
stock price" is a stock widget only, not also an entity card.
- The query may be in any language; arguments are always in the canonical \
English or standard-code form described above.
- Do not reason out loud, do not ask questions, do not add fields.

# Examples
Query: weather in tokyo
{"widgets":[{"tool":"weather","args":{"location":"Tokyo"}}]}

Query: 北京今天天气怎么样
{"widgets":[{"tool":"weather","args":{"location":"Beijing"}}]}

Query: nvda stock
{"widgets":[{"tool":"stock","args":{"ticker":"NVDA"}}]}

Query: 腾讯股价
{"widgets":[{"tool":"stock","args":{"ticker":"0700.HK"}}]}

Query: 100 usd to jpy
{"widgets":[{"tool":"currency","args":{"base_currency":"USD","target_currency":"JPY"}}]}

Query: nasdaq 100
{"widgets":[{"tool":"stock","args":{"ticker":"QQQ"}}]}

Query: who is sam altman
{"widgets":[{"tool":"entity","args":{"entity_name":"Sam Altman"}}]}

Query: who founded spacex
{"widgets":[{"tool":"entity","args":{"entity_name":"SpaceX"}}]}

Query: langchain vs llamaindex
{"widgets":[]}

Query: how do I center a div
{"widgets":[]}

Query: 微信怎么换头像
{"widgets":[]}

Query: 苹果好吃吗
{"widgets":[]}

Query: best noise cancelling headphones
{"widgets":[]}\
"""

_EMPTY_PREDICTION = '{"widgets":[]}'


def build_predictor_messages(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> list[tuple[str, str]]:
    """Render the exact (system, user) messages the predictor is asked with.

    Training-data generation, offline evals and production all go through this
    function so the prompt a fine-tuned model sees at inference time is the same
    string it was trained on.
    """
    context_lines: list[str] = []
    if user_local_datetime:
        context_lines.append(f"User's current local date/time: {user_local_datetime}")
    if user_location:
        context_lines.append(f"User's current location: {user_location}")

    system_prompt = _PREDICTOR_PROMPT
    if context_lines:
        system_prompt += (
            "\n\n# Context about the user\n"
            "Use this to resolve relative or implicit references such as "
            "'here', 'nearby', 'now', 'today':\n"
            + "\n".join(context_lines)
        )

    return [("system", system_prompt), ("user", query)]


# ── Prediction parsing ──────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_prediction(text: str) -> list[dict[str, Any]]:
    """Parse a raw model response into ``[{"tool": …, "args": {…}}, …]``.

    Deliberately forgiving about the wrapper (code fences, leading prose, a bare
    list instead of the object) and strict about the contents: an entry whose
    tool is unknown or whose args fail schema validation is dropped rather than
    passed on to a fetcher. Returns ``[]`` for anything unsalvageable.
    """
    if not text:
        return []

    blob = _FENCE_RE.sub("", text.strip()).strip()

    # Tolerate leading/trailing prose by slicing to the outermost brackets.
    start = min((i for i in (blob.find("{"), blob.find("[")) if i != -1), default=-1)
    if start == -1:
        return []
    end = max(blob.rfind("}"), blob.rfind("]"))
    if end < start:
        return []

    try:
        payload = json.loads(blob[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []

    if isinstance(payload, dict):
        raw_items = payload.get("widgets")
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = None
    if not isinstance(raw_items, list):
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool") or item.get("name")
        if not isinstance(tool, str):
            continue
        tool = tool.strip().lower()
        schema = _ARG_SCHEMAS.get(tool)
        if schema is None:
            continue
        args = item.get("args")
        if args is None:
            # Allow flat form: {"tool": "weather", "location": "Tokyo"}.
            args = {k: v for k, v in item.items() if k not in ("tool", "name")}
        if not isinstance(args, dict):
            continue
        try:
            validated = schema(**args).model_dump()
        except (ValidationError, TypeError):
            continue
        key = tool + json.dumps(validated, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append({"tool": tool, "args": validated})
        if len(out) >= _MAX_WIDGETS:
            break
    return out


def render_prediction(widgets: list[dict[str, Any]]) -> str:
    """Serialize widgets back to the canonical one-line JSON the model must emit.

    This is the SFT target format: one fixed surface form per label, so the
    model never has to choose between equivalent spellings.
    """
    items = [
        {"tool": w["tool"], "args": {k: w["args"][k] for k in sorted(w["args"])}}
        for w in widgets
    ]
    return json.dumps({"widgets": items}, ensure_ascii=False, separators=(",", ":"))


# ── Backends ────────────────────────────────────────────────────────────────
# "json": one plain-text completion parsed by `parse_prediction` — the path an
# SFT'd model plugs into. "tools": the original `bind_tools` classification,
# kept for A/B comparison against the fine-tune.

_BACKEND = os.getenv("WIDGET_PREDICTOR_BACKEND", "json").strip().lower()

# `reasoning_format`/`reasoning_effort` are Groq-only knobs, needed there
# because gpt-oss otherwise inlines its reasoning into `content` as
# <think>…</think> and lands it in front of the JSON. The fine-tuned model is
# served by W&B's OpenAI-compatible endpoint, which rejects both — so bind them
# only when the configured predictor is actually a Groq model. That keeps this
# module working whichever way `widget_predictor_llm` is pointed.
_json_model = widget_predictor_llm
if isinstance(widget_predictor_llm, ChatGroq):
    _json_model = widget_predictor_llm.bind(
        reasoning_format="parsed", reasoning_effort="low", max_tokens=256
    )


class _LegacyWeatherWidget(BaseModel):
    """Show a weather card. Use only when the user asks about current/forecast weather for a place."""
    location: str = Field(description="City or place name, e.g. 'Tokyo'.")


class _LegacyStockWidget(BaseModel):
    """Show a stock quote card. Use only when the user references a specific publicly-traded company or ticker."""
    ticker: str = Field(description="Ticker symbol, e.g. 'AAPL'.")


class _LegacyCurrencyWidget(BaseModel):
    """Show an FX rate card. Use only when the user asks to convert or compare two currencies."""
    base_currency: str = Field(description="Base currency code, e.g. 'USD'.")
    target_currency: str = Field(description="Target currency code, e.g. 'JPY'.")


class _LegacyEntityWidget(BaseModel):
    """Show a knowledge card for ONE specific named entity — a person, company, product, country, or landmark.
    Only call this when the ENTIRE query is about a single named entity.
    Do NOT call for comparisons (A vs B), how-to questions, multi-subject queries, or abstract topics."""
    entity_name: str = Field(description="The canonical name of the entity, e.g. 'Donald Trump' or 'LangChain'.")


_LEGACY_TOOL_NAMES = {
    "_LegacyWeatherWidget": "weather",
    "_LegacyStockWidget": "stock",
    "_LegacyCurrencyWidget": "currency",
    "_LegacyEntityWidget": "entity",
}

# Bound to gpt-oss-20b explicitly rather than to `widget_predictor_llm`: this
# path exists to reproduce the pre-fine-tune behaviour for comparison, and the
# LoRA was never trained to emit tool calls, so pointing it here would measure
# nothing. Reach it with WIDGET_PREDICTOR_BACKEND=tools.
_legacy_model = gpt_oss_20b.bind_tools(
    [_LegacyWeatherWidget, _LegacyStockWidget, _LegacyCurrencyWidget, _LegacyEntityWidget]
)


def response_text(resp: Any) -> str:
    """Flatten a chat response's content to text.

    Reasoning models on the OpenAI Responses API return a list of typed blocks
    rather than a string, so the JSON body has to be picked out of the blocks
    instead of str()-ing the whole list. Teacher labelling runs through the same
    parser as production, so this lives here rather than in the script.
    """
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


async def classify(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
    backend: str | None = None,
    model: Any = None,
    timeout: float = _PREDICT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Run the classifier only — no data fetching. Returns ``[{"tool","args"}]``."""
    widgets, _ = await classify_verbose(
        query, user_location, user_local_datetime, backend, model, timeout
    )
    return widgets


async def classify_verbose(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
    backend: str | None = None,
    model: Any = None,
    timeout: float = _PREDICT_TIMEOUT_S,
) -> tuple[list[dict[str, Any]], str]:
    """Classify, and also return the model's raw text.

    Split out from `predict_widgets` so evals and training-data generation can
    score the model's decision without paying for four upstream APIs. Pass
    ``model`` to drive an arbitrary chat model (a teacher, a hosted LoRA)
    through the identical prompt and parser. The raw string is what lets an
    eval separate "the model decided nothing applies" from "the model emitted
    something the parser could not read" — two very different failures that
    both surface as an empty list.
    """
    backend = backend or _BACKEND
    messages = build_predictor_messages(query, user_location, user_local_datetime)
    if model is None:
        model = _legacy_model if backend == "tools" else _json_model
    else:
        backend = "json"

    try:
        with tracing_context(project_name="widget-predictor"):
            resp = await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[widget_predictor] classification timed out after {timeout}s")
        return [], ""
    except Exception as exc:
        print(f"[widget_predictor] classification failed: {exc}")
        return [], ""

    if backend == "tools":
        calls = [
            {"tool": _LEGACY_TOOL_NAMES[tc["name"]], "args": tc.get("args", {})}
            for tc in (getattr(resp, "tool_calls", None) or [])
            if tc["name"] in _LEGACY_TOOL_NAMES
        ]
        return calls, render_prediction(calls)

    raw = response_text(resp)
    return parse_prediction(raw), raw


# ── Entity knowledge-graph fetcher ──────────────────────────────────────────

def _fetch_entity_image(entity_name: str) -> tuple[str, str]:
    """Return (imageUrl, sourceLink) from the first Serper image result, or ('', '')."""
    try:
        img_search = GoogleSerperAPIWrapper(k=1, type="images")
        raw = img_search.results(entity_name)
        images = raw.get("images") or []
        if images:
            first = images[0]
            return first.get("imageUrl") or "", first.get("link") or ""
    except Exception as exc:
        print(f"[widget_predictor] image search failed for {entity_name!r}: {exc}")
    return "", ""


def _fetch_entity(entity_name: str) -> dict[str, Any] | None:
    """Search for an entity and return its Knowledge Graph data + image."""
    import concurrent.futures

    # Run text (KG) and image searches in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        kg_future = pool.submit(
            lambda: GoogleSerperAPIWrapper(k=3, type="search").results(entity_name)
        )
        img_future = pool.submit(_fetch_entity_image, entity_name)
        try:
            raw = kg_future.result(timeout=8)
        except Exception as exc:
            print(f"[widget_predictor] entity search failed for {entity_name!r}: {exc}")
            return None
        image_url, source_link = img_future.result(timeout=8)

    kg = raw.get("knowledgeGraph")
    if not kg:
        print(f"[widget_predictor] no knowledgeGraph for {entity_name!r} — skipping")
        return None

    # Require both a knowledge graph AND an image to show the card.
    final_image = kg.get("imageUrl") or image_url
    if not final_image:
        print(f"[widget_predictor] no image for {entity_name!r} — skipping")
        return None

    data: dict[str, Any] = {"name": entity_name}
    data["title"] = kg.get("title") or entity_name
    data["type"] = kg.get("type") or ""
    data["image_url"] = final_image
    data["source_link"] = source_link

    print(f"[widget_predictor] entity widget built: title={data['title']!r} type={data['type']!r}")
    return {"widget": "entity", "data": data}


# ── Easter egg: Haozhe Li entity card ───────────────────────────────────────

_HAOZHE_RE = re.compile(
    r'李浩哲'
    r'|haozhe[\s\-]?li'
    r'|li[\s\-]?haozhe'
    r'|haozhe',
    re.IGNORECASE,
)

_HAOZHE_WIDGET: dict[str, Any] = {
    "widget": "entity",
    "data": {
        "name": "Haozhe Li （李浩哲）",
        "title": "Haozhe Li （李浩哲）",
        "type": "AI Engineer who builds Omni. Currently working on Agentic AI in finance.",
        "image_url": "https://cdn.haozheli.com/DSC03805.jpeg",
        "source_link": "https://haozhe.li",
    },
}


# ── Main predictor ───────────────────────────────────────────────────────────

_FETCHERS = {
    "weather": lambda a: {"widget": "weather", "data": get_weather_forecast(a["location"])},
    "stock": lambda a: {"widget": "stock", "data": get_stock_data(a["ticker"])},
    "currency": lambda a: {
        "widget": "currency",
        "data": get_realtime_currency_rate(a["base_currency"], a["target_currency"]),
    },
    "entity": lambda a: _fetch_entity(a["entity_name"]),
}


def _fetch(tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch the data payload for a predicted widget (runs in a worker thread)."""
    try:
        return _FETCHERS[tool](args)
    except Exception as exc:
        print(f"[widget_predictor] fetch failed for {tool}: {exc}")
        return None


async def predict_widgets(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> list[dict[str, Any]]:
    """Classify ``query`` and return a list of ready-to-emit widget payloads.

    Each item looks like ``{"widget": "weather", "data": {...}}``. Returns an
    empty list when nothing matches, the query is too long, or on any error.
    """
    print(f"[widget_predictor] called, query={query!r}, word_count={_word_count(query)}, limit={_WORD_LIMIT}")

    # Easter egg: any mention of Haozhe Li (in any form) → instant card, no LLM.
    if _HAOZHE_RE.search(query):
        print("[widget_predictor] haozhe easter egg triggered")
        return [_HAOZHE_WIDGET]

    # Gate: long queries are rarely single-widget requests — skip entirely.
    if _word_count(query) > _WORD_LIMIT:
        return []

    predicted = await classify(query, user_location, user_local_datetime)
    print(f"[widget_predictor] prediction for {query!r}: {render_prediction(predicted)}")
    if not predicted:
        return []

    results = await asyncio.gather(
        *(asyncio.to_thread(_fetch, p["tool"], p["args"]) for p in predicted)
    )
    return [r for r in results if r]
