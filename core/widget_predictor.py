"""Pre-flight widget predictor.

Before (and in parallel with) the main agent, a very fast model classifies the
user's query and decides which live-data *widgets* to surface — weather, a stock
quote, a place card, an FX rate, or an entity knowledge card. It runs on Groq
``gpt-oss-120b`` via ``bind_tools`` so the classification + argument extraction
come back as structured tool calls (0..N hits, "none" = no tool call). Each
hit's data is then fetched and pushed to the frontend as a ``widget`` SSE event
*before* the answer streams.

This path is intentionally fully decoupled from the agent's own tool loop: it
never feeds back into the agent, and an occasional duplicate fetch is acceptable.

Word-count gate: if the query contains more than 10 words (CJK characters are
counted individually), the predictor skips classification entirely and returns
an empty list. Long queries are almost never single-widget requests.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.tools.weather_tool import get_weather_forecast
from core.tools.stock_data_retriever import get_stock_data
from core.tools.web_search import google_search_places
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


# ── Widget tool schemas (classification + argument extraction) ──────────────

class WeatherWidget(BaseModel):
    """Show a weather card. Use only when the user asks about current/forecast weather for a place."""
    location: str = Field(description="City or place name, e.g. 'Tokyo'.")


class StockWidget(BaseModel):
    """Show a stock quote card. Use only when the user references a specific publicly-traded company or ticker."""
    ticker: str = Field(description="Ticker symbol, e.g. 'AAPL'.")


class PlaceWidget(BaseModel):
    """Show a places/map card. Use only when the user asks about venues, businesses, or locations to visit."""
    query: str = Field(description="Place search query, e.g. 'coffee shops in Seattle'.")


class CurrencyWidget(BaseModel):
    """Show an FX rate card. Use only when the user asks to convert or compare two currencies."""
    base_currency: str = Field(description="Base currency code, e.g. 'USD'.")
    target_currency: str = Field(description="Target currency code, e.g. 'JPY'.")


class EntityWidget(BaseModel):
    """Show a knowledge card for ONE specific named entity — a person, company, product, country, or landmark.
    Only call this when the ENTIRE query is about a single named entity.
    Do NOT call for comparisons (A vs B), how-to questions, multi-subject queries, or abstract topics."""
    entity_name: str = Field(description="The canonical name of the entity, e.g. 'Donald Trump' or 'LangChain'.")


_WIDGET_TOOLS = [WeatherWidget, StockWidget, PlaceWidget, CurrencyWidget, EntityWidget]

_PREDICTOR_PROMPT = (
    "You route a user query to live-data widgets. Call every widget tool that the "
    "query clearly and directly needs, with precise arguments. If the query does "
    "not obviously call for current weather, a specific stock, a place/venue, a "
    "currency conversion, or a single named entity, call NOTHING. Do not guess. Be fast."
)

# Low reasoning effort: this is a latency-critical pre-flight classifier.
_predictor_model = ChatGroq(
    model="openai/gpt-oss-120b",
    reasoning_effort="low",
    api_key=os.getenv("GROQ_API_KEY"),
).bind_tools(_WIDGET_TOOLS)


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

def _fetch(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch the data payload for a predicted widget (runs in a worker thread)."""
    try:
        if name == "WeatherWidget":
            return {"widget": "weather", "data": get_weather_forecast(args["location"])}
        if name == "StockWidget":
            return {"widget": "stock", "data": get_stock_data(args["ticker"])}
        if name == "PlaceWidget":
            return {"widget": "place", "data": google_search_places(args["query"])}
        if name == "CurrencyWidget":
            return {
                "widget": "currency",
                "data": get_realtime_currency_rate(
                    args["base_currency"], args["target_currency"]
                ),
            }
        if name == "EntityWidget":
            return _fetch_entity(args["entity_name"])
    except Exception as exc:
        print(f"[widget_predictor] fetch failed for {name}: {exc}")
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

    context_lines: list[str] = []
    if user_local_datetime:
        context_lines.append(f"User's current local date/time: {user_local_datetime}")
    if user_location:
        context_lines.append(f"User's current location: {user_location}")

    system_prompt = _PREDICTOR_PROMPT
    if context_lines:
        system_prompt += (
            "\n\nContext about the user (use to resolve relative or implicit "
            "references such as 'here', 'nearby', 'now', 'today'):\n"
            + "\n".join(context_lines)
        )

    try:
        resp = await _predictor_model.ainvoke(
            [
                ("system", system_prompt),
                ("user", query),
            ]
        )
    except Exception as exc:
        print(f"[widget_predictor] classification failed: {exc}")
        return []

    tool_calls = getattr(resp, "tool_calls", None) or []
    print(f"[widget_predictor] tool_calls for {query!r}: {[tc['name'] for tc in tool_calls]}")
    if not tool_calls:
        return []

    results = await asyncio.gather(
        *(asyncio.to_thread(_fetch, tc["name"], tc.get("args", {})) for tc in tool_calls)
    )
    return [r for r in results if r]
