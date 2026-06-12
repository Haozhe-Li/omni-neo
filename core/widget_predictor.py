"""Pre-flight widget predictor.

Before (and in parallel with) the main agent, a very fast model classifies the
user's query and decides which live-data *widgets* to surface — weather, a stock
quote, a place card, or an FX rate. It runs on Groq `gpt-oss-120b` via
``bind_tools`` so the classification + argument extraction come back as
structured tool calls (0..N hits, "none" = no tool call). Each hit's data is then
fetched and pushed to the frontend as a ``widget`` SSE event *before* the answer
streams.

This path is intentionally fully decoupled from the agent's own tool loop: it
never feeds back into the agent, and an occasional duplicate fetch is acceptable.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.tools.weather_tool import get_weather
from core.tools.stock_data_retriever import get_stock_data
from core.tools.web_search import google_search_places
from core.tools.currency_tool import get_realtime_currency_rate


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


_WIDGET_TOOLS = [WeatherWidget, StockWidget, PlaceWidget, CurrencyWidget]

_PREDICTOR_PROMPT = (
    "You route a user query to live-data widgets. Call every widget tool that the "
    "query clearly and directly needs, with precise arguments. If the query does "
    "not obviously call for current weather, a specific stock, a place/venue, or a "
    "currency conversion, call NOTHING. Do not guess. Be fast."
)

# Low reasoning effort: this is a latency-critical pre-flight classifier.
_predictor_model = ChatGroq(
    model="openai/gpt-oss-120b",
    reasoning_effort="low",
    api_key=os.getenv("GROQ_API_KEY"),
).bind_tools(_WIDGET_TOOLS)


def _fetch(name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Fetch the data payload for a predicted widget (runs in a worker thread)."""
    try:
        if name == "WeatherWidget":
            return {"widget": "weather", "data": get_weather(args["location"])}
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
    except Exception as exc:  # never let a widget failure break the chat
        print(f"[widget_predictor] fetch failed for {name}: {exc}")
    return None


async def predict_widgets(
    query: str,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> list[dict[str, Any]]:
    """Classify ``query`` and return a list of ready-to-emit widget payloads.

    Each item looks like ``{"widget": "weather", "data": {...}}``. Returns an
    empty list when nothing matches or on any error.
    """
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
    if not tool_calls:
        return []

    results = await asyncio.gather(
        *(asyncio.to_thread(_fetch, tc["name"], tc.get("args", {})) for tc in tool_calls)
    )
    return [r for r in results if r]
