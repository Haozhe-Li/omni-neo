from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.tools.web_search import tavily_search
from core.tools.stock_data_retriever import get_stock_data
from core.tools.weather_tool import get_weather


@tool
def tavily_search_light(
    query: str,
    topic: Literal["general", "news", "finance"] = "general",
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """Search the web and return raw Tavily result JSON for light agent."""
    _ = state
    return tavily_search(query=query, max_results=5, topic=topic)


@tool
def get_stock_data_light(
    symbol: str,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """Get latest stock data for one ticker and return normalized payload."""
    _ = state
    result = get_stock_data(symbol)
    return {
        "symbol": symbol,
        "stock": result,
    }

@tool
def get_weather_light(
    location: str,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """Get current weather for a location."""
    _ = state
    return get_weather(location)