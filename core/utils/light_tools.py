from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.tools.web_search import google_search, google_search_places
from core.tools.stock_data_retriever import get_stock_data
from core.tools.weather_tool import get_weather


@tool
def google_search_light(
    query: str,
    k: int = 5,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> list[dict]:
    """Search the web and return normalized Google search results for light agent."""
    _ = state
    return google_search(query=query, k=k)


@tool
def google_search_places_light(
    query: str,
    k: int = 5,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> list[dict]:
    """Search places and return normalized Google places results for light agent."""
    _ = state
    return google_search_places(query=query, k=k)


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