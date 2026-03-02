from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.tools.web_search import google_search, google_search_places
from core.tools.stock_data_retriever import get_stock_data
from core.tools.weather_tool import get_weather
from core.tools.web_page_reader import load_web_page
from core.tools.currency_tool import get_realtime_currency_rate


@tool
def google_search_light(
    query: str,
    k: int = 5,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> list[dict]:
    """
    Use Google Search for online information retrival.

    query (str): The query to search for.
    k (int, optional): The number of results to return. Defaults to 5. Max to be 10.
    """
    _ = state
    return google_search(query=query, k=k)


@tool
def google_search_places_light(
    query: str,
    k: int = 5,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> list[dict]:
    """
    Use Google Search for places, restaurants, etc.

    query (str): The place to search for, must include "near <location>"
    k (int, optional): The number of results to return. Defaults to 5. Max to be 10.
    """
    _ = state
    return google_search_places(query=query, k=k)


@tool
def get_stock_data_light(
    symbol: str,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """
    Use Yahoo Finance for real time stock data. (US Market only)

    symbol (str): The stock ticker symbol.
    """
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
    """
    Use OpenWeatherMap for current weather.

    location (str): The location to get the weather for. location MUST BE IN ENGLISH.
    """
    _ = state
    return get_weather(location)


@tool
def load_web_page_light(
    url: str,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """
    Visit a web page and return the content.

    url (str): The URL to load.
    """
    _ = state
    return load_web_page(url)


@tool
def get_realtime_currency_rate_light(
    base_currency: str,
    target_currency: str,
    state: Annotated[dict[str, Any], InjectedState] = None,
) -> dict[str, Any]:
    """
    Get the real-time exchange rate between two currencies.

    base_currency (str): The base currency.
    target_currency (str): The target currency.
    """
    _ = state
    result = get_realtime_currency_rate(base_currency, target_currency)
    return {
        "base_currency": base_currency,
        "target_currency": target_currency,
        "currency": result,
    }
