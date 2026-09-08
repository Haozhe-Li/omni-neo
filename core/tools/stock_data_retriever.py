import yfinance as yf
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional
from core.utils.redis_cache import l1cache
from core.utils.citations import register_citation


# @l1cache(ttl=3600 * 24 * 3)
def _get_history_trend_cached(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """
    Get historical stock trends with intelligent sampling to reduce resolution.

    Args:
        symbol: Stock ticker symbol (e.g., "TSLA").
        period: Time period ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max").

    Returns:
        Sampled OHLCV data and movement summary.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            return {"success": False, "error": f"No historical data for {symbol}"}

        # Sampling rules based on period
        sample_rules = {
            "1mo": "3D",  # Every 3 days
            "3mo": "5D",  # Every 5 days
            "6mo": "1W",  # Every week
            "1y": "2W",  # Every 2 weeks
            "2y": "1ME",  # Every month
            "5y": "3ME",  # Quarterly
            "max": "6ME",  # Every 6 months
        }

        resample_freq = sample_rules.get(period, "1ME")

        # Standard financial aggregation rules
        ohlcv_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }

        # Resample and drop empty rows
        df_resampled = df.resample(resample_freq).agg(ohlcv_dict).dropna()

        # Limit maximum rows to avoid token explosion (max 100 rows)
        max_rows = 100
        if len(df_resampled) > max_rows:
            step = len(df_resampled) // max_rows
            df_resampled = df_resampled.iloc[::step]

        trend_data = df_resampled.reset_index()
        trend_data["Date"] = trend_data["Date"].dt.strftime("%Y-%m-%d")
        trend_list = trend_data[
            ["Date", "Open", "High", "Low", "Close", "Volume"]
        ].to_dict("records")

        # Calculate additional trend indicators
        closes = trend_data["Close"].values
        trend_summary = {
            "total_points": len(trend_list),
            "start_price": float(closes[0]),
            "end_price": float(closes[-1]),
            "total_return": float((closes[-1] - closes[0]) / closes[0] * 100),
            "volatility": float(np.std(closes) / np.mean(closes) * 100),
            "resample_freq": resample_freq,
        }

        return {
            "success": True,
            "data": trend_list,
            "summary": trend_summary,
            "period": period,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get history for {symbol}: {str(e)}",
        }


def get_history_trend(
    symbol: str,
    period: str = "5y",
) -> Dict[str, Any]:
    """Get historical stock trend data for a symbol and period.

    Args:
        symbol: Stock ticker symbol (e.g., "TSLA").
        period: Time period ("1mo", "3mo", "6mo", "1y", "2y", "5y", "max").
    Returns:
        Historical OHLCV trend payload.
    """
    return _get_history_trend_cached(symbol, period)


# @l1cache(ttl=3600 * 24) # 1 day cache for stock data
def _get_stock_data_cached(symbol: str) -> Dict[str, Any]:
    """
    Get core stock metrics including price, valuation, and financial indicators.

    Args:
        symbol: Stock symbol (e.g., "TSLA", "AAPL", "600519.SS").

    Returns:
        JSON containing key performance indicators.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Extract core fields (approx. 20 fields)
        core_data = {
            "symbol": info.get("symbol", symbol),
            "companyName": info.get("longName", "N/A"),
            "currentPrice": info.get("currentPrice", info.get("regularMarketPrice")),
            "currency": info.get("currency", "USD"),
            "change": info.get("regularMarketChange"),
            "changePercent": info.get("regularMarketChangePercent"),
            "dayRange": info.get("regularMarketDayRange"),
            "volume": info.get("regularMarketVolume"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "epsTTM": info.get("trailingEps"),
            "forwardEps": info.get("forwardEps"),
            "revenueGrowth": info.get("revenueGrowth"),
            "earningsGrowth": info.get("earningsGrowth"),
            "profitMargins": info.get("profitMargins"),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": info.get("currentRatio"),
            "recommendation": info.get(
                "recommendationMean"
            ),  # 1=Strong Buy, 5=Strong Sell
            "targetMeanPrice": info.get("targetMeanPrice"),
            "fiftyTwoWeekRange": info.get("fiftyTwoWeekRange"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

        # Filter out None values
        core_data = {k: v for k, v in core_data.items() if v is not None}

        return {
            "success": True,
            "data": core_data,
            "timestamp": info.get("regularMarketTime"),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get stock data for {symbol}: {str(e)}",
        }


def _quote_summary_text(data: Dict[str, Any]) -> str:
    """One-line human-readable quote, used as the citation's content."""
    parts = [f"{data.get('companyName') or data['symbol']} ({data['symbol']})"]
    price = data.get("currentPrice")
    if price is not None:
        parts.append(f"{price} {data.get('currency', '')}".strip())
    change = data.get("changePercent")
    if change is not None:
        parts.append(f"{change:+.2f}% on the day")
    cap = data.get("marketCap")
    if cap:
        parts.append(f"market cap {cap:,}")
    return ", ".join(parts) + "."


def get_stock_data(
    symbol: str,
) -> Dict[str, Any]:
    """Get latest stock snapshot data for a symbol.

    Registers a citation for the quote — same contract as the weather tools,
    and the reason is the same: the agent is told to cite anything a tool gave
    it, so a payload with no `n` leaves it choosing between citing a number
    that does not exist and stating a live market figure with no source at all.
    It reliably picked the first, and the frontend rendered a [1] that linked
    nowhere.

    Args:
        symbol: Stock ticker symbol (e.g., "TSLA").
    Returns:
        Stock snapshot payload including key metrics, plus an `n` citation
        number when the quote came back.
    """
    result = _get_stock_data_cached(symbol)
    data = result.get("data") if result.get("success") else None
    # A price is what makes the quote real. yfinance answers an unknown ticker
    # with success=True and a near-empty payload (companyName "N/A", no price),
    # which would otherwise register a citation titled "N/A" pointing at a
    # Yahoo page that 404s.
    if data and data.get("currentPrice") is not None:
        ticker = data.get("symbol", symbol)
        n = register_citation(
            title=f"{data.get('companyName') or ticker} ({ticker}) stock quote",
            # Deduped on url, so it is per-ticker: two quotes in one thread stay
            # two citations, and re-quoting the same ticker reuses its number.
            url=f"https://finance.yahoo.com/quote/{ticker}",
            content=_quote_summary_text(data),
        )
        if n is not None:
            result["n"] = n
    return result


# if __name__ == "__main__":
#     # Example usage:
#     print(json.dumps(get_stock_data("TSLA"), indent=2))
# print(json.dumps(get_history_trend("TSLA"), indent=2))
