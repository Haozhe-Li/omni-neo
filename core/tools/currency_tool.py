import httpx
from core.utils.redis_cache import l1cache
from core.utils.citations import register_citation


# The rates Frankfurter serves are the ECB's daily reference rates, published
# here — the page a reader should land on when they click the citation.
_ECB_REFERENCE_RATES_URL = (
    "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/"
    "euro_reference_exchange_rates/html/index.en.html"
)


def _rate_source_url(base: str, target: str) -> str:
    """The ECB page, tagged with the pair so each one is its own citation.

    Citations dedupe on url, so the bare page would collapse every pair in a
    thread into one entry — and that entry keeps the *first* pair's title and
    content, which means a second pair's [n] would open a card describing a
    rate it never quoted. The fragment makes the url unique per pair while
    still landing the reader on exactly the same page. Same trick, for the same
    reason, as `_weather_source_url` in core/tools/weather_tool.py.
    """
    return f"{_ECB_REFERENCE_RATES_URL}#{base.lower()}-{target.lower()}"


def _rate_summary_text(base: str, target: str, rate: float, date: str) -> str:
    """One-line human-readable rate, used as the citation's content."""
    return f"1 {base} = {rate} {target} (ECB reference rate, {date})."


# @l1cache(ttl=60 * 60 * 12)
def get_realtime_currency_rate(base_currency: str, target_currency: str) -> dict:
    """
    Get the real-time exchange rate between two currencies.

    Registers a citation for the rate — same contract as the weather and stock
    tools. Without one the agent, which is told to cite whatever a tool gave
    it, would emit a [n] no source backs and the frontend would render a
    citation that links nowhere.

    The citation points at the ECB's own reference-rates page rather than at
    the Frankfurter request that fetched the number: Frankfurter republishes
    the ECB's daily rates, so the ECB page is the authority a reader should
    land on, and a raw JSON endpoint makes a poor source card.

    Each pair gets its own `#base-target` fragment on that page — see
    `_rate_source_url` for why.
    """
    url = f"https://api.frankfurter.dev/v1/latest?symbols={target_currency}&base={base_currency}"
    response = httpx.get(url)
    data = response.json()

    rate = (data.get("rates") or {}).get(target_currency)
    if rate is not None:
        n = register_citation(
            title=f"{base_currency} to {target_currency} exchange rate",
            url=_rate_source_url(base_currency, target_currency),
            content=_rate_summary_text(
                data.get("base", base_currency), target_currency, rate, data.get("date", "")
            ),
        )
        if n is not None:
            data["n"] = n
    return data


# print(get_realtime_currency_rate("USD", "CNY"))
