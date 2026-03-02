import httpx
from core.utils.redis_cache import l1cache


@l1cache(ttl=60 * 60 * 12)
def get_realtime_currency_rate(base_currency: str, target_currency: str) -> dict:
    """
    Get the real-time exchange rate between two currencies.
    """
    url = f"https://api.frankfurter.dev/v1/latest?symbols={target_currency}&base={base_currency}"
    response = httpx.get(url)
    data = response.json()
    return data


# print(get_realtime_currency_rate("USD", "CNY"))
