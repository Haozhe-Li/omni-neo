# import asyncio
import httpx

base_url = "https://shortly.haozheli.com"


def get_shorten_url(url: str) -> str | None:
    with httpx.Client() as client:
        response = client.post(base_url + "/api/shorten", json={"url": url})
        # check response status
        if response.status_code != 200:
            return None
        return response.json().get("shortUrl")
