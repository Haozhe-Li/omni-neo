from langchain_community.document_loaders import SpiderLoader
import os

from core.utils.redis_cache import l1cache


@l1cache(
    ttl=3600 * 24 * 90
)  # Cache for 90 days since historical web page content doesn't change
def load_web_page_spider(url: str) -> dict:
    """Load a web page and return its content.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        dict: The loaded web page content as a dictionary with URL and content keys.
    """
    try:
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_API_KEY"),
            url=url,
            mode="scrape",
        )
        documents = loader.load()
    except Exception:
        return {"url": url, "content": "Failed to load the web page.", "title": "Error"}
    if not documents:
        return {
            "url": url,
            "content": "No content found on the web page. This could happen if the firewall blocks the request or the page is empty.",
        }
    return {
        "url": url,
        "content": documents[0].page_content,
        "title": documents[0].metadata.get("title", "No title found"),
    }


def load_web_page(
    url: str,
):
    """Get the full text of a web page.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        dict: A dictionary containing the URL and the content of the web page.
    """
    return load_web_page_spider(url)
