from langchain_community.document_loaders import SpiderLoader
import os
from core.utils.redis_cache import l1cache
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor, TimeoutError


web_page_summarizer = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)

web_page_summarizer_system_prompt = """
You are an expert at skimming long webpages and extracting only the most relevant information for a given purpose.

Make sure you answer the purpose.

Output format:
- Key points from the webpage
- Evidence (optional but preferred when available)
"""


@l1cache(ttl=3600 * 24 * 90) # Cache for 90 days since historical web page content doesn't change
def load_web_page(url: str) -> str:
    """Load a web page and return its content.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        str: The loaded web page content as a string.
    """
    try:
        loader = SpiderLoader(
            api_key=os.getenv("SPIDER_API_KEY"),
            url=url,
            mode="scrape",
        )
        documents = loader.load()
    except Exception:
        return f"URL: {url}\n\nContent:\nFailed to load the web page."
    if not documents:
        return f"URL: {url}\n\nContent:\nNo content found on the web page. This could happen if the firewall blocks the request or the page is empty."
    return f"URL: {url}\n\nContent:\n{documents[0].page_content}"


def get_full_text(
    url: str,
):
    """Get the full text of a web page.

    Args:
        url (str): The URL of the web page to load.

    Returns:
        str: The full text of the web page.
    """
    print(f"Loading web page: {url}")
    return load_web_page(url)[:2000]


def skimming_web_page(url: str, purpose: str):
    """Skim a web page and return the most relevant information based on the purpose.

    Args:
        url (str): The URL of the web page to skim.
        purpose (str): The purpose of skimming the web page. Usually a question.

    Returns:
        str: The most relevant information from the web page.
    """
    document_content = load_web_page(url)[:1000]
    messages = [
        (
            "system",
            web_page_summarizer_system_prompt,
        ),
        (
            "human",
            f"The purpose is: {purpose}\nThe web page content is: {document_content}",
        ),
    ]
    res = web_page_summarizer.invoke(messages).content
    return res


def skimming_web_pages(urls: list[str], purpose: str) -> list:
    """Skim through multiple web pages in parallel and return the most relevant information based on the purpose.

    Args:
        urls (list[str]): A list of URLs of the web pages to skim.
        purpose (str): The purpose of skimming the web pages. Usually a question.

    Returns:
        list[str]: A list of the most relevant information from the web pages.
    """

    TIMEOUT_SECONDS = 10

    def _skim(url: str) -> dict:
        return {
            "url": url,
            "content": skimming_web_page(url, purpose),
        }

    res = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_skim, url): url for url in urls}
        for future in futures:
            url = futures[future]
            try:
                res.append(future.result(timeout=TIMEOUT_SECONDS))
            except (TimeoutError, Exception):
                res.append(
                    {
                        "url": url,
                        "content": f"[TIMEOUT] Skimming this page exceeded {TIMEOUT_SECONDS}s. Usually, this means this url blocks requests. DO NOT retry this URL.",
                    }
                )
    return res
