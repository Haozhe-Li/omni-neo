from core.tools.web_page_reader import load_web_page
from core.tools.web_search import tavily_search
import os
from langchain_groq import ChatGroq
from core.utils.redis_cache import l1cache
import time
from tavily import TavilyClient
from typing import Literal

from concurrent.futures import ThreadPoolExecutor, TimeoutError

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

@l1cache(ttl=3600 * 24 * 10)
def verify_claim(fact: str) -> str:
    """Check a claim using Tavily API.

    Args:
        fact (str): The claim to check. Should be a concise and short statement, no more than 10 words.

    Returns:
        str: The answer to the claim.
    """
    TIMEOUT_SECONDS = 10

    def _verify():
        return tavily_client.search(
            query=fact,
            include_answer="advanced",
        ).get("answer")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_verify)
        try:
            return future.result(timeout=TIMEOUT_SECONDS)
        except TimeoutError:
            return f"Verification timed out after {TIMEOUT_SECONDS} seconds."
        except Exception as e:
            return f"Verification failed: {str(e)}"
