from langchain_groq import ChatGroq
import os
from core.utils.utils import smart_split
from core.utils.redis_cache import l1cache
import dotenv


dotenv.load_dotenv()

auto_select_model_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

schema = {
    "smart": "boolean",
}
model_with_structure = auto_select_model_llm.with_structured_output(
    schema=schema, method="json_mode"
)
llm_system_prompt = (
    "You are a routing assistant. Your task is to analyze the user query "
    "and decide if it requires 'smart' (True) Deep Research or 'fast' (False) simple response. "
    "Select Deep Rearch when: the query shows a clear intent for a report, research, analysis in certain topic in-depth. "
    "Select Fast response when: the query is more casual, or ask for light weight real-time information. "
    "Output the result in the JSON, with the key 'smart' and boolean value only. "
)


def l1_naive_selector(query: str) -> str:
    tokens = smart_split(query)
    if len(tokens) < 5:
        return "fast"
    return "unknown"


@l1cache(ttl=3600 * 24 * 90)
def l2_llm_selector(query: str) -> str:
    messages = [
        (
            "system",
            llm_system_prompt,
        ),
        (
            "human",
            f"The query is: {query}",
        ),
    ]
    try:
        res = model_with_structure.invoke(messages)
        if res.get("smart"):
            return "smart"
        else:
            return "fast"
    except Exception as e:
        print(e)
        print("fall to light")
        return "fast"


def get_auto_select_model(query: str) -> str:
    res = l1_naive_selector(query)
    if res != "unknown":
        return res
    return l2_llm_selector(query)
