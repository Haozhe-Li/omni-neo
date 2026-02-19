from langchain_groq import ChatGroq
import os
import re
import dotenv
from pydantic import BaseModel, Field


dotenv.load_dotenv()

auto_select_model_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


class Decision(BaseModel):
    is_smart: bool = Field(
        description="True if query requires deep thinking, coding, or research. False if simple."
    )


model_with_structure = auto_select_model_llm.with_structured_output(Decision)
llm_system_prompt = (
    "Decide if the query requires deep reasoning (True) or is simple (False)."
)


def smart_split(text):
    pattern = r"[a-zA-Z0-9']+|[\u4e00-\u9fff]|[^\w\s]"
    result = re.findall(pattern, text)
    return [item for item in result if item.strip()]


def naive_selector(query: str) -> str:
    tokens = smart_split(query)
    q_lowered = query.lower()
    fast_indicators = ["quick answer", "快速回答"]
    for indicator in fast_indicators:
        if indicator.lower() in q_lowered:
            return "fast"
    research_indicators = [
        "研究",
        "分析",
        "深度",
        "详细",
        "解释",
        "deep",
        "research",
        "analysis",
        "explain",
        "compare",
        "对比",
        "评测",
        "评估",
        "review",
        "复杂",
        "canvas",
    ]
    for keyword in research_indicators:
        if keyword.lower() in q_lowered:
            return "smart"
    if len(tokens) > 20:
        return "smart"
    if len(tokens) < 5:
        return "fast"
    return "unknown"


def get_auto_select_model(query: str) -> str:
    res = naive_selector(query)
    if res != "unknown":
        return res
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
        res = model_with_structure.invoke(messages).is_smart
    except Exception as e:
        print(e)
        print("fall to light")
        res = False
    return "smart" if res else "fast"


if __name__ == "__main__":
    import time

    start_time = time.time()
    print(get_auto_select_model("给我介绍一下2026年的超级碗，研究一下"))
    print(time.time() - start_time)
