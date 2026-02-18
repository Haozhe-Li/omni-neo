from langchain_groq import ChatGroq
import os
import dotenv
from pydantic import BaseModel, Field
from typing import Literal, Annotated

dotenv.load_dotenv()

auto_select_model_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


class Model(BaseModel):
    model: Annotated[Literal["fast", "smart"], Field(description="The model to use")]


model_with_structure = auto_select_model_llm.with_structured_output(Model)
llm_system_prompt = """
You are an expert in choosing between two models, fast and smart.

You need to understand user's query and decide whether it needs deep thinking or not.

If the query is simple and straightforward, choose fast.
If the query is complex and needs deep thinking, choose smart.

Output:
You can only output class Model, with a model field, has to be in ["fast", "smart"].
"""


def naive_selector(query: str) -> str:
    if len(query) > 50:
        return "smart"
    keywords = [
        "研究",
        "分析",
        "深度",
        "详细",
        "解释",
        "Deep",
        "Research",
        "Analysis",
        "Explain",
        "compare",
        "对比",
    ]
    q_lowered = query.lower()
    for keyword in keywords:
        if keyword.lower() in q_lowered:
            return "smart"
    return "fast"


def get_auto_select_model(query: str) -> str:
    return naive_selector(query)
    # messages = [
    #     (
    #         "system",
    #         llm_system_prompt,
    #     ),
    #     (
    #         "human",
    #         f"The query is: {query}",
    #     ),
    # ]
    # res = model_with_structure.invoke(messages).model
    # return res


if __name__ == "__main__":
    import time

    start_time = time.time()
    print(get_auto_select_model("给我介绍一下2026年的超级碗，研究一下"))
    print(time.time() - start_time)
