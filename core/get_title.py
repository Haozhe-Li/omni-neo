from langchain_groq import ChatGroq
from langsmith import tracing_context
import os
import dotenv

dotenv.load_dotenv()

get_title_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

get_title_llm_system_prompt = """
You are a title generator for chat/query titles.

Task:
Given a single user query, produce a very short topic label/title.

Rules:
- Output ONLY the title, nothing else (no quotes, no punctuation, no explanation).
- Use the SAME language as the query (do not translate).
- Keep it extremely short: 1–3 words maximum; for CJK, 1–4 characters preferred.
- Express only the core topic (e.g., entity + key concept); remove filler like "how", "why", "this year", "please".
- Do not create a full sentence or question; do not add information not in the query.

Examples:
Query: How is Tesla's sales this year?
Title: Tesla sales

Query: Can I reset my iPhone?
Title: iPhone reset

Query: 比特币价格会涨吗
Title: 比特币价格

Now generate the title for the given query.
"""


def get_title(query: str) -> str:
    messages = [
        (
            "system",
            get_title_llm_system_prompt,
        ),
        (
            "human",
            f"The query is: {query}",
        ),
    ]
    with tracing_context(project_name="title"):
        res = get_title_llm.invoke(messages).content
    return res
