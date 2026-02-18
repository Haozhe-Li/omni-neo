from langchain_groq import ChatGroq
import os
import dotenv

dotenv.load_dotenv()

get_title_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)

get_title_llm_system_prompt = """
You are an expert in naming a title for a given query. 

You will be given a query from user, and you need to generate a title for it.

Output format:
- Title as a short phrase (no more than 5 words)
- Use the SAME language as the query
- return the title only!
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
    res = get_title_llm.invoke(messages).content
    return res
