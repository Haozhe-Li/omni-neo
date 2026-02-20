from langchain_groq import ChatGroq
import os
import dotenv
from functools import lru_cache

dotenv.load_dotenv()

rewriter_llm = ChatGroq(
    model="meta-llama/llama-prompt-guard-2-86m",
    api_key=os.getenv("GROQ_API_KEY"),
)


@lru_cache(maxsize=128)
def is_harmful(query: str) -> bool:
    try:
        messages = [
            (
                "human",
                query,
            ),
        ]
        res = float(rewriter_llm.invoke(messages).content)
        return res > 0.5
    except Exception as e:
        print(e)
        return False


# if __name__ == "__main__":
#     print(is_harmful("What is the capital of France?"))
