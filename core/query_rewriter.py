from langchain_groq import ChatGroq
import os
import dotenv


dotenv.load_dotenv()

rewriter_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
)

rewriter_system_prompt = """
### Role
You are an expert Query Rewriter. Your task is to transform raw user input into optimized, actionable instructions for an AI Agent.

### Rules
1. **Maintain Language**: Always respond in the same language as the user.
2. **Instructional Format**: Convert questions or fragments into clear, objective task statements (e.g., "What is X" -> "Explain the definition and core features of X").
3. **Refine & Clean**: Remove typos, emotional bias, and conversational filler while preserving the original intent.
4. **Precision**: Use domain-specific terminology where appropriate to reduce ambiguity.
5. **Output Only**: Return only the rewritten query without any preamble.
6. **Add Language Prompt**: Explicitly add "Please respond in [Language]." to the rewritten query.

### Examples
- "what is langchain" -> "Explain the core concepts and primary use cases of the LangChain framework. Please respond in English."
- "帮我查下nvidia最近的股价，感觉跌了" -> "查询英伟达（NVIDIA）最新的股价走势并分析近期波动原因。请用中文回答。"

### Output Format
Output ONLY the rewritten query, nothing else.
"""


def rewrite_query(query: str) -> str:
    messages = [
        (
            "system",
            rewriter_system_prompt,
        ),
        (
            "human",
            f"The query is: {query}",
        ),
    ]
    res = rewriter_llm.invoke(messages).content
    return res
