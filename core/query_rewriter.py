from langchain_openai import ChatOpenAI
import os
import dotenv


dotenv.load_dotenv()

rewriter_llm = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    api_key=os.getenv("OPENAI_API_KEY"),
)


rewriter_system_prompt = """
### Role
You are an expert in filtering information and combing them.

### Input
- User Query
- User Personalization: Including response language, memories etc.

### Rules
- You need to analyze the personalization information, combine it into user query. 
- You shuold only take the useful information from the personalization information.

### Example

User Query: "What is the weather like today?"
User Personalization: "Response Language: Chinese. Memories: I am a software engineer. I am working on a project about AI."
Rewritten Query: "What is the weather like today? Please respond in Chinese."

### Output
- Rewritten Query Only
"""


def rewrite_query(query: str, personalization: str) -> str:
    input_query = f"The query is: {query}. \n{personalization}"
    messages = [
        (
            "system",
            rewriter_system_prompt,
        ),
        (
            "human",
            input_query,
        ),
    ]
    res = rewriter_llm.invoke(messages).content
    return res
