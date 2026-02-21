from langchain_groq import ChatGroq
import os
import dotenv
from core.utils.data_model import Memories


dotenv.load_dotenv()

update_memories_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
)


model_with_structure = update_memories_llm.with_structured_output(Memories)
llm_system_prompt = """
You are a memory manager AI. Based on the user's past 3 queries and current memories, update their profile.

Extract or update the following IF there is new, persistent information:
1. User Profile: Name,Job, traits, or facts.
2. Current Focus: Ongoing projects or interests.
3. Interaction Style: Preferred tone, format, or conciseness.
4. Avoid Topics: Disliked subjects or formats.

Rules:
- Be conservative. Do not update for simple one-off questions.
- Write in friendly and natural language, as user could see it.
- Merge new insights seamlessly. Do NOT delete still-relevant older memories.
- If no update is needed, simply return the current memories exactly as they are.
"""


async def get_update_memories(query: str, current_memories: Memories) -> Memories:
    messages = [
        (
            "system",
            llm_system_prompt,
        ),
        (
            "human",
            f"Past 3 Queries: {query}   Current Memories: {current_memories}",
        ),
    ]
    try:
        res = await model_with_structure.ainvoke(messages)
        return res
    except Exception as e:
        print(e)
        return current_memories
