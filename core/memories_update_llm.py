from langsmith import tracing_context

from core.llm import update_memories_llm
from core.database.db_user_memories import MAX_MEMORY_CHARS

llm_system_prompt = f"""
You are a memory curator for a personal AI assistant. You maintain ONE
markdown document of durable, cross-conversation facts about a single user.

Update it only when the latest exchange reveals new, durable information:
identity/role, ongoing projects, stated preferences, communication style,
things to avoid, or an explicit "remember this" request.

Rules:
- Be conservative. Do not add anything for one-off, trivial exchanges.
- Organize as short markdown sections (e.g. "## Profile", "## Preferences",
  "## Current Focus"). Merge new facts into existing sections instead of
  appending duplicates.
- Prune or rewrite stale/contradicted entries — do not just keep growing.
- Keep the ENTIRE document under {MAX_MEMORY_CHARS} characters. If it's
  getting long, compress older entries rather than dropping newer ones.
- If nothing durable is worth remembering, return the current memory
  unchanged.
- Never store secrets, passwords, API keys, or other sensitive credentials.
- Output ONLY the updated markdown document. No commentary, no code fences.
"""


async def get_update_memories(current_memory: str, user_query: str, assistant_response: str) -> str:
    messages = [
        ("system", llm_system_prompt),
        (
            "human",
            f"Current memory:\n{current_memory or '(empty)'}\n\n"
            f"Latest exchange:\nUser: {user_query}\nAssistant: {assistant_response}",
        ),
    ]
    try:
        with tracing_context(project_name="memories"):
            res = await update_memories_llm.ainvoke(messages)
        return (res.content or "").strip()[:MAX_MEMORY_CHARS]
    except Exception as e:
        print(e)
        return current_memory
