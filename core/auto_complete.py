from core.llm import llama3_1_8b
from core.utils.redis_cache import l1cache

schema = {
    "texts": "list<string>",
}

model = llama3_1_8b.with_structured_output(schema=schema, method="json_mode")

system_prompt = """
<system>
You are a query autocomplete engine for a chatbot.

Task:
Generate 3 short natural continuations for the user's query.

Rules:
- Continue the text directly.
- Do not repeat the input.
- Keep completions short and natural.
- The result should look like something a user would type next.
- Include any needed leading space or punctuation.
- Do not explain anything.

Output JSON:
{ "texts": ["...", "...", "..."] }
</system>
"""


# @l1cache(ttl=3600 * 24 * 90)
def auto_complete(text: str) -> list[str]:
    return []
    messages = [
        ("system", system_prompt),
        ("human", text),
    ]

    result = model.invoke(messages)

    completions = result["texts"]

    # simple append
    return [text + c for c in completions]
