import re
from core.utils.data_model import Personalization


def smart_split(text):
    pattern = r"[a-zA-Z0-9']+|[\u4e00-\u9fff]|[^\w\s]"
    result = re.findall(pattern, text)
    return [item for item in result if item.strip()]


def format_system_reminder(personalization: Personalization | None) -> str:
    """Format the `<system_reminder>` block.

    Always non-empty: the identity line is the point of the block even for an
    anonymous turn with no client-supplied personalization at all. It restates
    who the assistant is right next to the user's question, where a long system
    prompt's own self-description is far away and easily diluted by a thread of
    tool output.

    Field order is deliberate for prompt-cache prefix matching (Groq/Gemini
    both cache on shared prefixes): the identity line is fixed, and
    `response_language`/`user_location` are stable across a user's requests,
    while `user_local_datetime` changes on every single turn — so the
    ever-changing field goes last, keeping the stable prefix intact instead of
    splitting it in two.
    """
    result = "You are Omni. If the user asks who you are, say you are Omni.\n"
    if not personalization:
        return result
    if personalization.response_language:
        result += f"Response Language: {personalization.response_language}\n"
    if personalization.user_location:
        result += f"User Location: {personalization.user_location}\n"
    if personalization.user_local_datetime:
        result += f"User Local Date Time: {personalization.user_local_datetime}\n"

    return result


def format_user_memory(memory_content: str | None) -> str:
    """Format the user's server-persisted long-term memory.

    Returns the body of the `<user_memory>` block that
    `build_message_content` wraps, or "" when there's nothing stored. Kept
    separate from `format_system_reminder` on two counts: memory is fetched
    from Postgres by the router (async, keyed by user_id) rather than supplied
    by the client, and it is a distinct block in the user message — the prompt
    tells the model to treat memory as background it may ignore, which is a
    weaker claim than the one personalization makes.
    """
    if not memory_content:
        return ""
    return (
        "Long-term facts about this user. Not all of it is relevant to the "
        f"current turn.\n{memory_content}"
    )
