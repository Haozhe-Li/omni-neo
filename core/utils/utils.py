import re
from core.utils.data_model import Personalization


def smart_split(text):
    pattern = r"[a-zA-Z0-9']+|[\u4e00-\u9fff]|[^\w\s]"
    result = re.findall(pattern, text)
    return [item for item in result if item.strip()]


def format_personalization(personalization: Personalization) -> str:
    if not personalization:
        return ""
    result = ""
    if personalization.response_language:
        result += f"Response Language: {personalization.response_language}\n"
    if personalization.user_local_datetime:
        result += f"User Local Date Time: {personalization.user_local_datetime}\n"
    if personalization.user_location:
        result += f"User Location: {personalization.user_location}\n"

    print(result)
    return result


def append_memory_context(personalization_str: str, memory_content: str | None) -> str:
    """Append the user's server-persisted long-term memory to the personalization block.

    Kept separate from format_personalization because memory is fetched from
    Postgres by the router (async, keyed by user_id), not supplied by the client.
    """
    if not memory_content:
        return personalization_str
    return personalization_str + (
        "\n\nUser Memory (long-term facts about this user; "
        f"not all of it may be relevant to the current task):\n{memory_content}\n"
    )
