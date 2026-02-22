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
    if personalization.memories:
        memories = personalization.memories
        has_any_memory = any(
            [
                memories.user_profile,
                memories.current_focus,
                memories.interaction_style,
                memories.avoid_topics,
            ]
        )
        if has_any_memory:
            result += (
                "Memories (Not all memories might be useful for the current task):\n"
            )
            if memories.user_profile:
                result += f"- User Profile: {memories.user_profile}\n"
            if memories.current_focus:
                result += f"- Current Focus: {memories.current_focus}\n"
            if memories.interaction_style:
                result += f"- Interaction Style: {memories.interaction_style}\n"
            if memories.avoid_topics:
                result += f"- Avoid Topics: {memories.avoid_topics}\n"
    if personalization.user_local_datetime:
        result += f"User Local Date Time: {personalization.user_local_datetime}\n"
    if personalization.user_location:
        result += f"User Location: {personalization.user_location}\n"

    print(result)
    return result
