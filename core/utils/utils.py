import re
from core.utils.data_model import Personalization


def smart_split(text):
    pattern = r"[a-zA-Z0-9']+|[\u4e00-\u9fff]|[^\w\s]"
    result = re.findall(pattern, text)
    return [item for item in result if item.strip()]


def format_personalization(personalization: Personalization) -> str:
    result = ""
    if personalization.response_language:
        result += f"Response Language: {personalization.response_language}\n"
    if personalization.memories:
        result += "Memories (Not all memories might be useful for the current task):\n"
        for i, memory in enumerate(personalization.memories):
            result += f"{i + 1}. {memory}\n"
    return result
