import re


def smart_split(text):
    pattern = r"[a-zA-Z0-9']+|[\u4e00-\u9fff]|[^\w\s]"
    result = re.findall(pattern, text)
    return [item for item in result if item.strip()]
