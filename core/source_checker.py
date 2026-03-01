from rapidfuzz import fuzz


def check_source(source: dict, text_selection: str):
    """
    使用 RapidFuzz 进行极速的局部字符串匹配。
    """
    sources = source.get("final_sources", [])
    if not sources or not text_selection:
        return {}

    best_score = 0.0
    best_result = None

    # 统一小写，提升匹配准确率
    text_selection_lower = text_selection.lower()

    for item in sources:
        content = item.get("content", "")
        if not content:
            continue

        # partial_ratio 会在 content 中寻找与 text_selection 最匹配的滑动窗口（子串）
        # 返回分数范围是 0.0 到 100.0
        score = fuzz.partial_ratio(text_selection_lower, content.lower())

        if score > best_score:
            best_score = score
            best_result = item

        # 如果找到完美匹配，直接跳出循环
        if best_score == 100.0:
            break

    # print(f"Best score: {best_score}")

    # rapidfuzz 的 100 分制，50 相当于你的 0.5
    if best_score > 35.0 and best_result:
        return {
            "title": best_result["title"],
            "url": best_result["url"],
            "content": best_result["content"],
        }

    return {}
