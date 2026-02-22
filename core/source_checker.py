from core.utils.utils import smart_split


def check_source(source: dict, text_selection: str):
    """
    Checks if the text_selection exists in the source documents using a sliding window algorithm.
    Returns the matching source if the similarity score is > 0.5.
    """
    sources = source.get("sources", [])
    if not sources:
        return {}

    selection_tokens = smart_split(text_selection.lower())
    if not selection_tokens:
        return {}

    n = len(selection_tokens)
    selection_set = set(selection_tokens)

    best_score = 0.0
    best_result = None

    for item in sources:
        content = item.get("content", "")
        content_tokens = smart_split(content.lower())

        if len(content_tokens) < n:
            continue

        doc_max_score = 0.0

        for i in range(len(content_tokens) - n + 1):
            window = content_tokens[i : i + n]
            window_set = set(window)

            intersection = len(selection_set.intersection(window_set))
            union = len(selection_set.union(window_set))

            score = intersection / union if union > 0 else 0.0

            if score > doc_max_score:
                doc_max_score = score

            if doc_max_score == 1.0:
                break

        if doc_max_score > best_score:
            best_score = doc_max_score
            best_result = item

    # print(f"Best score: {best_score}")
    if best_score > 0.1 and best_result:
        return {
            "title": best_result["title"],
            "url": best_result["url"],
            "content": best_result["content"],
        }

    return {}
