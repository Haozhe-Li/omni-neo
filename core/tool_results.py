"""What a finished tool call retrieved, as the trace shows it.

The chat stream used to say only that a tool was *called*. The frontend's step
trace wants the other half — "Searching ‘x’" followed by the pages that search
came back with — so each finished retrieval tool is reduced here to a short list
of `{title, url}` and sent as a `tool_result` event (see core/stream.py).

Read from the tool's own return value, not the citation registry: the registry
only reports URLs that are *new* to the thread, so a second search overlapping
the first would show up with a fraction of its results. What the agent was
actually handed is what the user should see it read — which also means junk
sources, withheld from the agent, stay out of the trace.
"""
from __future__ import annotations

import json
from typing import Any, Callable

# Most pages one step reports. A search asks for at most 10.
MAX_RESULTS = 10


def _payload(content: Any) -> Any:
    """A ToolMessage's content as Python data (None if it isn't JSON)."""
    if isinstance(content, list) and content and all(
        isinstance(part, dict) and part.get("type") == "text" for part in content
    ):
        # Content blocks: `[{"type": "text", "text": "<json>"}]`. Any other list
        # is already the tool's own result list.
        content = "".join(part.get("text", "") for part in content)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None
    return content


def tool_result_pages(
    content: Any,
    citation_by_n: Callable[[int], dict | None] | None = None,
) -> list[dict]:
    """The `{title, url}` pages a tool's return value names, in order.

    Handles the three shapes retrieval tools return: a list of results
    (`web_search`), one result dict (`fetch_url`), and a payload with only an
    `n` citation number (`stock_search`, `currency_convert`), which is resolved
    through `citation_by_n` to the source it registered.
    """
    data = _payload(content)
    items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []

    pages: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or ""
        title = item.get("title") or ""
        if not url and citation_by_n and isinstance(item.get("n"), int):
            record = citation_by_n(item["n"]) or {}
            url, title = record.get("url") or "", record.get("title") or title
        if not url or url in seen:
            continue
        seen.add(url)
        pages.append({"title": title, "url": url})
        if len(pages) >= MAX_RESULTS:
            break
    return pages
