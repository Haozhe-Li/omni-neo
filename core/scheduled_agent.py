"""Run one scheduled-research-task turn to completion and extract its output.

Unlike the interactive `/chat` path (core/stream.py), this is invoked once via
`agent.ainvoke()` — there is no SSE listener, so there is nothing to stream to.
The agent still runs inside the normal LangGraph checkpointer against the
thread_id it's given, so the resulting thread is a completely ordinary
conversation afterward (the user can open it and keep chatting).

The model's final message contains, inline, a `<summary>...</summary>` block
(taught in core/agent.py's `_SCHEDULED_ADDENDUM`) followed by a
`<report title="...">...</report>` block (taught in skills/report-writing) —
same "write it inline, pull it out after" convention the frontend normally
uses, just parsed here instead since there's no frontend in this path.
"""

from __future__ import annotations

import logging
import re

from core.agent import get_scheduled_agent, SCHEDULED_SKILL_FILES
from core.database.db_user_threads import upsert_thread_messages
from core.utils.citations import reset_citation_registry, all_citations

logger = logging.getLogger(__name__)

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_REPORT_RE = re.compile(r'<report(?:\s+title="([^"]*)")?\s*>(.*?)</report>', re.DOTALL)


class ScheduledRunError(Exception):
    pass


def _extract_output(text: str) -> dict:
    summary_m = _SUMMARY_RE.search(text)
    report_m = _REPORT_RE.search(text)

    if not report_m:
        raise ScheduledRunError("Agent produced no <report> block.")

    title = (report_m.group(1) or "").strip()
    report_body = report_m.group(2).strip()
    summary = summary_m.group(1).strip() if summary_m else ""

    if not summary:
        # Fall back to the first couple of sentences of the report rather than
        # failing the whole run over a missing (optional-looking) tag.
        plain = re.sub(r"[#*`_>-]", "", report_body)
        summary = " ".join(plain.split())[:400]

    if not title:
        heading_m = re.search(r"^#{1,2}\s+(.+)$", report_body, re.MULTILINE)
        title = heading_m.group(1).strip() if heading_m else "Scheduled Research Report"

    return {"title": title, "summary": summary, "report": report_body}


async def run_scheduled_task(thread_id: str, user_id: str, prompt: str) -> dict:
    """Run the scheduled agent to completion for one task firing.

    Persists the turn to Postgres (same shape as an ordinary chat turn) and
    returns {"title", "summary", "report", "sources"} for the caller to
    publish + email.
    """
    agent = get_scheduled_agent()
    if agent is None:
        raise ScheduledRunError("Scheduled agent is not initialized.")

    # Brand-new thread created just for this run (see core/routers/scheduled_tasks.py),
    # so this is always its first turn.
    reset_citation_registry(thread_id, turn=1)

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {
        "messages": [{"role": "user", "content": prompt}],
        "files": SCHEDULED_SKILL_FILES,
    }

    result = await agent.ainvoke(input_state, config=config)
    messages = result.get("messages", [])
    if not messages:
        raise ScheduledRunError("Agent returned no messages.")

    final = messages[-1]
    content = getattr(final, "content", "")
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = content or ""

    if not text:
        raise ScheduledRunError("Agent produced no text output.")

    parsed = _extract_output(text)

    # Same source of truth core/stream.py uses: the citation registry populated
    # by the retrieval tools as they ran, not re-parsed from raw ToolMessages.
    sources = [
        {
            "n": c.get("n"),
            "title": c.get("title", ""),
            "url": c.get("url", ""),
            "content": c.get("content", ""),
        }
        for c in all_citations()
        if c.get("n") is not None
    ]

    upsert_thread_messages(
        thread_id,
        user_id,
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text, "sources": sources},
        ],
    )

    return {**parsed, "sources": sources}
