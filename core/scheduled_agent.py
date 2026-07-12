"""Run one scheduled-research-task turn to completion and extract its output.

Unlike the interactive `/chat` path (core/stream.py), this is invoked once via
`agent.ainvoke()` — there is no SSE listener, so there is nothing to stream to.
The agent still runs inside the normal LangGraph checkpointer against the
thread_id it's given, so the resulting thread is a completely ordinary
conversation afterward (the user can open it and keep chatting).

The agent's `response_format` (see core/agent.py's `ScheduledReportOutput` /
`build_scheduled_agent`) forces the model's final turn through Gemini's native
structured output, so `{title, summary, report}` come back pre-parsed in
`result["structured_response"]` — no regex-scraping of hand-written tags out
of free text.

The frontend's history renderer (`lib/report-parser.ts`'s `parseReports`) does
still look for a literal `<report>…</report>` block in stored message content
— live-streamed and replayed-from-history messages share that one code path
(components/chat-view.tsx) — so `_tagged_content` below reconstructs that
wrapper purely for what gets persisted to Postgres, not for how the model
produces its output.
"""

from __future__ import annotations

import logging

from core.agent import get_scheduled_agent, SCHEDULED_SKILL_FILES
from core.database.db_user_threads import upsert_thread_messages
from core.utils.citations import reset_citation_registry, all_citations

logger = logging.getLogger(__name__)


class ScheduledRunError(Exception):
    pass


def _tagged_content(title: str, summary: str, report: str) -> str:
    """Reconstruct the `<summary>`/`<report>` wrapper the chat-thread history
    renderer expects (see module docstring) — storage format only."""
    safe_title = title.replace('"', "'")
    return f'<summary>{summary}</summary>\n\n<report title="{safe_title}">\n{report}\n</report>'


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

    structured = result.get("structured_response")
    if structured is None:
        # Most likely cause: the run hit ToolCallLimitMiddleware's cap before
        # the model ever reached a tool-call-free turn to finalize on.
        raise ScheduledRunError(
            "Agent did not produce a structured report (possibly hit the tool-call limit)."
        )

    title = (structured.title or "").strip() or "Scheduled Research Report"
    summary = (structured.summary or "").strip()
    report_body = (structured.report or "").strip()
    if not report_body:
        raise ScheduledRunError("Agent produced an empty report.")
    if not summary:
        raise ScheduledRunError("Agent produced an empty summary.")

    parsed = {"title": title, "summary": summary, "report": report_body}

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
            {
                "role": "assistant",
                "content": _tagged_content(title, summary, report_body),
                "sources": sources,
            },
        ],
    )

    return {**parsed, "sources": sources}
