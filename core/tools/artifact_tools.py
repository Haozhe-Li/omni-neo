"""Artifact tools for the pro agent.

These two tools do not perform any external I/O. Instead they validate the
model-supplied payload and return a *sentinel-wrapped* JSON string. The chat
stream handler (`core/stream.py`) detects the sentinel on the resulting
`ToolMessage` and forwards a clean, validated `artifact` / `report` SSE event to
the frontend. The short human-readable suffix is what the model sees as the tool
result, keeping its context lean.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.tools import tool

# Sentinel keys recognised by core.stream.run_agent_stream
ARTIFACT_SENTINEL = "__omni_artifact__"
REPORT_SENTINEL = "__omni_report__"


@tool
def render_chart(title: str, option: dict[str, Any]) -> str:
    """Render an interactive chart in the user's side panel.

    Use this whenever a chart communicates the answer better than prose
    (trends, comparisons, distributions, proportions).

    Args:
        title: A short, human-readable chart title (max ~8 words).
        option: A complete Apache ECharts `option` object as JSON. It MUST be a
            valid ECharts configuration containing at least a `series` field, and
            should include `xAxis`/`yAxis` for cartesian charts. Do NOT include
            any JavaScript functions or callbacks — JSON-serialisable values only.

    Returns:
        A confirmation string. The chart itself is streamed to the UI.
    """
    if not isinstance(option, dict) or not option:
        return (
            "Error: `option` must be a non-empty ECharts option object. "
            "Provide a JSON object with at least a `series` field."
        )
    if "series" not in option:
        return (
            "Error: ECharts `option` is missing the required `series` field. "
            "Add a `series` array describing the data to plot."
        )
    try:
        # Reject non-JSON-serialisable payloads (e.g. functions) early.
        json.dumps(option)
    except (TypeError, ValueError) as exc:
        return f"Error: `option` is not JSON-serialisable ({exc}). Use plain JSON values only."

    payload = {
        ARTIFACT_SENTINEL: {
            "id": f"chart_{uuid.uuid4().hex[:12]}",
            "title": title.strip() or "Chart",
            "kind": "echarts",
            "spec": option,
        }
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def write_report(title: str, markdown: str) -> str:
    """Publish a long-form report into the user's side panel.

    Use this for substantial, structured deliverables (research write-ups,
    multi-section analyses, guides). In your chat reply, briefly introduce the
    report (e.g. "Here's a detailed report:") and call this tool — the full
    document is rendered side-by-side rather than inline.

    Args:
        title: A concise report title (max ~10 words).
        markdown: The full report body in GitHub-flavoured Markdown. Use `$...$`
            / `$$...$$` for math. Do not wrap the whole thing in a code fence.

    Returns:
        A confirmation string. The report itself is streamed to the UI.
    """
    if not isinstance(markdown, str) or not markdown.strip():
        return "Error: `markdown` must be a non-empty report body."

    payload = {
        REPORT_SENTINEL: {
            "id": f"report_{uuid.uuid4().hex[:12]}",
            "title": title.strip() or "Report",
            "content": markdown,
        }
    }
    return json.dumps(payload, ensure_ascii=False)
