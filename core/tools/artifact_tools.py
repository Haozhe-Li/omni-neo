"""Artifact tool for the pro agent.

`render_chart` does not perform any external I/O. It validates the
model-supplied payload and returns a *sentinel-wrapped* JSON string. The chat
stream handler (`core/stream.py`) detects the sentinel on the resulting
`ToolMessage` and forwards a clean, validated `artifact` SSE event to the
frontend. The short human-readable suffix is what the model sees as the tool
result, keeping its context lean.

Reports are NOT a tool: the agent streams them inline as a `<report>…</report>`
block (see the report-writing skill), exactly like charts are streamed inline as
```echarts fences. The frontend parses the block out of the answer stream and
renders it live in the side reader.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.tools import tool

# Sentinel key recognised by core.stream.run_agent_stream
ARTIFACT_SENTINEL = "__omni_artifact__"


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
