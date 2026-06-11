---
name: report-writing
description: How to publish a long-form report into the side panel with the write_report tool. Use ONLY when the user explicitly asks for a report/document/write-up, during deep research, or when a complete answer would clearly exceed ~500 words and reads better as a structured document. For ordinary questions, answer directly in chat instead — do not write a report.
---

# Report Writing

## When to write a report
Write a report ONLY if at least one of these holds:
- The user explicitly asked for a report, document, write-up, or deep analysis.
- You are running the deep-research workflow.
- A complete, high-quality answer would clearly exceed ~500 words AND benefits from
  being a structured document.

Otherwise, answer normally in chat. Most questions do NOT need a report — do not
reach for one by default.

## How
- Call `write_report(title, markdown)`. The body renders in a side panel; do NOT
  also paste the full report into the chat.
- Structure it: `##` / `###` headings, lists, tables. Use `$...$` / `$$...$$` for math.
- You may embed charts directly in the report markdown using ```echarts fenced
  blocks (see the charting skill) — they render inline inside the report.
- After calling `write_report`, you MUST still write a short chat reply (1–3
  sentences) introducing it and pointing to the panel. Never end your turn with an
  empty chat message — a tool call alone is not a reply.
