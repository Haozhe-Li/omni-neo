---
name: report-writing
description: Stream a long-form report. Always use this when the answer involves research, structured analysis, or output could be longer than 400 words.
---

# Report Writing

## When to write a report
Write a report ONLY if at least one of these holds:
- The user explicitly asked for a report, document, write-up, or deep analysis.
- You are running the deep-research workflow.
- A complete, high-quality answer would clearly exceed ~500 words AND benefits from
  being a structured document.

Otherwise, answer normally in chat as always. Most questions do NOT need a report — do not
reach for one by default.

## How
A report is NOT a tool. You stream it inline by wrapping the full document in a
`<report>…</report>` block — exactly the way a chart is an inline ```echarts
fence. The frontend pulls the block out of your answer and renders it live in the
side reader as the words arrive, so just write it out normally.

```
Here's the report you asked for — it's opening in the reader on the right.

<report title="State of EV Batteries, 2026">
## Overview
…the full report body in GitHub-flavoured Markdown…

## Cost Trends
…
</report>
```

## Rules
- Put a concise `title="…"` (max ~10 words) on the opening tag. It becomes the
  reader's tab + heading. If you omit it, the first heading is used.
- Write exactly ONE `<report>` block per turn, and never nest one inside another.
- The body is normal GFM: `##` / `###` headings, lists, tables, and `$…$` /
  `$$…$$` for math. Do NOT wrap the whole report in a code fence.
- You may embed charts directly in the report using ```echarts fenced blocks (see
  the charting skill) — they render inline inside the report.
- Outside the block, write a short chat reply (1–3 sentences) that introduces the
  report and points to the reader. The report body lives ONLY inside the tags —
  do not also paste it into the chat. Never end your turn with the block alone.
- **Citations, restated because reports are where this slips:** every
  `google_search`/`load_web_page` result carries a `n` — cite it, in ASCII
  `[n]` (never full-width 【】), whenever a paragraph uses it, no exceptions.
  Put the [n]s (stacked, e.g. [1][2]) at the very END of the paragraph that
  relies on them, not mid-sentence and not scattered after every clause. Do
  NOT collect them into a references/bibliography list after the conclusion —
  a report has many sections and many sources, and it's tempting to punt
  citing until the end like an academic paper, but a reader looking at
  section 3 needs to see right there which sources back it. Inline,
  per-paragraph, every time — same rule as normal chat, just easier to forget
  across a long document.
