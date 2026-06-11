---
name: deep-research
description: Use for deep, multi-faceted research requests — when the user asks to "research", "deeply analyze", "investigate", compare many options, or wants a comprehensive overview of a topic. Runs a plan → gather → synthesize → report workflow. Do NOT use for quick factual lookups.
---

# Deep Research

Use this when a question is broad or complex enough to need systematic
investigation — not a quick factual lookup.

## Workflow
1. **Plan** — call `write_todos` to lay out 3–6 concrete research steps
   (e.g. "find official docs", "compare alternatives", "gather real examples").
2. **Gather** — work through the todos. Use `google_search`, then `load_web_page`
   to read the best sources. Update each todo (in_progress → completed) via
   `write_todos` as you go.
3. **Think & iterate** — after each batch, reflect on what's missing and revise the
   todo list.
4. **Report** — synthesize everything into a long-form report using the
   `report-writing` skill. Embed charts (see the `charting` skill) wherever data is
   clearer shown than told.

## Budget — important
- Keep it focused: aim for roughly **6–12 quality sources total**, not dozens.
  A handful of strong sources beats exhaustive searching.
- As soon as you have enough to answer well (or two searches stop adding anything
  new), STOP gathering and write the report. Do not keep searching.
- You MUST finish by writing the `<report>…</report>` block (see the
  `report-writing` skill) followed by a short chat reply — never run out of steps
  still searching.

## Rules
- Cite sources naturally in the report.
- Depth and accuracy over length — do not pad.
- End your chat reply with 1–3 sentences pointing the user to the report panel.
