---
name: deep-research
description: Use for deep, multi-faceted research requests — when the user asks to "research", "deeply analyze", "investigate", compare many options, or wants a comprehensive overview of a topic. Runs a plan → gather → synthesize → report workflow. Do NOT use for quick factual lookups.
---

# Deep Research

Use this skill when a question is broad or complex enough to need systematic
investigation across multiple sources — not a quick factual lookup.

---

## Workflow

### Step 0 — Clarify (if needed)

Before planning, ask yourself: *Do I know enough to research the right thing?*

If the request is ambiguous — scope too broad, key parameters missing (region,
timeframe, use case, budget, audience…), or multiple valid interpretations
exist — **stop and use the `question` skill** to ask the user one focused
clarifying question. End the turn there. Do not guess and proceed.

Once the user answers, resume from Step 1 with their answer in hand.

**Skip this step** if the request is specific enough to plan without
ambiguity. Do not ask questions just to seem thorough.

---

### Step 1 — Plan

Call `write_todos` before any search. Structure the plan as a research arc:

1. **Orient** — one broad search to understand the landscape (key players,
   sub-topics, controversies, timeframe). This shapes everything else.
2. **Dive** — one todo per major sub-topic or angle (typically 3–5 dives).
   Each should be narrow enough to answer in 2–3 searches.
3. **Compare / contrast** — if the task involves options, tradeoffs, or
   competing claims, add an explicit synthesis todo here.
4. **Report** — the final todo is always writing the report.

Good example for "compare electric SUVs in 2024":
- Orient: survey the EV SUV market
- Gather: Tesla Model Y specs & reviews
- Gather: Ford Mustang Mach-E specs & reviews
- Gather: Hyundai Ioniq 5 specs & reviews
- Gather: charging infrastructure & real-world range data
- Compare: price / range / charging / reliability tradeoffs
- Report

Aim for **6–10 todos**. Fewer is fine if the scope is tight.

---

### Step 2 — Gather

Work through the todos in order.

**Searching:**
- Start each sub-topic with one targeted `google_search`.
- Scan the result titles and snippets. Only `load_web_page` on results that
  are clearly relevant and not paywalled.
- Read 2–4 pages per sub-topic — stop when two consecutive pages add nothing
  new to that sub-topic.
- **Hard cap: at most 2 searches per todo item** (one initial + one reformulation if the first is weak). Never run a third search on the same sub-topic — move on with what you have.

**Computing:**
- When the research involves numbers — statistics, comparisons, unit conversions,
  derived metrics, or verifying a quantitative claim — use `run_python` rather
  than approximating in prose. Accurate numbers make the report more credible.
- Keep scripts self-contained; include results in the report as plain text or a
  table. Do not attempt to plot inside `run_python` — use the charting skill for
  that.

**Source quality:**
- Prefer primary sources (official docs, studies, manufacturer specs) and
  established outlets over aggregator summaries.
- When sources contradict each other, note the disagreement explicitly — do
  not silently pick one; the report should surface it.

**Todo hygiene (strict):**
- Mark a todo `in_progress` immediately before you start its work.
- The very next action after finishing a todo's work must be a `write_todos`
  call marking it `completed` — before any other tool call or text.
- Never carry a finished todo as uncompleted into the next step.

---

### Step 3 — Reflect (lightweight)

After every 2–3 dives, ask: *Is there a significant angle I planned to cover
that the sources haven't addressed yet?* If yes, add a new todo and continue.
If no, proceed.

This reflection is a quick gut-check — not a reason to keep searching. If you
have enough to write a strong report, stop gathering.

---

### Step 4 — Report

When all gather todos are complete:

1. Mark the gather todos `completed`.
2. Mark the report todo `in_progress`.
3. Write the full report using the `report-writing` skill (follow that skill's
   structure and formatting rules exactly).
4. Embed charts using the `charting` skill wherever data is clearer shown than
   told — trends, comparisons, distributions.
5. Mark the report todo `completed`.
6. Follow the report with a short chat reply (2–4 sentences) summarizing the
   key finding and pointing the user to the report panel.

Every todo must be `completed` before you write the final chat reply.

---

## Budget

- **Sources**: 6–12 quality sources total. A handful of strong sources beats
  exhaustive searching.
- **Stopping signal**: stop gathering the moment two consecutive searches on
  the same sub-topic yield no new facts or perspectives. Do not search for
  completeness — search for understanding.
- **Hard stop**: if you are approaching the tool-call limit, skip remaining
  gather todos and write the report with what you have. A partial but honest
  report beats running out of steps mid-search.

---

## Rules

- Cite sources naturally in prose — author, outlet, or document name. No
  bare URLs.
- Surface contradictions between sources; do not silently resolve them.
- Depth and accuracy over length — do not pad with filler paragraphs.
- Never end the turn still searching. Always finish with the report +
  chat reply.
