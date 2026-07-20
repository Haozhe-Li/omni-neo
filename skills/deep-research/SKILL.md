---
name: deep-research
description: Use for deep, multi-faceted research requests — when the user asks to "research", "deeply analyze", "investigate", compare many options, or wants a comprehensive overview of a topic. Runs a plan → gather → synthesize → report workflow. Do NOT use for quick factual lookups.
---

# Deep Research

## Step 0 — Clarify

If the scope, angle, or goal is ambiguous, **immediately load the `ask question` skill** and emit a `<question>` block before proceeding. Ask only what is genuinely needed to research the right thing — do not ask for information you can reasonably assume. One focused `<question>` block with 1–3 questions is enough; do not stack multiple blocks.

Skip this step entirely if the request is already specific enough to plan directly.

---

## Step 1 — Plan

Structure the work as a research arc:

1. **Orient** — one broad search to map the landscape (key players, sub-topics, timeframe).
2. **Dive** — 3–5 major sub-topics or angles, each narrow enough to answer in 2–3 searches.
3. **Compare** — if the task involves options or tradeoffs, add an explicit synthesis pass.
4. **Report** — always last. **Important:** for this step, explicitly load the `charting` and `report-writing` skills!

Feel free to use `write_todos` to lay this out and track progress as you go — use your judgment on when it's actually helpful; it's not a box-checking exercise.

---

## Step 2 — Gather

**Searching:**
- One targeted `google_search` per sub-topic. `load_web_page` only on clearly relevant, non-paywalled results.
- Read 2–4 pages per sub-topic. Stop when two consecutive pages add nothing new.
- **Hard cap: 2 searches per sub-topic** (initial + one reformulation). Never a third — move on.
- Roughly 5 tool calls max per sub-topic overall (e.g., 2 searches + 1–2 page loads + 1 compute) — if you hit that with no result, move on rather than linger.

**Computing:**
- For numbers, comparisons, or quantitative verification, use `run_python` — don't approximate in prose.

**Sources:**
- Prefer primary sources and established outlets over aggregator summaries.
- If sources contradict, surface the disagreement — don't silently pick one.

---

## Step 3 — Reflect

After every 2–3 dives: *Is there a significant angle the sources haven't addressed?* If yes, go cover it. If no, proceed. This is a quick gut-check, not a reason to keep searching.

---

## Step 4 — Report

This is the LAST step.

1. IMPORTANT: immediately load the `charting` and `report-writing` skills.
2. Explicitly follow the report-writing and charting rules! Write a high quality
   report ~1000 words. Place at least 2 charts in the report, where data is
   clearer shown than told — comparisons, trends, distributions.
3. Follow with a short chat reply (2–4 sentences) summarizing the key finding.
4. Stop. This message is the final output of the turn — no further tool calls,
   no further thinking, no further text after the chat reply.

---

## Budget

- **Sources**: 6–12 total. A handful of strong sources beats exhaustive searching.
- **Stop signal**: two consecutive searches on the same sub-topic yield nothing new — stop gathering.
- **Hard stop**: if approaching the tool-call limit, skip remaining gather steps and write the report with what you have. A partial honest report beats running out of steps mid-search.

---

## Rules

- Surface contradictions between sources; do not silently resolve them.
- If current knowledge does not support you from answering the question, be honest and say I don't know, and explain what information you have gathered could be useful. 
- Depth and accuracy over length — no filler paragraphs.
- Never end the turn still searching. Always finish with the report + chat reply.
