---
name: guided-learning
description: Teach the user a topic through structured explanation, charts, interactive quizzes, and a final study-notes report. Use when the user asks to "learn about", "teach me", "explain in depth", "quiz me on", or wants to understand a subject from scratch or in more detail.
---

# Guided Learning

A thinking-first workflow: understand the learner → design the explanation → decide how to present — then execute. Each step is a deliberation, not a fixed action.

---

## Step 1 — Understand the learner

Before doing anything else, figure out who you're teaching and what they actually need.

Ask yourself:
- **What do they already know?** Can you infer their level from the way they phrased the request, the vocabulary they used, or prior messages? If not, load the `ask question` skill and ask — but only ask what you genuinely can't infer. One question block, 1–3 questions max.
- **What is their goal?** Quick mental model? Deep mastery? Preparing for something specific (exam, project, interview)?
- **What specifically are they confused about, or curious about?** A broad topic like "machine learning" needs to be scoped — are they asking about the math, the intuition, how to use it, or something else?

Only proceed once you have a clear picture of the learner. If you need to ask, emit the `<question>` block and stop — wait for their answer before continuing.

---

## Step 2 — Design the explanation

Now think through *how* to teach this, calibrated to what you learned in Step 1.

Ask yourself:
- **What are the 2–4 core concepts** the user needs to understand? What order makes sense — does concept B depend on concept A?
- **What is the right entry point?** For a beginner, lead with intuition and analogy. For someone with background, you can be more direct and precise.
- **What analogies or examples** will make this click? Think of 1–2 concrete real-world examples per concept. Avoid textbook definitions as the first thing you say.
- **Where are the common misconceptions?** Name and address them proactively.
- **What is the one key insight** the user should walk away with? Make sure your explanation builds toward it.

---

## Step 3 — Decide how to present

Now decide on the *form* of the lesson. This is where you choose your tools. Think through each of these:

**Plain text vs. charts:**
Does any concept become significantly clearer with a visual? If yes, load the `charting` skill and plan a chart. If the explanation works fine as prose, skip the chart — do not add one for decoration.
Ask: is there a comparison, trend, proportion, hierarchy, or flow that a chart would communicate faster than words?

**Quiz:**
Would testing the user help them retain this? A quiz makes sense when:
- The topic has facts, rules, or formulas worth reinforcing.
- The user wants to check their understanding.
- The material has been fully explained (never quiz before teaching).
If yes, plan to load the `ask question` skill after the explanation and emit a graded `<question>` block (3–6 questions, always set `correct_answer`).

**Review report:**
Should you produce a study-notes report at the end? A report makes sense when:
- The topic is complex enough that the user would benefit from a clean summary to revisit.
- The lesson covered multiple concepts or had a quiz with results to include.
If yes, plan to load the `report-writing` skill after the explanation (and after grading the quiz if there is one).

**Write todos:**
If the lesson involves 3+ distinct tasks (e.g., teach concept A → teach concept B → quiz → report), call `write_todos` now to plan the execution sequence. For a simple single-concept explanation, skip todos.

---

## Execute

Now carry out what you planned. A few rules that always apply:

- **Teach before quizzing.** Never emit a `<question>` quiz block mid-explanation. Finish the full lesson first.
- **Quiz before reporting.** If there is a quiz, grade it and give brief feedback before opening the report.
- **The `<question>` block is always the last thing in a turn.** Do not add text after it.
- **Charts must earn their place.** Only visualise something if a chart is genuinely clearer than prose.
- **The report stands alone.** Write it so someone reading only the report understands the topic — no reliance on the chat context above.

### If you write a report

Load `report-writing` and `charting`, then follow the report-writing skill rules. A good study-notes report contains:
- A brief topic summary (what it is and why it matters)
- One section per core concept, each with its chart if applicable
- A quick-reference section (5–10 key facts, rules, or formulas in bullet form)
- Quiz results if there was a quiz (score + table: question / answer / correct / ✓✗)
- 2–4 suggestions for what to explore next

After the report, write a short chat reply (2–3 sentences): acknowledge the quiz score if there was one, note one thing worth revisiting, and suggest the clearest next step. Then stop.

---

## Rules

- Never invent facts. If uncertain, say so or search (2 searches per concept, max).
- Calibrate relentlessly to the user's level — pitching too high or too low is the biggest teaching failure.
- Quiz questions must only cover concepts that were actually taught.
- Do not add charts, quizzes, or reports just because the skill mentions them — add them only if they serve this specific learner and topic.
