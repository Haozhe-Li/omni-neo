---
name: scheduled-research
description: Propose a recurring research task the user can turn into a scheduled job — a daily/weekly/monthly report emailed to them automatically. Use whenever the user wants something researched repeatedly going forward ("every morning", "each week", "keep me posted on X", "email me monthly"), not a one-off answer.
---

# Scheduled Research

Some requests aren't asking for an answer right now — they're asking for a
standing job: "give me a daily AI news digest", "check on this stock every
Friday", "send me a monthly recap of X". For those, propose a scheduled task
instead of just answering once.

## When to use

- The user asks for something on a recurring cadence: "every day", "each
  week", "monthly", "from now on", "keep me updated on…", "remind me
  periodically with…".
- The natural deliverable is a research report, not an action you can take
  right now (this is about *research* run on a timer, not generic
  reminders/alarms).

## When NOT to use

- A one-off research request, even a big one — that's a normal answer, or the
  report-writing / web-research skill. Do not propose a schedule the user
  didn't ask to repeat.
- The user is asking to view, edit, pause, or delete an *existing* schedule —
  that happens in Settings → Scheduled Research, not through this block. Tell
  them where to find it instead of emitting a new proposal.
- The user only wants to be asked once and reminded, with no research
  involved.

## How

Nothing is scheduled by you emitting this block — it only stages a proposal.
The frontend renders it as a card with **Confirm** / **Not now** buttons; the
task is created only if the user clicks Confirm, and reports are delivered to
the email on their account. Never say the task is already scheduled or
active — say you've put together a proposal, and point at the card.

Stream the proposal inline as a `<scheduled-research>` block — exactly the
way a report is a `<report>` block, not a tool call:

```
Here's a scheduled research proposal — confirm it below to turn it on.

<scheduled-research title="AI News Digest" frequency="daily" time="08:00">
Research and summarize the most significant AI news and developments from the
past 24 hours, covering major model releases, funding, research breakthroughs,
and industry moves.
</scheduled-research>
```

### Attributes (on the opening tag)

- **`title`** — required. Short (3–6 words), title case, no punctuation.
- **`frequency`** — required. One of `daily`, `weekly`, `monthly`. Never
  propose anything more often than daily.
- **`time`** — required. 24-hour `"HH:MM"` in the user's own local time (you
  already know it from `<system_reminder>`). If the user didn't say one, pick
  a sensible default for the content: a morning digest → `"08:00"`, an
  evening wrap-up → `"18:00"`, otherwise `"09:00"`.
- **`weekday`** — only when `frequency="weekly"`. `0`=Sunday … `6`=Saturday.
  Infer it from phrasing like "every Friday" (`5`); if unspecified, default to
  `1` (Monday). Omit entirely for `daily`/`monthly`.
- **`day_of_month`** — only when `frequency="monthly"`. `1`–`28` (capped to 28
  so it exists in every month). Default to `1` if unspecified. Omit entirely
  for `daily`/`weekly`.

### Body

The body is the research instruction itself — rewrite the user's request as a
clear, self-contained task for an unattended agent to execute on its own each
time it fires. No "you asked for…" framing, no conversational voice — just
the research task, the same way you'd brief someone who will run it without
you present. Expand a vague request into something concretely researchable.

Plain text only inside the body — no markdown headers, no nested tags.

## Rules

- Exactly one `<scheduled-research>` block per turn, never nested.
- **This must always be the last thing in your response** — same rule as the
  ask-question skill. Do not add anything after the closing tag; the user
  needs to act on the card (Confirm / Not now) before the conversation
  continues, and their click comes back as their next message.
- One short line of commentary before the block, introducing the proposal and
  making clear it needs confirmation. Never restate the schedule or the
  instruction again as plain chat text outside the block.
- Do not mention email addresses — delivery goes to whatever email is on the
  user's account, and the card shows that on its own.
- If the user's next message says they declined or didn't confirm it, don't
  re-propose the same schedule unprompted; treat it like any other answered
  turn and respond to whatever they say next.

## Examples

### Recurring digest, no explicit time
```
<scheduled-research title="AI News Digest" frequency="daily" time="08:00">
Research and summarize the most significant AI news and developments from the
past 24 hours, covering major model releases, funding, research breakthroughs,
and industry moves.
</scheduled-research>
```

### Weekly, explicit day
User: "give me a stock market recap every Friday evening"
```
<scheduled-research title="Weekly Market Recap" frequency="weekly" time="18:00" weekday="5">
Research and summarize this week's stock market performance, major index
movements, and notable company or macroeconomic events.
</scheduled-research>
```

### Monthly
User: "monthly digest of space exploration news"
```
<scheduled-research title="Space Exploration Digest" frequency="monthly" time="09:00" day_of_month="1">
Research and summarize the past month's major space exploration news,
including launches, missions, and industry developments.
</scheduled-research>
```
