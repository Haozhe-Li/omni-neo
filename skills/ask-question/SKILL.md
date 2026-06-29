---
name: ask question
description: use this skill when you need input from the user before you can proceed, e.g. quiz, clarifying questions, or gathering preferences.
---

# Ask Question

When you need input from the user before you can proceed, end your turn with a
single `<question>…</question>` block containing a JSON object with a `questions`
array. The frontend renders it as an interactive form; the user's answers come
back as their next message.

**This must always be the last thing in your response.** Do not output a
`<question>` block mid-answer or mid-report.

---

## Schema

```json
{
  "questions": [
    {
      "id": "q1",
      "type": "single" | "multiple" | "text",
      "prompt": "The question to ask the user.",
      "options": [
        { "id": "A", "label": "Option text" },
        { "id": "B", "label": "Option text" },
        { "id": "C", "label": "Other", "has_text_input": true }
      ],
      "text_placeholder": "Hint text shown inside the text input.",
      "correct_answer": null
    }
  ]
}
```

### Top-level fields

**`questions`** — array of one or more question objects. Group all questions for
a single clarification round into one block; do not emit multiple `<question>`
blocks in one turn.

### Per-question fields

**`id`** — short unique identifier within this block (`"q1"`, `"q2"`, …).

**`type`** — controls the input mechanic:
- `"single"` — radio buttons, user picks exactly one option.
- `"multiple"` — checkboxes, user picks one or more options.
- `"text"` — fill-in-the-blank, no options. Omit `options` entirely.

**`prompt`** — the question text shown to the user. Be concise and specific.

**`options`** — array of choices for `single` and `multiple` types.
- `id`: short identifier used in `correct_answer` (`"A"`, `"B"`, …).
- `label`: the display text.
- `has_text_input` *(optional, bool)*: when `true`, selecting this option
  reveals a text field. Use for "Other — please specify" style choices.
  Only one option per question should have this.

**`text_placeholder`** *(optional)* — hint shown inside the text input.
Applies to `type: "text"` and to any option with `has_text_input: true`.

**`correct_answer`** — reserved for quiz mode. Set to `null` when asking a
genuine survey/clarification question.
- `"single"` type: a single option id string — `"B"`.
- `"multiple"` type: an array of option id strings — `["A", "C"]`.
- `"text"` type: the expected answer string — `"Paris"`.
- Not a quiz: `null`.

---

## Examples

### Single clarifying question
```
<question>
{
  "questions": [
    {
      "id": "q1",
      "type": "single",
      "prompt": "Which aspect matters most to you?",
      "options": [
        { "id": "A", "label": "Price" },
        { "id": "B", "label": "Performance" },
        { "id": "C", "label": "Availability in my region" }
      ],
      "text_placeholder": null,
      "correct_answer": null
    }
  ]
}
</question>
```

### Multiple questions at once (e.g. trip planning intake)
```
<question>
{
  "questions": [
    {
      "id": "q1",
      "type": "text",
      "prompt": "Where are you departing from, and what are your travel dates?",
      "text_placeholder": "e.g. New York, Jun 10 – Jun 17",
      "correct_answer": null
    },
    {
      "id": "q2",
      "type": "single",
      "prompt": "What is your total budget for the trip?",
      "options": [
        { "id": "A", "label": "Under $1,000" },
        { "id": "B", "label": "$1,000 – $3,000" },
        { "id": "C", "label": "$3,000 – $6,000" },
        { "id": "D", "label": "Above $6,000" },
        { "id": "E", "label": "I have a specific number", "has_text_input": true }
      ],
      "text_placeholder": "Enter your budget…",
      "correct_answer": null
    },
    {
      "id": "q3",
      "type": "multiple",
      "prompt": "What kind of experiences are you most interested in?",
      "options": [
        { "id": "A", "label": "Food & dining" },
        { "id": "B", "label": "Nature & outdoors" },
        { "id": "C", "label": "Art, museums & culture" },
        { "id": "D", "label": "Nightlife & entertainment" },
        { "id": "E", "label": "Shopping" }
      ],
      "text_placeholder": null,
      "correct_answer": null
    }
  ]
}
</question>
```

### Fill-in-the-blank
```
<question>
{
  "questions": [
    {
      "id": "q1",
      "type": "text",
      "prompt": "What city should I base the weather and cost-of-living data on?",
      "text_placeholder": "e.g. Tokyo",
      "correct_answer": null
    }
  ]
}
</question>
```

### Quiz (with correct answer)
```
<question>
{
  "questions": [
    {
      "id": "q1",
      "type": "single",
      "prompt": "Which protocol does HTTP/2 use for multiplexing?",
      "options": [
        { "id": "A", "label": "WebSocket" },
        { "id": "B", "label": "Streams over a single TCP connection" },
        { "id": "C", "label": "Multiple parallel TCP connections" }
      ],
      "text_placeholder": null,
      "correct_answer": "B"
    }
  ]
}
</question>
```

---

## When to use

- **Before deep research or trip planning**: gather all necessary inputs in one
  round — ask all your clarifying questions together rather than one per turn.
- **Personalizing recommendations**: when the right answer depends on the
  user's situation (budget, location, use case).
- **Quiz / learning mode**: when the user asks you to test them on a topic.

## When NOT to use

- When you can make a reasonable assumption and note it inline.
- When the question is trivial enough to answer both ways ("I'll cover both X
  and Y since you didn't specify").
- Mid-report or mid-answer — always terminal.
