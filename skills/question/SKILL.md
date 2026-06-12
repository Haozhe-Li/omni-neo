---
name: question
description: use this skill when you need input from the user before you can proceed, e.g. quiz, clarifying questions, or gathering preferences.
---

# Question

When you need input from the user before you can proceed, end your turn with a
single `<question>…</question>` block containing a JSON object. The frontend
renders it as an interactive form; the user's answer comes back as their next
message.

**This must always be the last thing in your response.** Do not output a
`<question>` block mid-answer or mid-report.

---

## Schema

```json
{
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
```

### Fields

**`type`** — controls the selection mechanic:
- `"single"` — radio buttons, user picks exactly one option.
- `"multiple"` — checkboxes, user picks one or more options.
- `"text"` — pure fill-in-the-blank, no options. Omit `options` entirely.

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

### Single choice
```
<question>
{
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
</question>
```

### Multiple choice
```
<question>
{
  "type": "multiple",
  "prompt": "Which topics should the report cover? Pick all that apply.",
  "options": [
    { "id": "A", "label": "Market size & growth" },
    { "id": "B", "label": "Key players & competition" },
    { "id": "C", "label": "Regulatory landscape" },
    { "id": "D", "label": "Technology trends" }
  ],
  "text_placeholder": null,
  "correct_answer": null
}
</question>
```

### Fill-in-the-blank
```
<question>
{
  "type": "text",
  "prompt": "What city should I base the weather and cost-of-living data on?",
  "options": [],
  "text_placeholder": "e.g. Tokyo",
  "correct_answer": null
}
</question>
```

### Single choice + free text (combination)
```
<question>
{
  "type": "single",
  "prompt": "What's your budget range?",
  "options": [
    { "id": "A", "label": "Under $500" },
    { "id": "B", "label": "$500 – $1,500" },
    { "id": "C", "label": "Above $1,500" },
    { "id": "D", "label": "I have a specific number in mind", "has_text_input": true }
  ],
  "text_placeholder": "Enter your budget…",
  "correct_answer": null
}
</question>
```

### Quiz (with correct answer)
```
<question>
{
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
</question>
```

---

## When to use

- **Before deep research**: if the scope is ambiguous, ask one focused
  clarifying question rather than guessing. One question per turn — do not
  stack multiple `<question>` blocks.
- **Personalizing recommendations**: when the right answer depends on the
  user's situation (budget, location, use case).
- **Quiz / learning mode**: when the user asks you to test them on a topic.

## When NOT to use

- When you can make a reasonable assumption and note it inline.
- When the question is trivial enough to answer both ways ("I'll cover both X
  and Y since you didn't specify").
- Mid-report or mid-answer — always terminal.
