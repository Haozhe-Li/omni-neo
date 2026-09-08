---
name: draft-email
description: Draft, reply to, or rewrite an email. Use whenever the deliverable is an email — "write an email to…", "reply to this", "帮我写封邮件", a follow-up, an apology, an intro, a cold outreach, a resignation letter sent as email.
---

# Draft Email

An email is not just prose in a box — it has a recipient, a subject line, and a
job to do. The frontend renders it as a copyable email card, so it has its own
delivery format, different from the `text` fence used for ordinary rewrites.

## Delivery format

The finished email — and nothing else — goes inside a `<textblock>` block whose
opening tag carries `type="email"` and the subject line:

```
One short line of commentary.

<textblock type="email" subject="Re: Q3 Budget Proposal">
Hi Sarah,

Thanks for sending the revised figures over.

...

Best,
Alex
</textblock>

Want this warmer, or keep it strictly professional?
```

Rules for the tag:

- `type="email"` is required, and `"email"` is the only value that exists. Never
  invent another one.
- `subject` is always present. Never put a double quote inside it — it breaks
  the tag; use single quotes or 「」 instead.
- For a reply, prefix the subject with `Re: ` and keep the original subject
  wording when you have it.
- One `<textblock>` per email. Emit several only when the user explicitly asked
  for parallel versions (e.g. two tones) — one block each, with a one-line label
  above.

Inside the block: plain finished text only — greeting, body, sign-off. No
markdown decoration, no citation markers, no notes about what you changed, no
placeholder scaffolding like `[Your Name]` unless you genuinely have no way to
know it. Never also paste the same email again as plain chat text outside the
block.

## Writing the email

Work out four things before writing: who the recipient is, what you want them to
do, how much standing the sender has to ask it, and how much context the reader
already has. Those four decide length and register far more than any style rule.

- Open with the point, not with throat-clearing. "I'm writing to follow up on…"
  wastes the one line most likely to be read.
- One ask per email, stated explicitly, with a deadline when there is one.
  Buried or implied asks are the most common reason a real email fails.
- Match the register of the thread. If the user pasted the message they are
  replying to, mirror its formality, greeting style, and sign-off rather than
  imposing your own.
- Keep it short. Most business email is three short paragraphs or fewer; if it
  runs longer, the extra length should be structure (a short list of options,
  numbered questions), not more prose.
- Preserve concrete details the user gave you — names, dates, amounts, document
  titles — verbatim. Never invent a fact to make a sentence flow better, and
  never fabricate a commitment on the sender's behalf.
- Sign off as the user when you know who they are; otherwise use a neutral
  sign-off and let them fill in the name.

Write in the language of the request, unless the user asks for another one or
the thread they are replying to is clearly in a different language.

## After the block

End with one brief follow-up question on its own line — tone, length, whether to
add a specific point ("Want me to add the revised timeline, or keep it short?").
One question, no reply required.
