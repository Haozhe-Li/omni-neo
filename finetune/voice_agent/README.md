# Voice-agent fine-tune

Distilling `gpt-6-luna` on the live-call voice agent into
`google/gemma-4-26B-A4B-it`, trained on W&B Serverless Training.

```
queries.yaml -> collect.py -> dataset/traces.jsonl -> filter.py -> dataset/sft_train.jsonl -> train.py
```

- `queries.yaml` — spoken-style zh/en queries. An `answer:` field is a
  hand-written reply that replaces the teacher's (identity questions: who built
  Omni Voice, which model powers it, "are you ChatGPT").
- `collect.py` — runs the production call agent (same prompt, tools, turn
  framing) with luna as the model. `--ids a,b` re-samples just those, up to
  `--attempts` times, until identity / tool error / empty / no lead-in clear;
  `--ids hand` re-writes every hand-answered query.
- `filter.py` — annotates, never deletes. Blocking: `too_long` (> 75 words for
  English, > 75 characters for Chinese), `no_leadin`, `identity` (a self-claim
  of being another model or vendor), `tool_error`, `empty`, `no_answer`.
  `markup` is annotated but not blocking — left to `core/voice/tts_text.py`.
- `train.py` — 3 epochs, batch 1, lr 1e-4, cosine.

**The system prompt is part of the training data.** Every row carries
`VOICE_SYSTEM_PROMPT + VOICE_CALL_PROMPT_ADDENDUM` and the tool schemas.
Change either and the whole set has to be re-collected and re-trained.

## Runs

| run | rows | prompt | identity (12 held-out phrasings) | tool turns with a lead-in |
|---|---|---|---|---|
| v1 | 62 | old | not measured | not measured |
| v2 | 90 | old | "Who developed you?" -> "Google" | not measured |
| v3 | 143 | identity + lead-in rules | 12/12 | 4/8 |
| v4 | 143 | same as v3 | 12/12 | 4/9 |

v4 is what `core/llm.py::rix_voice` serves. It differs from v3 only in how the
lead-in is stored (below).

## Gemma 4 gotchas, all measured

1. **The chat template moves text out of a tool-call message.** An assistant
   message with both `content` and `tool_calls` renders as
   `<|tool_call>...<tool_response|>Let me check.It's sunny.` — the text lands
   *after* the tool response. Two consecutive assistant messages (text, then
   call) render as `Let me check.<|tool_call>...<tool_response|>It's sunny.`.
   `filter.py::split_lead_ins` does this split. v3 was trained the first way.
2. **The generation prompt already ends with an empty thought block**
   (`<|turn>model\n<|channel>thought\n<channel|>`), and the training rendering
   of a model turn never contains one. The served model then emits a *second*
   `<|channel>thought\n<channel|>` (sometimes as bare `thought\n<channel|>` or
   `:thought\n`) and goes straight to the tool call. That mismatch is the best
   explanation for English lead-ins still arriving after the result, unchanged
   between v3 and v4. `core/voice/agent.py::_ThinkBlockStripper` removes the
   markers so they are never spoken.
   Tried and rejected: `reasoning_content=" "` on assistant messages renders
   `thought\n \n` — not the same tokens as the generation prompt's block.
3. **The server's loss mask covers tool responses**, not just the assistant's
   own words (5,127-token row, 2,932 loss-bearing tokens). Read off the
   `data/gradient_step_loss_bearing_tokens` lines in the training log.
4. `W&B` refuses `:latest` at inference; reference the concrete `:v1`.

## Open

Lead-in before an English tool call is not learned. Options not yet tried:
inject a canned lead-in at runtime when a tool call arrives with no words
before it (and drop the model's late "let me check" after the result), or find
a way to make the training rendering carry the same empty thought block the
generation prompt does.
