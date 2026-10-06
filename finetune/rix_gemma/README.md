# rix_gemma — distilling luna into Gemma from real conversations

The next `rix` is a LoRA on Gemma, trained on what real users thumbed up. This
replaces the fixed ~130-query collection in `finetune/pro_agent/` as the source
of teacher data: the queries are whatever people actually ask.

```
user thumbs-up on a luna answer
  -> POST /api/threads/{id}/feedback            core/routers/feedback.py
  -> read the thread's LangGraph checkpoint     (every tool call / result / image the agent saw)
  -> core/sft_capture.py                        LangChain messages -> OpenAI-style messages
  -> sft_examples  (+ harness_snapshots, sft_images)         Postgres, schema.sql
  -> curate.py     review / reject                           optional but recommended
  -> build_dataset.py  filter, dedupe, truncate, split lead-ins
  -> dataset/sft_train.jsonl  {"messages": [...], "tools": [...]}
  -> finetune/voice_agent/train.py --file ...   (base google/gemma-4-26B-A4B-it)
```

## One-time setup

```bash
python -m scripts.init_db     # creates sft_examples, harness_snapshots, sft_images (idempotent)
```

`DATABASE_URL` must be reachable from where you run the scripts. Railway's
`postgres.railway.internal` is private-network only; from a laptop use the
public proxy URL.

## What gets recorded

A thumbs-up on a turn whose assistant messages **all** report a luna model
(`response_metadata.model_name` contains `luna`; `best` and `luna` both qualify,
override with `SFT_TEACHER_MODEL_MARKERS`). The row holds the whole conversation
from the start of the thread through that answer, because the model reads all of
it and the loss covers every assistant turn. The system prompt and tool schemas
are stored once per `harness_hash` in `harness_snapshots`, not per row.

Not recorded (the endpoint answers `status: "skipped"`, the UI never knows):
other models, voice / scheduled-research threads, safety-locked threads,
unfinished turns, a turn interrupted mid-tool, anything over 2M chars. Fails
closed: a turn with no model attribution is not assumed to be luna.

Taking the thumb back, or thumbing down, deletes the row. A thumbs-down stores
nothing. Regenerating or editing an earlier turn (`/rewind`) deletes the rows
for that turn and everything after it. `DELETE /user-data` erases a user's rows;
a guest's rows follow them on sign-in merge. Deleting a **thread** does not
delete its examples — decide whether that is the behaviour you want.

## Build and train

```bash
python finetune/rix_gemma/curate.py stats
python finetune/rix_gemma/curate.py list
python finetune/rix_gemma/curate.py set rejected 45 --note "user misclicked"

python finetune/rix_gemma/build_dataset.py --report          # what would go in, and why rows were dropped
python finetune/rix_gemma/build_dataset.py --only-accepted --holdout 20

WANDB_PROJECT=omni-rix-gemma python finetune/voice_agent/train.py \
    --file finetune/rix_gemma/dataset/sft_train.jsonl --name rix-gemma-v1
```

`train.py` is the voice agent's, reused as-is: it is generic apart from the
default W&B project (overridden above) and already targets Gemma 4 26B-A4B,
the Gemma that W&B Serverless Training can train. That is the MoE, not the 31B
dense model `gemma_4_31b` serves on Cerebras.

## Things to know before trusting a result

- **A new prompt invalidates the data.** Every row is tied to a `harness_hash`;
  edit the system prompt, a skill's `description:`, a tool docstring, or bump
  deepagents and old rows stop being buildable by default (`--harness current`).
  Skill *bodies* may change freely — they arrive as `read_file` results, which
  are in the row. This is the same constraint `pro_agent/fingerprint.py` guards.
- **Thumbs-up selects for pleasing, not correct.** It will favour long,
  confident, well-formatted answers, including ones that are wrong. `curate.py`
  and the eval suite (`evals/`) are the counterweight; do not skip the eval
  because the data is "real".
- **Selection bias in what gets thumbed.** Quick, satisfying queries get more
  clicks than hard research ones, so the tool-use mix will skew simple.
  `build_dataset.py` prints the tool-use and skill-load counts; check that
  `read_file` on `/skills/` is still well represented, since skill loading is the
  behaviour the base model lacks most.
- **A later turn drags earlier ones in.** Thumbing only turn 3 trains on turn 1's
  answer too, approved or not. Rows with a non-luna earlier turn are dropped.
- **Privacy defaults.** Rows with `<user_memory>` or uploaded files are excluded
  by default (a model can repeat what it was trained on); location and local
  time in `<system_reminder>` remain, as in production. Tool results can still
  contain whatever a fetched page said. Whether users have agreed to their
  conversations being used for training is a product/legal question this code
  does not answer.
- **Images** are stored (deduplicated, in `sft_images`) but excluded from builds
  until the trainer is confirmed to accept image rows; try `--images keep` on a
  small file first. Dropping `VisionModelMiddleware` (the reason Gemma lets
  `best` stop swapping models on images) depends on that, so it is not part of
  this change.
- **Not exercised against a real database.** The SQL in
  `core/database/db_sft_examples.py` and `schema.sql` was written carefully but
  has not been run; `tests/test_sft_capture.py` and `tests/test_rix_gemma_build.py`
  cover everything else offline.
