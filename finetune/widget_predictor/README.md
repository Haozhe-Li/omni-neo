# Widget predictor fine-tune

A LoRA over `OpenPipe/Qwen3-14B-Instruct` that replaces `gpt-oss-20b` on the
widget-routing path in `core/widget_predictor.py`. Trained on W&B Serverless
Training (ART) and hosted by W&B Inference, so there is no adapter to upload and
no GPU to provision.

**93/100 vs 79/100** on the held-out split — 15 rows fixed against 1 regression,
McNemar exact two-sided *p* = 0.0005. Latency is at parity on p50 (171ms vs
156ms) and better on p95 (351ms vs 575ms). Training took 812s and was free under
the public preview.

## The prompt is part of the weights

The adapter has only ever seen queries under `_PREDICTOR_PROMPT` from
`core/widget_predictor.py`, rendered through `build_predictor_messages`. Every
script here renders the prompt with that same function, so training, evaluation
and production cannot drift apart.

**Editing that prompt invalidates the adapter.** Retrain; do not assume it
generalises to a rewritten system message.

Two consequences worth remembering:

- A query used as a few-shot example inside the prompt ships with its own answer
  attached, so it can never be a fair test row. `gen_queries.py` strips those
  automatically.
- `build_dataset.py` pins the 89 hand-written seed queries to *train*. The
  system prompt was rewritten while reading them one by one, so scoring on them
  would measure that fit rather than the model.

## Pipeline

Each step reads the previous step's output from `dataset/`; run them from the
repo root.

| Step | Command | Writes |
|---|---|---|
| 1. Generate candidates | `python finetune/widget_predictor/gen_queries.py -n 500` | `queries_new.tsv` |
| 2. Label with a teacher | `python finetune/widget_predictor/label.py --samples 3 --compare` | `labelled.jsonl`, `ground_truth.tsv` |
| 3. Review by hand | edit column 3 of `ground_truth.tsv` | — |
| 4. Split + build | `python finetune/widget_predictor/build_dataset.py --test-size 100` | `train.tsv`, `test.tsv`, `sft_train.jsonl` |
| 5. Train | `python finetune/widget_predictor/train.py --epochs 4` | a hosted adapter |
| 6. Score | `python finetune/widget_predictor/evaluate.py --lora <uri>` | — |

Step 2 **overwrites** `ground_truth.tsv`, discarding hand corrections from the
previous round. Label into a scratch file (`-r`) when adding to an already
reviewed set.

Step 3 is not optional. On the first 100 rows the teacher and the student agreed
on 89 and were *both wrong* on 5 of the rest — a disagreement flag cannot catch
an error the two models share.

## Comparing candidates

`evaluate.py` drives any model through the identical prompt and parser, so runs
differ only in weights:

```bash
python finetune/widget_predictor/evaluate.py                      # whatever is wired in
python finetune/widget_predictor/evaluate.py --model gpt_oss_20b  # the pre-LoRA baseline
python finetune/widget_predictor/evaluate.py --lora wandb-artifact:///...
```

On 100 rows the headline accuracy carries a confidence interval of about ±4
points, so two close systems are not distinguishable by that number alone. Use
`--dump` on both runs and compare per row: a paired test only counts the rows
where the two disagree, which is what made 93-vs-79 a real result rather than a
plausible one. The false-positive rate on no-widget rows and the printed
mismatch list matter as much as the total.

## `dataset/` is not in git

The whole directory is gitignored, so a fresh clone has the pipeline but no
data. What lives there:

| File | What it is |
|---|---|
| `ground_truth.tsv` | 586 rows, 3 teacher votes each, reviewed by hand. Vote counts survive in the trailing `#` note. |
| `train.tsv` / `test.tsv` | The split the numbers above were measured on. |
| `seed_queries.tsv` | The hand-written 100. `build_dataset.py` reads it to know which rows to pin to train. |
| `sft_train.jsonl` | Built. 2.7MB, mostly the 5KB system prompt repeated 486 times. |
| `labelled.jsonl` | Raw teacher output including every vote. |

`build_dataset.py` rebuilds the last four deterministically (fixed seed) from
`ground_truth.tsv` and `seed_queries.tsv`. Those two are the ones that matter,
and neither is reproducible: `ground_truth.tsv` cost ~1800 teacher calls plus a
manual pass, and `gen_queries.py` invents different queries on every run, so
re-running the pipeline produces a *different* test set that cannot be compared
against the results above.

Keep a copy outside the repo — a W&B artifact next to the adapter is the
natural home. Without one, losing this working copy means the eval can no
longer be re-run and the next fine-tune starts from nothing.

## Known issues

- **Crypto labels are inconsistent.** Four Chinese crypto price queries are
  labelled with an entity card while `bitcoin price today` is labelled empty.
  The adapter's single regression is it being more self-consistent than the
  labels. Fixing these and retraining is free.
- **Cold start is ~2.9s.** W&B scales the adapter to zero, so the first request
  after an idle period blows the widget's latency budget and its card will not
  reach the client before the answer. Steady state is fine.
- **Residual label noise is ~4%**, the teacher's self-disagreement rate. At the
  current accuracy, correcting labels buys more than adding rows.
- `assistant_turns` appears in the ART docs but not in the installed 0.5.18
  signature. Harmless for single-turn data — there is exactly one assistant
  message — but it matters for multi-turn work.
