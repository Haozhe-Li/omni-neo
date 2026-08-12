# Pro-agent fine-tune

Distilling `gpt-5.6-luna`'s behaviour on the pro agent into
`Qwen/Qwen3-30B-A3B-Instruct-2507`, trained on W&B Serverless Training and
served by W&B Inference.

## v1 result — 0.789 → 0.866 (teacher 0.962)

```
wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign/omni-pro-agent/omni-pro-104-0812-0016:v1
```

104 rows, 1 epoch, rank-1 LoRA, 18 minutes. Supabase label
`sft-v1-104rows-1epoch`.

**Shipped 2026-08-12.** `core/llm.py::chat_llm` points here and the fast/pro
split is gone — every interactive request is served by this adapter. Two things
that had to happen first:

- **One profile, one prompt.** A LoRA has exactly one compatibility key. The old
  fast profile had a shorter prompt and a two-skill roster, which is a *different*
  system prompt, not a lighter one. `core/agent.py` now assembles a single
  `SYSTEM_PROMPT`; the refactor was verified byte-identical to the old
  `PRO_PROMPT` (sha `6a10fe08…`, 12,762 chars) and `fingerprint.py` reports
  `harness unchanged`.
- **Image turns route away.** W&B serves this text-only, so
  `VisionModelMiddleware` sends any conversation containing an image to
  `gemma_4_31b`. Image answers therefore behave like the gemma baseline, not
  like the fine-tune.

Note this is a ~3-point trade on paper, not an upgrade: `gemma-4-31b` scored
0.897 on the same 28 cases (older rubric, 1 run). What it buys is skill loading
the base could not do at all, and a model we control end to end. Reverting is
one line — point `chat_llm` back at `gemma_4_31b`.

**The result that mattered: skill loading went from 4/51 runs to 26/51.** That
was the whole hypothesis — the base model's failures are protocol compliance,
not capability — and it held. It also *generalised*: `web-research/
solid-state-batteries` improved 0.639 → 0.870 even though all 15 deep-research
trajectories were excluded from training.

| suite | base | v1 | teacher |
|---|---|---|---|
| mapping | 0.802 | **0.993** | 0.993 |
| report-writing | 0.714 | 0.900 | 0.986 |
| trip-advisor | 0.646 | 0.821 | 0.884 |
| charting | 0.811 | 0.917 | 0.975 |
| about / adversarial / ask-question / general / guided-learning | +0.04 to +0.10 each | | |
| language | 0.916 | 0.901 | 1.000 |
| web-research | 0.638 | **0.523** | 0.966 |

`adversarial` rose (+0.072, `injection-in-page` 0.630 → 0.800), so the feared
degradation from fine-tuning on 100 benign trajectories did not appear here.
Keep watching it.

The web-research regression is **not a quality result**: both runs of
`sea-lions-vs-seals` died with `ReadError` / `RemoteProtocolError` and scored
0.000. All 5 errors hit the heaviest multi-turn cases; no light case failed.
That is a serving-stability problem with the hosted adapter and matters more
for production than the score does.

## The constraint that shaped everything: longest-sequence provisioning

The backend provisions against the dataset's **longest** sequence, so one
208k-token row degraded *every* step:

| dataset | per step | outcome |
|---|---|---|
| 129 rows (max 208k) | ~540s | 1 gradient step in 2 hours, then stuck |
| 104 rows (max 32k) | ~10s | 104 steps in 18 minutes |

A ~54x difference. This is not "19% of rows can't train" — it is "19% of rows
stop the other 81% from training". Truncating tool results does not help: those
rows are long because of the *number* of pages read, so even a 600-token cap
per result still leaves deep-research at 0/15.

Cost of the 32k filter: all 15 `deep-research` and 6 of 10 `chart`
trajectories. That is why v1 should not be expected to move web-research.

**Next version:** hard-truncate those 25 rows to 32k rather than dropping them,
and train 4 epochs (~1.5 hours at 10s/step). That recovers deep-research, which
is the one remaining gap.

## Other things worth knowing before touching this

- **LoRA rank is fixed at 1.** `SFTTrainingConfig` exposes only `batch_size`
  and `learning_rate`; `models.create` takes no rank. The identity LoRA path is
  literally `identity-loras/.../rank-1/`.
- **`v0` is never a fine-tune.** It is the identity LoRA downloaded at
  registration. Only `v1`+ with a `step > 0` checkpoint carrying non-empty
  metrics is a trained adapter. `check_run.py` checks exactly this.
- **Tool-call tokens *are* trained.** Verified against the real Qwen tokenizer:
  observed `num_trainable_tokens` of 80 and 312 match content+tool_calls (76,
  302), not content alone (45, 96). The `raise ValueError("...ignores tool
  calls")` in `art/preprocessing/tokenize.py` is the LocalBackend path only —
  serverless does not use it.
- **Jobs queue ~2 minutes** before `training_start`. Silence early on is normal;
  do not read failure into it.
- Read run state with `check_run.py`, not the client's stdout. During this work
  the client's progress bar and the W&B run page each implied things that the
  job-status and event APIs contradicted.

## Baselines — full 28-case rubric, `evals/cases.yaml`

| | score | pass_rate | errors | cost |
|---|---|---|---|---|
| gpt-5.6-luna (teacher) | **0.962** | 0.969 | 0 | $0.30 |
| qwen3-30b-a3b (base) | **0.789** | 0.820 | 5 | $1.09 |

Supabase labels `baseline-28-luna-v2-langfix` and
`baseline-28-qwen3-30b-v2-langfix`. Earlier numbers (luna 0.895 / 0.965, qwen
0.720 / 0.786) were measured on rubrics with known defects and are not
comparable; these are the reference points for judging the fine-tune.

Both are stable across the rubric fixes (luna 0.965 → 0.962, qwen 0.786 →
0.789), which is the expected result: the last fix removed a false positive
that only ever fired on the teacher.

The gap is 17.9 points and it is almost entirely protocol compliance, not
capability. Across 56 runs the base model loaded a skill **zero** times, which
takes every artifact contract down with it:

```
skill_loaded:*  0/26 across seven skills     has_report      0/9
question_block  0/9                          charts_valid    0/7
map_fence       0/5                          skill_load_order 0/2
```

Two diagnostics pin down why:

- **It can reach the tools.** Given `<requested_skill>web-research</requested_skill>`
  — which SYSTEM_PROMPT says to "load before anything else" — it called
  `write_todos` twice and `read_file` never. The deepagents tool surface works;
  reading a skill simply is not in its behavioural repertoire. That is what SFT
  fixes.
- **It is not slow.** 118 tok/s empty, 74 tok/s at 40k context, and a median
  3.32s per turn — faster per turn than luna's 5.27s. The timeouts are step
  count: 64 turns where luna takes 10. Teaching it to stop turns latency from a
  liability into an advantage.

## What can be cut from the prompt (and why the answer is "nothing yet")

Asked directly: which sections could move into the weights and out of the
2,889-token `SYSTEM_PROMPT`? For **this** adapter, none — it has only ever seen
the full string. The question only has a real answer for the *next* collection
round, so here is what the 129 traces support.

The criterion that decides it: **SFT teaches what to do, not what not to do.**
A positive, high-frequency rule is demonstrated dozens of times and compresses
into weights. A negative constraint appears in the data only as an *absence* —
no gradient describes where the boundary is — so it has to stay in the prompt.

| section | tokens | evidence in 129 traces | verdict |
|---|---|---|---|
| `_S_RETRIEVAL` | 429 | search 74, load_web_page 48, places 8, stock/fx/weather 1–4 | ✅ best candidate |
| `_S_COMPUTATION` | 154 | `run_python` 31 | ✅ |
| `_S_PLANNING` | 66 | `write_todos` 40 | ✅ |
| `_S_INPUT_FORMAT` | 279 | rarer tags only 6–12 rows each | ⚠️ compress, don't cut |
| `_S_TOOL_DISCIPLINE` | 123 | 0 mixed turns — learned perfectly | ⚠️ but catastrophic if it breaks |
| `_S_WRITING_FORMAT` | 402 | `<textblock>` in only 10/129 | ❌ |
| `_S_FORMATTING` / `_S_CITATIONS` | 591 | negative constraints | ❌ |
| `_S_LISTS` / `_S_HEADERS` / `_S_TONE` / `_S_GOAL` | 566 | exactly what the judge scores | ❌ |

Ceiling: **649 tokens, 22% of the prompt** — and only after re-collecting and
retraining. Not worth a round on its own.

## The harness is frozen

`fingerprint.py` hashes the **assembled** system prompt and the full tool
schema — not `SYSTEM_PROMPT`, which is only 2,889 of the 4,368 assembled tokens.

```bash
python finetune/pro_agent/fingerprint.py            # verify, exit 1 on drift
python finetune/pro_agent/fingerprint.py --update   # re-bless (see below)
python finetune/pro_agent/fingerprint.py --show     # dump what is hashed
```

Verified behaviour, by test rather than by assumption:

| change | drifts? |
|---|---|
| one word in a `SYSTEM_PROMPT` heading | yes |
| a skill's `description:` line | yes |
| a skill's SKILL.md **body** | **no** |

That last row is the useful one. Skill bodies arrive at runtime as `read_file`
results, so they are training data, not weights — they stay editable after
collection. Names and descriptions are inlined into deepagents'
`## Skills System` section, so they are weights.

`deepagents` is pinned to `==0.6.10` for the same reason; the version is part
of the fingerprint.

**Re-blessing is not a formality.** It means every trajectory collected under
the old fingerprint is stale and any adapter trained on them needs retraining.

## Data plan

130 queries → best-of-3 luna rollouts → hard-gate filter → ~100 trajectories.

The gate is the query-independent subset of `common_checks`, applied as
pass/fail rather than weighted:

```
tool_discipline  no_hyperlinks  no_ascii_art     citation_exists  citation_format
no_leading_header  latex_sanity  search_discipline  no_prompt_leak  response_language
```

Measured yield on the 56 baseline luna traces: **48/56 = 86%**, so best-of-3
gives 99.7% chance of at least one usable trace per query. The four checks luna
trips (`response_language` 4, `citation_format` 2, `no_ascii_art` 1,
`citation_exists` 1) are genuine violations — the gate is doing its job.

`citation_coverage` is deliberately **not** a gate. It is a heuristic that
frontier models fail ~40% of the time; gating on it would discard good data and
select for a quirk. It stays a low-weight scored item in the benchmark only.

### Composition

Two orthogonal axes. Category decides the must-pass checks:

| category | n | must pass |
|---|---|---|
| `budget-exhausted` | 20 | collected at low `run_limit`; must still produce the deliverable |
| `deep-research` | 15 | `skill_loaded` web-research + report-writing, `has_report`, `chart_count`, `citation_count` |
| `search-fact` | 15 | `tool_called:google_search`, `citation_count`, `citation_exists` |
| `single-tool` | 12 | weather / stock / fx, 4 each |
| `chart` | 10 | `skill_loaded:charting`, `chart_count`, `charts_valid` |
| `teach` | 10 | `skill_loaded:guided-learning`, `question_block` |
| `write-rewrite` | 10 | `textblock`, `followup_question`, `no_tool_calls` |
| restraint negatives | 10 | `no_tool_calls` / `no_report` / `no_map` / `no_question_block` |
| `places` | 8 | `google_search_places`, `skill_loaded:mapping`, `map_fence` |
| `compute` | 8 | `tool_called:run_python`, `no_report` |
| `ask-question` | 8 | `skill_loaded:ask-question`, `question_block` |
| `about` | 4 | `skill_loaded:about-omni` / `about-haozheli` |

Tag decides which optional user-message blocks ride along, because production
emits seven and the eval harness only ever produced three:

| block | n | note |
|---|---|---|
| `<user_memory>` | 12 | common in production, never seen in training so far |
| `<attached_files>` | 8 | synthetic doc mounted at `/uploads/` |
| `<priority_sources>` | 6 | was undocumented in `_S_INPUT_FORMAT` until 2026-08-11 |
| `<follow_up_selection>` | 6 | |
| `<requested_skill>` | 6 | |

Language mix 60% zh / 40% en. The datetime, location and language *values*
inside `<personalization>` must vary across rows while the field format stays
fixed — otherwise `2026-08-02` gets baked into the weights.

### Two rules that are easy to get wrong

- **Zero overlap with `evals/cases.yaml`.** Enforce it in the generator, not by
  eye. The 28 benchmark queries are the test set.
- **`budget-exhausted` traces need trimming.** luna retried 8 times before
  giving up at `run_limit=6`. Fed raw, that teaches "retry 8 times, then
  finish". Keep one blocked call, drop the rest — the message sequence stays
  valid and the lesson becomes "budget hit, write the answer".

## Known risks

- **Injection resistance may degrade.** The base already scores 0.630 on
  `adversarial/injection-in-page` (`judge:treats_as_data` 0/2). Fine-tuning on
  100 benign trajectories is a known way to make this worse, and the benchmark
  has exactly two cases that would catch it. Watch them specifically, not just
  the total.
- **Serving cost is not free even though training is.** The 402 that killed the
  first benchmark run was a W&B Inference quota exhaustion, and evaluation plus
  production traffic both bill against it.
- **`_S_INPUT_FORMAT` should stay in the prompt for round one.** Compressing it
  into the weights (~234 tokens) is only safe once the training data covers all
  seven blocks; until then it is the model's only source of tag semantics.
