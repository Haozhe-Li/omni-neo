"""Drop rows the training backend cannot fit, measured the way it measures.

    python finetune/pro_agent/filter_context.py --cap 32000

## Why this exists

W&B Serverless Training has a hard **32,768-token** sequence limit, and it does
not reject an over-long row — it *hangs*. The job keeps reporting `RUNNING`
forever, no terminal event ever arrives, and no checkpoint is written. Two runs
died this way before the limit was located.

The evidence that pins it down, from the v2 run that stalled at step 97 of 516:

    longest row it trained successfully   32,566
    shortest row it never reached         32,842

The boundary sits between those, and 2^15 = 32,768 is inside that interval.
Every one of the 97 completed steps was under it; both rows above it were still
untrained when the job stopped. Under a random shuffle the odds of both
over-limit rows landing in the untrained tail by chance are ~6%, so this is not
a coincidence — the run stopped on the first row it could not fit.

## Why filter rather than truncate further

`build_dataset.py` already water-fills tool results down toward a cap, and that
truncation is what let the 15 deep-research trajectories exist at all (they
start at up to 208k tokens). Filtering from the *raw* traces would throw all of
them away again, which is exactly the hole v1 fell into. So this runs on the
already-truncated file and removes only what still doesn't fit.

## Why the measurement here is different

The length must be measured as the trainer sees it: the Qwen chat template,
tokenized with Qwen's own tokenizer. `build_dataset.py` originally used
`tiktoken.o200k_base` over `json.dumps(...)`, which is both the wrong vocabulary
and full of JSON syntax that never reaches the model. That measure read the
longest v2 row as 32,768 when it was really 34,489 — the cap silently did not
hold, and it also made the earlier diagnosis look wrong.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = HERE / "dataset"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# The hard backend limit. The default cap sits below it: the largest row ever
# observed to train is 32,566, so 32,000 is inside proven-safe territory with
# room for any per-row overhead the trainer adds that this rendering does not
# reproduce. The cost of being wrong is another multi-hour silent hang; the cost
# of the margin is a handful of rows.
HARD_LIMIT = 32_768


def load_renderer():
    """`len(tokens)` of the fully rendered conversation, tools included."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(BASE_MODEL)

    def measure(row: dict) -> int:
        text = tk.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        return len(tk.encode(text, add_special_tokens=False))

    return measure


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "sft_train_v2.jsonl"))
    ap.add_argument("--out", default=str(DATA / "sft_train_v3.jsonl"))
    ap.add_argument("--cap", type=int, default=32_000)
    args = ap.parse_args()

    from build_dataset import dump_row, read_jsonl

    src, out = Path(args.src), Path(args.out)
    rows = read_jsonl(src)
    if args.cap > HARD_LIMIT:
        raise SystemExit(f"--cap {args.cap} is above the {HARD_LIMIT} backend limit")

    measure = load_renderer()
    lengths = [measure(r) for r in rows]

    # Recover each row's category from the source traces so the report says what
    # the filter actually costs — dropping three `single-tool` rows and dropping
    # three `deep-research` rows are not the same event.
    # Keyed on the `<user_query>` block, not on the head of the message. Every
    # user turn opens with a near-identical `<personalization>` block, so a
    # prefix key collides across rows and silently reports impossible category
    # counts (`teach` growing from 10 to 12 under a filter that only removes).
    def query_key(messages: list[dict]) -> str:
        for m in messages:
            if m["role"] != "user":
                continue
            body = m.get("content") or ""
            i, j = body.find("<user_query>"), body.find("</user_query>")
            return body[i:j] if i != -1 and j != -1 else body[-200:]
        return ""

    cat_by_query: dict[str, str] = {}
    traces_path = DATA / "traces.jsonl"
    if traces_path.exists():
        for t in read_jsonl(traces_path):
            cat_by_query[query_key(t["messages"])] = t["cat"]

    def cat_of(row: dict) -> str:
        return cat_by_query.get(query_key(row["messages"]), "?")

    keep = [(r, n) for r, n in zip(rows, lengths) if n <= args.cap]
    drop = [(r, n) for r, n in zip(rows, lengths) if n > args.cap]

    s = sorted(lengths)
    print(f"read {len(rows)} rows from {src.name}")
    print(f"  length: min {s[0]:,}  median {int(statistics.median(s)):,}  "
          f"p90 {s[int(0.9 * len(s))]:,}  max {s[-1]:,}")
    print(f"  backend hard limit {HARD_LIMIT:,}  |  cap {args.cap:,}")
    print(f"  over hard limit: {sum(1 for n in lengths if n > HARD_LIMIT)}")

    print(f"\ndropped {len(drop)} rows:")
    for r, n in sorted(drop, key=lambda x: -x[1]):
        print(f"  {n:>7,}  {cat_of(r)}")
    if drop:
        print("  by category:", dict(Counter(cat_of(r) for r, _ in drop)))

    print(f"\nkept {len(keep)} rows")
    print("  by category:", dict(Counter(cat_of(r) for r, _ in keep)))
    kl = sorted(n for _, n in keep)
    print(f"  length: median {int(statistics.median(kl)):,}  max {kl[-1]:,}")

    with out.open("w", encoding="utf-8") as f:
        for r, _ in keep:
            f.write(dump_row({"messages": r["messages"], "tools": r["tools"]}) + "\n")

    # Re-read exactly as a line-based consumer would, and re-measure — a file
    # that only satisfies the cap in memory is not the file being trained on.
    reread = read_jsonl(out)
    assert len(reread) == len(keep), f"round trip lost rows: {len(keep)} -> {len(reread)}"
    worst = max(measure(r) for r in reread)
    assert worst <= args.cap, f"a row still exceeds the cap after write: {worst:,}"
    print(f"\nwrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  verified on re-read: {len(reread)} rows, longest {worst:,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
