"""Turn collected traces into the JSONL `train_sft_from_file` consumes.

    python finetune/pro_agent/build_dataset.py

Reads `dataset/traces.jsonl`, writes `dataset/sft_train.jsonl` plus a
length audit. Both live under the gitignored `dataset/`.

## Row shape

One row per trajectory:

    {"messages": [system, user, assistant(tool_calls), tool, ..., assistant],
     "tools": [...15 OpenAI tool schemas...]}

The system prompt is the **assembled** one captured at collection time, not
`SYSTEM_PROMPT` — deepagents appends ~1,525 tokens of its own sections, and
training on the shorter string would serve the model something it never saw.
Every collected trace carries its own copy and all 129 agree, which is checked
here rather than assumed.

`tools` comes from `fingerprint.capture()`, the same code path the compatibility
hash uses, so the schemas in the training file and the schemas in production
cannot drift apart silently.

## Loss masking

ART trains on **every** assistant turn by default, which is what agentic SFT
needs — the tool-calling turns are most of what the student is missing. Do not
pass `assistant_turns="last"`. (That parameter is documented but absent from the
installed openpipe-art 0.5.18; the default is already correct.)

## The U+2028 hazard

`json.dumps(ensure_ascii=False)` leaves U+2028, U+2029 and U+0085 unescaped —
they are legal inside a JSON string — but Python's `str.splitlines()` breaks on
all three. Exactly one U+2028 arrived in a fetched web page and split one record
of `traces.jsonl` in two. Any line-based reader hits the same trap, so they are
escaped explicitly on the way out. Cheaper than `ensure_ascii=True`, which would
escape every Chinese character and roughly triple the file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

DATA = HERE / "dataset"

# Characters that are valid in a JSON string but that `str.splitlines()` treats
# as line terminators. Escaping them keeps one row on one line for every reader.
_LINE_BREAKERS = {" ": "\\u2028", " ": "\\u2029", "": "\\u0085"}


def read_jsonl(path: Path) -> list[dict]:
    """Split on `\\n` only — never `splitlines()`. See the module docstring."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


TRUNCATION_MARKER = "\n\n…[内容已截断]"


def load_tokenizer():
    """Return `len(encode(s))` using the **base model's own** tokenizer.

    This used to be `tiktoken.o200k_base`, which is a different vocabulary than
    the one the trainer actually uses. On this corpus Qwen is 2-6% heavier, so a
    row truncated to a 32,768 "token" ceiling measured in o200k came out at up
    to 36,307 real tokens — the cap silently did not hold. Measure with the
    tokenizer that will do the tokenizing.

    Falls back to o200k, then to a character estimate, so the audit still runs
    on a machine without `transformers` — but a fallback means the ceiling is
    approximate and a training file built under one should not be trusted to
    respect a hard limit.
    """
    try:
        from transformers import AutoTokenizer

        enc = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Instruct-2507")
        return lambda s: len(enc.encode(s))
    except Exception as e:
        print(f"WARNING: Qwen tokenizer unavailable ({type(e).__name__}); "
              f"lengths below are estimates and any cap is approximate")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(s) // 3


def shrink_row(row: dict, cap: int, tok) -> tuple[dict, int]:
    """Truncate tool results until the row fits `cap` tokens. Returns (row, cut).

    Water-filling, not a flat cap: every tool result gets the same ceiling and
    the ceiling is raised as high as the budget allows, so short results survive
    intact and only the long ones lose anything. A flat 600-token cap was the
    first thing tried and it mangled 20 small results to save one big one.

    Why this is worth doing at all: the backend provisions against the dataset's
    *longest* sequence, so a single 208k-token row made every step 54x slower.
    Dropping those rows costs all 15 deep-research trajectories — the exact
    behaviour the student is weakest at — so they are shrunk instead.

    The cut is marked in the text. The teacher's answer was written against the
    full evidence, so a silent truncation would train the model to assert things
    its context no longer contains; a visible marker at least makes the gap
    something the model can see.
    """
    tool_idx = [i for i, m in enumerate(row["messages"]) if m["role"] == "tool"]
    if not tool_idx:
        return row, 0

    # Measured the way the row is actually serialised, not as a sum of field
    # lengths: JSON keys, quoting and role names are real tokens, and ignoring
    # them left 20 rows above a 32,768 target on the first attempt.
    def row_tokens(r: dict) -> int:
        return tok(json.dumps(r["messages"], ensure_ascii=False)) + tok(json.dumps(r["tools"]))

    if row_tokens(row) <= cap:
        return row, 0

    lens = [tok(row["messages"][i].get("content") or "") for i in tool_idx]

    def apply(ceiling: int) -> tuple[dict, int]:
        msgs = list(row["messages"])
        cut = 0
        for i, n in zip(tool_idx, lens):
            if n <= ceiling:
                continue
            body = msgs[i].get("content") or ""
            keep = body[: max(1, int(len(body) * ceiling / max(n, 1)))]
            while tok(keep) > ceiling and len(keep) > 200:
                keep = keep[: int(len(keep) * 0.9)]
            msgs[i] = {**msgs[i], "content": keep + TRUNCATION_MARKER}
            cut += 1
        return {**row, "messages": msgs}, cut

    # Highest uniform ceiling whose *serialised* row still fits.
    lo, hi, best = 0, max(lens), None
    while lo <= hi:
        mid = (lo + hi) // 2
        cand, cut = apply(mid)
        if row_tokens(cand) <= cap:
            best = (cand, cut)
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best else apply(0)


def dump_row(row: dict) -> str:
    out = json.dumps(row, ensure_ascii=False)
    for ch, esc in _LINE_BREAKERS.items():
        out = out.replace(ch, esc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0,
                    help="rows reserved from training (default 0 — see below)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = read_jsonl(DATA / "traces.jsonl")
    # Later rows win: the retry pass re-collected queries the first pass
    # rejected, and its version is the one that cleared the rubric.
    by_id = {t["id"]: t for t in traces}
    rows_in = list(by_id.values())
    print(f"read {len(traces)} records -> {len(rows_in)} unique queries")

    prompts = {t["system_prompt"] for t in rows_in}
    if len(prompts) != 1:
        raise SystemExit(
            f"traces disagree on the assembled system prompt ({len(prompts)} variants) — "
            "the harness drifted mid-collection and the data is not trainable as one set"
        )
    system_prompt = prompts.pop()

    from fingerprint import capture

    _sys_now, tools = capture()
    if _sys_now != system_prompt:
        raise SystemExit(
            "the harness no longer assembles the prompt the traces were collected "
            "under — re-collect, or check out the commit they were collected at"
        )
    print(f"system prompt {len(system_prompt)} chars, {len(tools)} tool schemas — matches live harness")

    rows: list[dict] = []
    bad: list[str] = []
    for t in rows_in:
        msgs = [{"role": "system", "content": system_prompt}] + t["messages"]
        if msgs[-1]["role"] != "assistant":
            bad.append(f"{t['id']}: last message is {msgs[-1]['role']}, must be assistant")
            continue
        if not any(m["role"] == "assistant" for m in msgs):
            bad.append(f"{t['id']}: no assistant turn to train on")
            continue
        # Every tool result must answer a call that exists, or the chat template
        # renders a dangling id and the row teaches a malformed conversation.
        called = {c["id"] for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
        results = {m["tool_call_id"] for m in msgs if m["role"] == "tool"}
        if called != results:
            bad.append(f"{t['id']}: {len(called)} tool calls vs {len(results)} results")
            continue
        rows.append({"messages": msgs, "tools": tools, "_id": t["id"], "_cat": t["cat"]})

    if bad:
        print("\ndropped:")
        for b in bad:
            print(f"  {b}")

    random.Random(args.seed).shuffle(rows)
    holdout = rows[: args.holdout]
    train = rows[args.holdout:]

    def write(path: Path, subset: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(dump_row({"messages": r["messages"], "tools": r["tools"]}) + "\n")

    write(DATA / "sft_train.jsonl", train)
    if holdout:
        write(DATA / "sft_holdout.jsonl", holdout)

    # Re-read exactly as a line-based consumer would; a row that does not
    # survive the round trip is a row that will silently break in training.
    reread = read_jsonl(DATA / "sft_train.jsonl")
    assert len(reread) == len(train), f"round trip lost rows: {len(train)} -> {len(reread)}"
    raw = (DATA / "sft_train.jsonl").read_text()
    assert len(raw.splitlines()) == len(raw.split("\n")) - 1, "a row still contains a line breaker"

    tok = load_tokenizer()

    lengths = []
    for r in train:
        n = tok(json.dumps(r["messages"], ensure_ascii=False)) + tok(json.dumps(r["tools"]))
        lengths.append((n, r["_id"], r["_cat"]))
    lengths.sort()
    vals = [n for n, _, _ in lengths]
    total_assistant = sum(
        1 for r in train for m in r["messages"] if m["role"] == "assistant"
    )

    print(f"\nwrote {len(train)} training rows -> {DATA/'sft_train.jsonl'}"
          f" ({(DATA/'sft_train.jsonl').stat().st_size/1e6:.1f} MB)")
    if holdout:
        print(f"wrote {len(holdout)} held-out rows -> {DATA/'sft_holdout.jsonl'}")
    print(f"supervised assistant turns: {total_assistant}")
    print(f"by category: {dict(Counter(r['_cat'] for r in train))}")

    print(f"\nsequence length (tokens)")
    print(f"  min {vals[0]:,}   median {vals[len(vals)//2]:,}   "
          f"p90 {vals[int(0.9*len(vals))]:,}   max {vals[-1]:,}")
    for cap in (16384, 32768, 65536):
        over = sum(1 for v in vals if v > cap)
        print(f"  over {cap:>6,}: {over:>3} rows ({over/len(vals):.0%})")
    if vals[-1] > 32768:
        print("\n  longest rows:")
        for n, qid, cat in lengths[-5:]:
            print(f"    {n:>8,}  {qid:<8} {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
