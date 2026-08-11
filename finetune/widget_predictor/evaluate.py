"""Score a widget-predictor backend against the reviewed ground truth.

Every candidate — the production Groq model, the legacy `bind_tools` path, a
teacher, a fine-tuned LoRA — runs through `core.widget_predictor.classify`, so
they all see the identical prompt and are parsed identically. The only thing
that varies is the weights.

    python finetune/widget_predictor/evaluate.py                 # wired-in model
    python finetune/widget_predictor/evaluate.py --model gpt_oss_20b  # pre-LoRA
    python finetune/widget_predictor/evaluate.py --backend tools      # tool-call path

Exact set match is the headline number, but on a 100-row set it carries a
confidence interval of roughly +/-4 points, so two systems within a few points
of each other are not distinguishable here. The per-class breakdown, the
false-positive rate on the negative rows, and the printed mismatch list are what
actually tell you what changed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parent / "dataset"
dotenv.load_dotenv()

from core import llm  # noqa: E402
from core.widget_predictor import (  # noqa: E402
    _ARG_SCHEMAS,
    classify_verbose,
    parse_prediction,
    render_prediction,
)

# The held-out split by default. Point -t at train.tsv only to inspect fit.
DEFAULT_TRUTH = DATA / "test.tsv"


def read_ground_truth(path: Path) -> list[tuple[str, str | None, str]]:
    """Read `query <TAB> location <TAB> label`, ignoring comments and notes."""
    rows: list[tuple[str, str | None, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        query, location, label = fields[0].strip(), fields[1].strip(), fields[2]
        rows.append((query, location or None, label.split("\t#")[0].strip()))
    return rows


def widget_set(label: str) -> set[tuple]:
    """Order-insensitive key for one prediction, so widget order never counts."""
    return {
        (w["tool"], tuple(sorted(w["args"].items())))
        for w in parse_prediction(label)
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--backend", default=None, choices=["json", "tools"])
    ap.add_argument("--model", default=None, help="variable name in core/llm.py")
    ap.add_argument("--lora", default=None,
                    help="wandb-artifact:/// URI of an adapter served by W&B Inference")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dump", type=Path, default=None,
                    help="write per-row results as JSON, for paired comparison")
    args = ap.parse_args()

    if args.lora:
        # Served through the plain OpenAI-compatible endpoint rather than ART's
        # client, because that is how production would call it — the eval and
        # the deployed path should differ only in which file holds the URI.
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=args.lora,
            base_url="https://api.inference.wandb.ai/v1",
            api_key=os.environ["WANDB_API_KEY"],
            temperature=0,
            max_tokens=128,
        )
    else:
        model = getattr(llm, args.model) if args.model else None
    rows = read_ground_truth(args.truth)
    print(f"scoring {len(rows)} rows from {args.truth}"
          f"  backend={args.backend or 'default'}  model={args.model or 'default'}\n")

    sem = asyncio.Semaphore(args.concurrency)

    async def run(query, location, expected):
        async with sem:
            t0 = time.perf_counter()
            got, raw = await classify_verbose(
                query, user_location=location, backend=args.backend,
                model=model, timeout=90.0,
            )
            return {
                "query": query,
                "location": location,
                "expected": expected,
                "got": render_prediction(got),
                "raw": raw,
                "ms": (time.perf_counter() - t0) * 1000,
            }

    results = await asyncio.gather(*(run(q, loc, exp) for q, loc, exp in rows))

    # A prediction is unparseable when the model said something but none of it
    # survived validation — distinct from a deliberate empty answer.
    unparseable = [
        r for r in results
        if not parse_prediction(r["raw"]) and r["raw"].strip() not in ('{"widgets":[]}', "")
    ]

    exact = 0
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    neg_total = neg_fired = 0
    mismatches = []

    for r in results:
        want, have = widget_set(r["expected"]), widget_set(r["got"])
        if want == have:
            exact += 1
        else:
            mismatches.append(r)
        for item in have & want:
            tp[item[0]] += 1
        for item in have - want:
            fp[item[0]] += 1
        for item in want - have:
            fn[item[0]] += 1
        if not want:
            neg_total += 1
            neg_fired += bool(have)

    n = len(results)
    latencies = sorted(r["ms"] for r in results)
    print(f"exact set match     {exact}/{n}  ({exact / n:.0%})")
    print(f"unparseable output  {len(unparseable)}/{n}")
    print(f"false positives on negatives  {neg_fired}/{neg_total} "
          f"({neg_fired / neg_total:.0%} of no-widget queries fired something)")
    print(f"latency  p50 {statistics.median(latencies):.0f}ms  "
          f"p95 {latencies[int(0.95 * (n - 1))]:.0f}ms\n")

    print(f"{'tool':<10} {'TP':>4} {'FP':>4} {'FN':>4}  {'prec':>6} {'recall':>7}")
    for tool in _ARG_SCHEMAS:
        t, f, m = tp[tool], fp[tool], fn[tool]
        prec = t / (t + f) if t + f else 1.0
        rec = t / (t + m) if t + m else 1.0
        print(f"{tool:<10} {t:>4} {f:>4} {m:>4}  {prec:>6.0%} {rec:>7.0%}")

    if args.dump:
        args.dump.write_text(json.dumps(
            {r["query"]: widget_set(r["expected"]) == widget_set(r["got"]) for r in results},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote per-row results to {args.dump}")

    if mismatches:
        print(f"\n{len(mismatches)} mismatches:")
        for r in mismatches:
            loc = f" @{r['location']}" if r["location"] else ""
            print(f"  {r['query']}{loc}")
            print(f"      want {r['expected']}")
            print(f"      got  {r['got']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
