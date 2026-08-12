"""LoRA SFT on W&B Serverless Training.

    python finetune/pro_agent/train.py --smoke      # 4 rows, incl. the longest
    python finetune/pro_agent/train.py --epochs 3

`--smoke` exists to answer one unpublished question before the real run: what
sequence length does serverless training accept? 19% of our rows exceed 32k and
the longest is 208k tokens. Truncating tool results does not fix it — the length
comes from the *number* of pages read, not their size, so even a 1,200-token cap
leaves rows over 32k. If the backend rejects or silently truncates long rows,
the heaviest deep-research and budget-exhausted trajectories are exactly what
gets lost, and those are the behaviours the student is missing most.

Base model is `Qwen/Qwen3-30B-A3B-Instruct-2507` — the only serverless-trainable
model with room for this agent (the other, OpenPipe/Qwen3-14B-Instruct, caps at
32.8k on W&B Inference).

Loss masking is ART's default: **every** assistant turn, not just the last. The
tool-calling turns are most of what is being taught. `assistant_turns="last"`
would throw that away; the parameter is documented but absent from the installed
0.5.18 anyway, and the default is already what we want.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
dotenv.load_dotenv(ROOT / ".env")

DATA = HERE / "dataset"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def build_smoke_file(src: Path, dst: Path, n: int = 4, include_long: bool = True) -> tuple[int, int]:
    """A tiny file: the shortest few, optionally plus the longest row.

    `include_long=True` is the probe — the longest row is the whole point, since
    a smoke run of only short trajectories would pass and tell us nothing about
    the sequence-length limit.

    `include_long=False` is the *control*, and it has to be a real option rather
    than something assembled by hand: the first attempt at isolating the long
    row failed because this function force-included it regardless, so the
    "control" silently re-ran the same experiment.
    """
    from build_dataset import dump_row, read_jsonl

    rows = read_jsonl(src)
    sized = sorted(rows, key=lambda r: len(json.dumps(r, ensure_ascii=False)))
    picked = sized[: n - 1] + [sized[-1]] if include_long else sized[:n]
    with dst.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(dump_row(r) + "\n")
    biggest = max(len(json.dumps(r, ensure_ascii=False)) for r in picked)
    return len(picked), biggest


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-long", action="store_true",
                    help="control: shortest rows only, excludes the long-sequence probe")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--file", default=None,
                    help="train on an explicit jsonl instead of the full set")
    args = ap.parse_args()

    import art
    from art.serverless.backend import ServerlessBackend
    from art.utils.sft import train_sft_from_file

    src = DATA / "sft_train.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing — run build_dataset.py first")

    if args.file:
        path = Path(args.file)
        n = sum(1 for line in path.read_text().split("\n") if line.strip())
        print(f"explicit file: {n} rows from {path.name}")
    elif args.smoke:
        path = DATA / ("sft_short.jsonl" if args.no_long else "sft_smoke.jsonl")
        n, biggest = build_smoke_file(src, path, include_long=not args.no_long)
        print(f"smoke: {n} rows, largest {biggest:,} chars")
    else:
        path = src
        n = sum(1 for line in path.read_text().split("\n") if line.strip())
        print(f"full: {n} rows")

    name = args.name or (
        "omni-pro-smoke" if args.smoke else f"omni-pro-{time.strftime('%m%d-%H%M')}"
    )
    model = art.TrainableModel(
        name=name,
        project=os.getenv("WANDB_PROJECT", "omni-pro-agent"),
        base_model=BASE_MODEL,
    )
    print(f"registering {name} on {BASE_MODEL} …")
    await model.register(ServerlessBackend())

    started = time.perf_counter()
    await train_sft_from_file(
        model=model,
        file_path=str(path),
        # --smoke is a probe and always 1 epoch; --file is just an input
        # override and must respect --epochs like any other run.
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        peak_lr=args.lr,
        schedule_type="cosine",
        warmup_ratio=0.1,
        verbose=True,
    )
    print(f"\ntrained in {time.perf_counter() - started:.0f}s")
    try:
        print(f"inference name: {model.get_inference_name()}")
    except Exception as e:
        print(f"(could not resolve inference name: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
