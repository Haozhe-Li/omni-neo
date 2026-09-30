"""LoRA SFT of the voice agent on W&B Serverless Training.

    python finetune/voice_agent/train.py               # 3 epochs on dataset/sft_train.jsonl
    python finetune/voice_agent/train.py --epochs 3 --name omni-voice-v1

Trains on `dataset/sft_train.jsonl` (the clean rows `filter.py` writes). Loss
masking is ART's default — every assistant turn, so the spoken lead-in, the
tool call and the final answer are all taught, not just the last message.

Every row carries the production live-call system prompt and tool schemas, so
the adapter is trained for exactly the agent `core/voice/agent.py` builds for a
call (`voice_call_agent`). The typed-continuation agent has a different prompt
and no `end_call`; a LoRA has one compatibility key, so it is not covered.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

BASE_MODEL = "google/gemma-4-26B-A4B-it"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--file", default=str(HERE / "dataset" / "sft_train.jsonl"))
    args = ap.parse_args()

    import art
    from art.serverless.backend import ServerlessBackend
    from art.utils.sft import train_sft_from_file

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"{path} missing — run collect.py then filter.py first")
    n = sum(1 for line in path.read_text(encoding="utf-8").split("\n") if line.strip())
    print(f"{n} rows from {path.name}, {args.epochs} epochs, batch {args.batch_size}, lr {args.lr}")

    name = args.name or f"omni-voice-{time.strftime('%m%d-%H%M')}"
    model = art.TrainableModel(
        name=name,
        project=os.getenv("WANDB_PROJECT", "omni-voice-agent"),
        base_model=BASE_MODEL,
    )
    print(f"registering {name} on {BASE_MODEL} …")
    await model.register(ServerlessBackend())

    started = time.perf_counter()
    await train_sft_from_file(
        model=model,
        file_path=str(path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        peak_lr=args.lr,
        schedule_type="cosine",
        warmup_ratio=0.1,
        verbose=True,
    )
    print(f"\ntrained in {time.perf_counter() - started:.0f}s")
    try:
        print(f"inference name: {model.get_inference_name()}")
    except Exception as e:  # noqa: BLE001
        print(f"(could not resolve inference name: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
