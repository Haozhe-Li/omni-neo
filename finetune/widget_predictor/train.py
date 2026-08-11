"""Fine-tune a widget-predictor LoRA on W&B Serverless Training (ART).

Training runs on W&B's managed GPUs and the resulting adapter is hosted
automatically, so there is no artifact upload step and no GPU to provision. The
dataset comes from `build_dataset.py`, whose system prompt is
rendered by the same `build_predictor_messages` production calls — if that
prompt changes, the adapter has to be retrained or it will be served a prompt
it never saw.

    python finetune/widget_predictor/train.py --epochs 4
    python finetune/widget_predictor/train.py \
        --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 --name widget-predictor-30b

Afterwards, point the eval at the trained endpoint:

    python finetune/widget_predictor/evaluate.py --lora <inference-name-printed-below>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parent / "dataset"
dotenv.load_dotenv()

DEFAULT_DATA = DATA / "sft_train.jsonl"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--base-model", default="OpenPipe/Qwen3-14B-Instruct")
    ap.add_argument("--name", default="widget-predictor-14b")
    ap.add_argument("--project", default="omni-widgets")
    ap.add_argument("--epochs", type=int, default=4)
    # 486 rows at batch 8 is ~61 steps/epoch. The completion is only ~20 tokens
    # per row, so the gradient signal is thin and the run wants more passes than
    # a typical SFT — 4 epochs, not 2.
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--peak-lr", type=float, default=1e-4)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "constant"])
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    args = ap.parse_args()

    import art
    from art.serverless.backend import ServerlessBackend
    from art.utils.sft import train_sft_from_file

    rows = sum(1 for line in args.data.open(encoding="utf-8") if line.strip())
    steps = -(-rows * args.epochs // args.batch_size)
    print(f"base model : {args.base_model}")
    print(f"dataset    : {args.data} ({rows} rows)")
    print(f"schedule   : {args.epochs} epochs, batch {args.batch_size}, "
          f"~{steps} steps, peak_lr {args.peak_lr}, {args.schedule}\n")

    model = art.TrainableModel(
        name=args.name, project=args.project, base_model=args.base_model,
    )
    print("registering with serverless backend...")
    await model.register(ServerlessBackend())
    print("registered\n")

    t0 = time.perf_counter()
    await train_sft_from_file(
        model=model,
        file_path=str(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        peak_lr=args.peak_lr,
        schedule_type=args.schedule,
        warmup_ratio=args.warmup_ratio,
        verbose=True,
    )
    print(f"\ntrained in {time.perf_counter() - t0:.0f}s")

    inference_name = model.get_inference_name()
    print(f"inference name: {inference_name}")

    # Smoke the endpoint before trusting any eval numbers: a served adapter that
    # answers with prose, or not at all, is a different failure from a served
    # adapter that answers wrongly.
    client = model.openai_client()
    from core.widget_predictor import build_predictor_messages

    for query in ("weather in tokyo", "腾讯股价", "how to learn python"):
        system, user = build_predictor_messages(query)
        t = time.perf_counter()
        resp = await client.chat.completions.create(
            model=inference_name,
            messages=[{"role": "system", "content": system[1]},
                      {"role": "user", "content": user[1]}],
            max_tokens=128, temperature=0,
        )
        ms = (time.perf_counter() - t) * 1000
        print(f"  {ms:6.0f}ms  {query!r:<22} -> {resp.choices[0].message.content!r}")

    print(f"\nnext:\n  python finetune/widget_predictor/evaluate.py "
          f"--lora {inference_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
