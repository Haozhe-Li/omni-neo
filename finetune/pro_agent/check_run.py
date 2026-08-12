"""Check a serverless training run from scratch — no local process needed.

    python finetune/pro_agent/check_run.py                    # latest run
    python finetune/pro_agent/check_run.py --name omni-pro-129-0811

Training happens on W&B's machines. The local `train.py` only polls for events
and prints a progress bar, so closing the laptop does not stop the job — but it
does lose the only view of it. This queries the API directly instead, which is
also the view that proved trustworthy when the client's output did not.

The one thing that matters: **`v0` is the untrained identity LoRA** downloaded
at registration. A real fine-tune shows up as `v1` or later, with a checkpoint
at `step > 0` carrying non-empty metrics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

HERE = Path(__file__).resolve().parent
STATE = HERE / "dataset" / "last_run.json"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROJECT = "omni-pro-agent"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--data-url", default=None)
    args = ap.parse_args()

    name, data_url = args.name, args.data_url
    if not name and STATE.exists():
        saved = json.loads(STATE.read_text())
        name = saved.get("name")
        data_url = data_url or saved.get("data_url")
    if not name:
        raise SystemExit("no run name — pass --name or run train.py first")

    from art.serverless.client import Client

    c = Client()
    m = await c.models.create(entity=None, project=PROJECT, name=name,
                              base_model=BASE_MODEL, return_existing=True)
    print(f"model {name}  id={m.id}")

    # Job status. `create` with the same model+data returns the existing job
    # rather than starting a second one, which is what makes recovery possible
    # after the local client is gone.
    # Best-effort: the job handle is only recoverable while the data artifact
    # still resolves. It 400s once that artifact is gone, which says nothing
    # about the trained adapter — so a failure here must not stop the
    # checkpoint and artifact checks below, which are the ones that matter.
    if data_url:
      try:
        job = await c.sft_training_jobs.create(
            model_id=m.id, training_data_url=data_url,
            config={"batch_size": 1, "learning_rate": [1e-4]},
        )
        st = await c.get(f"/preview/training-jobs/{job.id}", cast_to=object)
        print(f"job    {job.id}")
        print(f"  status={st.get('status')}  start={st.get('training_start')}  "
              f"end={st.get('training_end')}")
        if st.get("error_message"):
            print(f"  ERROR: {st['error_message']}")
        last = None
        n_steps = 0
        from openai._types import NOT_GIVEN
        async for ev in c.training_jobs.events.list(training_job_id=job.id, after=NOT_GIVEN):
            if ev.type == "gradient_step":
                n_steps += 1
                last = ev.data
            elif ev.type in ("training_ended", "training_failed"):
                print(f"  terminal event: {ev.type} {json.dumps(ev.data, default=str)[:200]}")
        print(f"  gradient steps seen: {n_steps}")
        if last:
            keep = ("gradient_step", "loss", "num_tokens", "num_trainable_tokens",
                    "tokens_per_second", "learning_rate")
            print(f"  last: { {k: last[k] for k in keep if k in last} }")
      except Exception as e:
        print(f"  (job handle unavailable: {type(e).__name__}) — checkpoints below still authoritative")

    cps = await c.models.checkpoints.list(model_id=m.id)
    items = list(getattr(cps, "data", cps) or [])
    trained = [cp for cp in items if cp.step > 0]
    print(f"\ncheckpoints: {len(items)}  (trained: {len(trained)})")
    for cp in items:
        print(f"  step={cp.step}  metrics={cp.metrics}")
    await c.close()

    try:
        import wandb

        api = wandb.Api()
        ent = m.entity
        coll = api.artifact_collection("lora", f"{ent}/{PROJECT}/{name}")
        vers = [a.version for a in coll.artifacts()]
        print(f"\nlora artifact versions: {vers}")
        if any(v != "v0" for v in vers):
            print("  ✅ 微调产物存在 (v1+)")
            print(f"  serve as: wandb-artifact:///{ent}/{PROJECT}/{name}:"
                  f"{sorted(v for v in vers if v != 'v0')[-1]}")
        else:
            print("  ❌ 只有 v0 — 那是注册时的 identity LoRA,不是训练结果")
    except Exception as e:
        print(f"\nartifact check failed: {type(e).__name__} {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
