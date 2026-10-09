"""Build the Gemma SFT file from thumbs-up examples in Postgres.

    python finetune/rix_gemma/build_dataset.py                 # current harness, sensible defaults
    python finetune/rix_gemma/build_dataset.py --only-accepted # only rows a human marked 'accepted'
    python finetune/rix_gemma/build_dataset.py --report        # counts only, write nothing

Reads `sft_examples` (written by POST /api/threads/{id}/feedback, see
core/sft_capture.py), writes under the gitignored `dataset/`:

    sft_train.jsonl     {"messages": [system, user, assistant(tool_calls), tool, ..., assistant],
                         "tools": [...]}  — one row per example, the shape
                         finetune/voice_agent/train.py (and pro_agent's) consume
    sft_holdout.jsonl   only with --holdout N
    manifest.json       which example ids went into this file, under which harness

Train it with the existing script — it already targets google/gemma-4-26B-A4B-it:

    WANDB_PROJECT=omni-rix-gemma python finetune/voice_agent/train.py \\
        --file finetune/rix_gemma/dataset/sft_train.jsonl --name rix-gemma-v1

## What is filtered, and why those defaults

A thumbs-up is a weak label from an unreviewed click, and what it labels is a
real user's conversation. So the defaults lean toward leaving rows out:

- **One harness.** A LoRA is keyed to exactly one assembled system prompt +
  tool list; rows collected under another are an input the adapter would never
  be served. `--harness current` (default) keeps only rows from the harness this
  checkout assembles; mixing is refused outright, like pro_agent/build_dataset.py.
- **Teacher only.** Every assistant message in the row must come from the
  teacher (`--teacher luna`). The loss covers all of them, so one earlier turn
  from another model would be distilled as if luna had written it.
- **No `<user_memory>`, no attachments** (`--include-memory`, `--include-attachments`
  to override). Memory blocks are personal facts about a real person and uploaded
  documents are their private files; both would be read into the weights, and a
  model can later repeat what it was trained on. The exception is rows from the
  collector page (`source = 'collector'`): their memory and uploaded files were chosen by
  an annotator for the purpose, so those rows keep them — that is the point of collecting
  them. Images still follow `--images` (the default drops them).
- **No images** until the trainer is known to accept them (`--images keep`).
- **Not rejected.** Rows a reviewer marked 'rejected' (finetune/rix_gemma/curate.py)
  never come through; `--only-accepted` additionally drops the unreviewed.

Two things are done to what remains: a thread whose turns 1 and 3 were both
thumbed up yields only the turn-3 row (it contains turn 1, so keeping both
would train the same assistant message twice), and assistant messages that carry
both text and tool calls are split in two for Gemma's chat template — see
`split_lead_ins`.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

DATA = HERE / "dataset"
GEMMA_TOKENIZER = "google/gemma-4-26B-A4B-it"
IMAGE_REF_PREFIX = "omni-image://"


def _load_pro_agent_builder():
    """The pro_agent builder's row-level helpers (water-filling truncation, the
    U+2028-safe writer), loaded by path.

    Not `import build_dataset`: this file is also called build_dataset.py and
    sits first on sys.path, so that import would find itself.
    """
    path = ROOT / "finetune" / "pro_agent" / "build_dataset.py"
    spec = importlib.util.spec_from_file_location("pro_agent_build_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tokenizer():
    """`len(encode(s))` with Gemma's own tokenizer, or a character estimate.

    The estimate (3 chars/token) is only a fallback for a machine that cannot
    fetch the tokenizer; a cap enforced with it is approximate, and the script
    says so rather than pretending otherwise.
    """
    try:
        from transformers import AutoTokenizer

        enc = AutoTokenizer.from_pretrained(GEMMA_TOKENIZER)
        return lambda s: len(enc.encode(s))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: Gemma tokenizer unavailable ({type(e).__name__}); "
              f"token counts and the --cap are estimates (len/3)")
        return lambda s: len(s) // 3


def split_lead_ins(messages: list[dict]) -> list[dict]:
    """[assistant(content + tool_calls)] -> [assistant(content), assistant(tool_calls)].

    Gemma 4's chat template moves text that shares a message with a tool call to
    *after* the tool response, so a row trained as-is teaches "narrate after the
    lookup". Two consecutive assistant messages render as text-then-call, which is
    what the served model has to produce. Same transform and same reason as
    finetune/voice_agent/filter.py::split_lead_ins; the agent prompt discourages
    narrating between tool calls, so this is usually a no-op here.
    """
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls") and (m.get("content") or "").strip():
            out.append({"role": "assistant", "content": m["content"]})
            out.append({**m, "content": ""})
        else:
            out.append(m)
    return out


def drop_prefix_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Within a thread, drop an example whose conversation is a strict prefix of
    a later example's. Keeps the longest, which trains the same assistant turns
    once instead of twice."""
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_thread[r["thread_id"]].append(r)
    keep: list[dict] = []
    dropped = 0
    for group in by_thread.values():
        group.sort(key=lambda r: r["turn"])
        for i, r in enumerate(group):
            n = len(r["messages"])
            covered = any(
                len(o["messages"]) > n and o["messages"][:n] == r["messages"] for o in group[i + 1:]
            )
            if covered:
                dropped += 1
            else:
                keep.append(r)
    return keep, dropped


def fetch_examples(statuses: list[str]) -> list[dict]:
    from core.database import pg

    return pg.fetch_all(
        "SELECT e.id, e.thread_id, e.turn, e.user_id, e.harness_hash, e.models_seen, e.messages, e.source, "
        "       e.has_image, e.has_memory, e.has_attachments, e.compacted, e.tools_used, "
        "       h.system_prompt, h.tools "
        "FROM sft_examples e JOIN harness_snapshots h USING (harness_hash) "
        "WHERE e.status = ANY(%s) ORDER BY e.id",
        (statuses,),
    )


def fetch_images(shas: set[str]) -> dict[str, str]:
    """sha256 -> data URL, for the example images that are being kept."""
    import base64

    from core.database import pg

    if not shas:
        return {}
    rows = pg.fetch_all("SELECT sha256, mime, data FROM sft_images WHERE sha256 = ANY(%s)", (list(shas),))
    return {
        r["sha256"]: f"data:{r['mime']};base64,{base64.b64encode(bytes(r['data'])).decode()}" for r in rows
    }


def rehydrate_images(messages: list[dict], images: dict[str, str]) -> bool:
    """Swap `omni-image://<sha>` references for data URLs in place. False if any
    image is missing, so the caller can drop the row instead of training on a
    dangling reference."""
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            if part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            if url.startswith(IMAGE_REF_PREFIX):
                data_url = images.get(url[len(IMAGE_REF_PREFIX):])
                if data_url is None:
                    return False
                part["image_url"]["url"] = data_url
    return True


def current_harness_hash() -> str:
    from core.harness_snapshot import live_harness

    return asyncio.run(live_harness())[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", default="current",
                    help="'current' (this checkout's harness), or an explicit harness_hash")
    ap.add_argument("--only-accepted", action="store_true", help="drop unreviewed ('pending') rows")
    ap.add_argument("--teacher", default="luna",
                    help="substring every assistant message's model name must contain; '' disables")
    ap.add_argument("--include-memory", action="store_true")
    ap.add_argument("--include-attachments", action="store_true")
    ap.add_argument("--images", choices=["drop", "keep"], default="drop")
    ap.add_argument("--max-per-user", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--cap", type=int, default=30_000,
                    help="water-fill tool results until each row fits this many tokens")
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="print counts, write nothing")
    args = ap.parse_args()

    statuses = ["accepted"] if args.only_accepted else ["accepted", "pending"]
    rows = fetch_examples(statuses)
    print(f"{len(rows)} example(s) with status {statuses}")
    if not rows:
        return 1

    # ── one harness ──
    hashes = Counter(r["harness_hash"] for r in rows)
    print("by harness:", dict(hashes))
    target = current_harness_hash() if args.harness == "current" else args.harness
    print(f"target harness: {target}")
    stale = len(rows) - hashes.get(target, 0)
    rows = [r for r in rows if r["harness_hash"] == target]
    if not rows:
        print("no example was collected under that harness — the prompt/tools/skills changed "
              "since, or pass --harness <hash> to build for an older one")
        return 1

    # ── filters, each counted so a surprising yield is explainable ──
    dropped: Counter = Counter()
    if stale:
        dropped["other harness"] = stale
    kept: list[dict] = []
    for r in rows:
        if args.teacher and not (
            r["models_seen"] and all(args.teacher.lower() in m.lower() for m in r["models_seen"])
        ):
            dropped["non-teacher turn in history"] += 1
        elif r["compacted"]:
            dropped["history was summarised (model saw less than the row)"] += 1
        # Collector rows (source='collector', written by core/routers/collector.py)
        # carry memory an annotator invented for the example, not a real person's
        # stored facts, so the privacy reason for this filter does not apply to them.
        elif r["has_memory"] and r.get("source") != "collector" and not args.include_memory:
            dropped["has <user_memory>"] += 1
        # Same exemption for uploads: a collector row's files are test inputs an
        # annotator chose, not a real person's documents.
        elif r["has_attachments"] and r.get("source") != "collector" and not args.include_attachments:
            dropped["has attachments"] += 1
        elif r["has_image"] and args.images == "drop":
            dropped["has image"] += 1
        else:
            kept.append(r)
    rows = kept

    rows, n_prefix = drop_prefix_rows(rows)
    if n_prefix:
        dropped["superseded by a later turn of the same thread"] = n_prefix

    if args.max_per_user:
        per_user: Counter = Counter()
        capped: list[dict] = []
        for r in rows:
            per_user[r["user_id"]] += 1
            if per_user[r["user_id"]] <= args.max_per_user:
                capped.append(r)
            else:
                dropped[f"over --max-per-user {args.max_per_user}"] += 1
        rows = capped

    images: dict[str, str] = {}
    if args.images == "keep":
        wanted = {
            p["image_url"]["url"][len(IMAGE_REF_PREFIX):]
            for r in rows for m in r["messages"] if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "image_url" and p["image_url"]["url"].startswith(IMAGE_REF_PREFIX)
        }
        images = fetch_images(wanted)

    for reason, n in dropped.most_common():
        print(f"  dropped {n:>4}  {reason}")
    print(f"{len(rows)} example(s) usable")
    if not rows:
        return 1

    system_prompt, tools = rows[0]["system_prompt"], rows[0]["tools"]
    pro = _load_pro_agent_builder()
    tok = load_tokenizer()

    # Truncate with image references still in place (measuring a base64 blob as
    # tokens would make the cap unreachable), then swap the real images in.
    built: list[dict] = []
    n_shrunk = 0
    for r in rows:
        row = {"messages": [{"role": "system", "content": system_prompt}] + split_lead_ins(r["messages"]),
               "tools": tools}
        row, cut = pro.shrink_row(row, args.cap, tok)
        n_shrunk += bool(cut)
        if args.images == "keep" and not rehydrate_images(row["messages"], images):
            print(f"  example {r['id']}: image missing from sft_images, skipped")
            continue
        built.append({"row": row, "src": r})
    if n_shrunk:
        print(f"water-filled {n_shrunk} row(s) to <= {args.cap:,} tokens")
    if not built:
        print("nothing left to write")
        return 1

    lengths = sorted(
        tok(json.dumps(b["row"]["messages"], ensure_ascii=False)) + tok(json.dumps(tools)) for b in built
    )
    print(f"\nsequence length (tokens): min {lengths[0]:,}  median {lengths[len(lengths)//2]:,}  "
          f"max {lengths[-1]:,}")
    tool_use = Counter(t for b in built for t in b["src"]["tools_used"])
    print("tool use across rows:", dict(tool_use.most_common()))
    skill_loads = sum(
        1 for b in built for m in b["row"]["messages"] if m["role"] == "assistant"
        for c in m.get("tool_calls") or []
        if c["function"]["name"] == "read_file" and "/skills/" in c["function"]["arguments"]
    )
    print(f"skill loads (read_file on /skills/): {skill_loads}")
    print(f"by user: {len({b['src']['user_id'] for b in built})} distinct")

    if args.report:
        return 0

    random.Random(args.seed).shuffle(built)
    holdout, train = built[: args.holdout], built[args.holdout:]
    DATA.mkdir(exist_ok=True)

    def write(path: Path, subset: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for b in subset:
                f.write(pro.dump_row(b["row"]) + "\n")

    write(DATA / "sft_train.jsonl", train)
    if holdout:
        write(DATA / "sft_holdout.jsonl", holdout)
    # A line-based reader must see exactly one row per line — see the U+2028
    # note in pro_agent/build_dataset.py.
    assert len(pro.read_jsonl(DATA / "sft_train.jsonl")) == len(train)

    manifest = {
        "harness_hash": target,
        "args": {k: v for k, v in vars(args).items()},
        "train_example_ids": [b["src"]["id"] for b in train],
        "holdout_example_ids": [b["src"]["id"] for b in holdout],
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(train)} train row(s) -> {DATA / 'sft_train.jsonl'}"
          + (f" and {len(holdout)} held out" if holdout else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
