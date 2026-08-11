"""Split the labelled queries and emit the ART/W&B serverless SFT dataset.

Two things happen here that decide whether the eval number at the end means
anything:

1. Every query from the original hand-written seed set is forced into *train*.
   Those hundred rows were read one by one while the system prompt was being
   rewritten, so the prompt is partly fitted to them. Scoring on them would
   measure that fit, not the model.
2. The remaining rows are split stratified by label signature, so the test set
   holds the same mix of weather / stock / currency / entity / nothing as the
   training set rather than whatever a random draw produced.

The training file is ART's format: one JSON object per line with `messages`,
last message from the assistant. The system and user messages come from
`build_predictor_messages`, the same function production calls, so the prompt
the LoRA trains on is byte-identical to the one it will be served with.

    python finetune/widget_predictor/build_dataset.py --test-size 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parent / "dataset"
dotenv.load_dotenv()

from core.widget_predictor import (  # noqa: E402
    build_predictor_messages,
    parse_prediction,
    render_prediction,
)

def read_labelled(path: Path) -> list[tuple[str, str | None, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        rows.append((fields[0].strip(), fields[1].strip() or None, fields[2].strip()))
    return rows


def signature(label: str) -> str:
    """Stratification key: which tools fire, ignoring their arguments."""
    tools = sorted(w["tool"] for w in parse_prediction(label))
    return "+".join(tools) or "none"


def write_tsv(path: Path, rows: list[tuple[str, str | None, str]], header: str) -> None:
    lines = [f"# {header}", "# Columns: query <TAB> location <TAB> label"]
    lines += [f"{q}\t{loc or ''}\t{label}" for q, loc, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", type=Path, default=DATA / "ground_truth.tsv")
    ap.add_argument("--seed-set", type=Path, default=DATA / "seed_queries.tsv",
                    help="queries that must stay in train (prompt was tuned on them)")
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_labelled(args.labelled)
    contaminated = {
        line.split("\t")[0].strip()
        for line in args.seed_set.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    eligible = [r for r in rows if r[0] not in contaminated]
    forced_train = [r for r in rows if r[0] in contaminated]
    print(f"{len(rows)} labelled rows: {len(forced_train)} pinned to train "
          f"(prompt tuned on them), {len(eligible)} eligible for test")

    # Stratified draw: take each signature's share of the test set.
    by_sig: dict[str, list] = defaultdict(list)
    for row in eligible:
        by_sig[signature(row[2])].append(row)
    rng = random.Random(args.seed)
    test: list = []
    for sig, group in sorted(by_sig.items()):
        rng.shuffle(group)
        take = round(args.test_size * len(group) / len(eligible))
        test.extend(group[:take])
    test_keys = {r[0] for r in test}
    train = forced_train + [r for r in eligible if r[0] not in test_keys]

    write_tsv(DATA / "test.tsv", test, "Held-out test set. Never train on this file.")
    write_tsv(DATA / "train.tsv", train, "Training split.")

    sft_path = DATA / "sft_train.jsonl"
    with sft_path.open("w", encoding="utf-8") as fh:
        for query, location, label in train:
            system, user = build_predictor_messages(query, user_location=location)
            canonical = render_prediction(parse_prediction(label))
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": system[1]},
                {"role": "user", "content": user[1]},
                {"role": "assistant", "content": canonical},
            ]}, ensure_ascii=False) + "\n")

    print(f"\ntrain {len(train)}  test {len(test)}")
    print(f"{'signature':<18} {'train':>6} {'test':>5}")
    tr, te = Counter(signature(r[2]) for r in train), Counter(signature(r[2]) for r in test)
    for sig in sorted(set(tr) | set(te), key=lambda s: -tr[s]):
        print(f"{sig:<18} {tr[sig]:>6} {te[sig]:>5}")
    print(f"\nwrote {sft_path}, {DATA / 'train.tsv'}, {DATA / 'test.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
