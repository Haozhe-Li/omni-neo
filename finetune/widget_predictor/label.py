"""Label widget-predictor queries with a strong teacher model.

The student (`gpt-oss-20b`, and later a fine-tuned LoRA) can only be as good as
its labels, so the teacher is a frontier model — self-distilling from the model
already in production would just make its current mistakes more consistent.

Both teacher and student go through `core.widget_predictor.classify`, which
means they see the byte-identical prompt production uses and their output runs
through the same parser. Whatever prompt this script labelled with is the prompt
the fine-tune must be served with.

    # a few rows, teacher + student side by side, for human review
    python finetune/widget_predictor/label.py --limit 20 --compare

    # full run, 3 votes per query
    python finetune/widget_predictor/label.py --samples 3 --compare

Output is JSONL, one record per query, with the canonical target string already
rendered in `target` — that field is the SFT completion.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parent / "dataset"
dotenv.load_dotenv()

from core import llm  # noqa: E402
from core.widget_predictor import classify, render_prediction  # noqa: E402

DEFAULT_INPUT = DATA / "queries_new.tsv"
DEFAULT_OUTPUT = DATA / "labelled.jsonl"
# The review file IS the ground truth: this overwrites it, so re-labelling
# discards hand corrections made to the previous round.
DEFAULT_REVIEW = DATA / "ground_truth.tsv"


def read_queries(path: Path) -> list[tuple[str, str | None]]:
    """Read `query <TAB> location?` lines, skipping blanks and # comments."""
    rows: list[tuple[str, str | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        query, _, location = line.partition("\t")
        rows.append((query.strip(), location.strip() or None))
    return rows


async def label_one(
    query: str,
    location: str | None,
    teacher,
    samples: int,
    student: bool,
) -> dict:
    """Label one query: `samples` teacher votes, plus the student's answer."""
    votes = await asyncio.gather(
        *(classify(query, user_location=location, model=teacher, timeout=90.0)
          for _ in range(samples))
    )
    rendered = [render_prediction(v) for v in votes]
    counts = Counter(rendered)
    label, agree = counts.most_common(1)[0]

    record = {
        "query": query,
        "user_location": location,
        "target": label,
        "teacher_agreement": f"{agree}/{samples}",
        "consistent": agree == samples,
    }
    if samples > 1:
        record["teacher_votes"] = rendered
    if student:
        record["student"] = render_prediction(await classify(query, user_location=location))
        record["match"] = record["student"] == label
    return record


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("-r", "--review", type=Path, default=DEFAULT_REVIEW,
                    help="human-editable ground-truth TSV")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N queries")
    ap.add_argument("--samples", type=int, default=1, help="teacher votes per query")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--teacher", default="gpt_5_6_luna", help="variable name in core/llm.py")
    ap.add_argument("--compare", action="store_true", help="also run the production student")
    args = ap.parse_args()

    teacher = getattr(llm, args.teacher)
    rows = read_queries(args.input)
    if args.limit:
        rows = rows[: args.limit]
    print(f"labelling {len(rows)} queries with {args.teacher} "
          f"({args.samples} sample{'s' if args.samples > 1 else ''} each)\n")

    sem = asyncio.Semaphore(args.concurrency)

    async def guarded(query, location):
        async with sem:
            return await label_one(query, location, teacher, args.samples, args.compare)

    records = await asyncio.gather(*(guarded(q, loc) for q, loc in rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Rows the human should look at hardest sort to the top: teacher/student
    # disagreement first, then any query the teacher itself was unstable on.
    ranked = sorted(records, key=lambda r: (r.get("match", True), r["consistent"]))
    write_review(ranked, args.review, args.compare)

    header = f"{'':2} {'query':<26} {'loc':<14} {'teacher label':<62}"
    if args.compare:
        header += "  student label"
    print(header)
    print("-" * (len(header) + 4))
    for rec in ranked:
        flag = "" if rec.get("match", True) and rec["consistent"] else "!!"
        line = (f"{flag:2} {rec['query'][:25]:<26} {(rec['user_location'] or '')[:13]:<14} "
                f"{rec['target']:<62}")
        if args.compare:
            line += f"  {rec['student']}"
        print(line)

    if args.compare:
        agreed = sum(r["match"] for r in records)
        print(f"\nstudent agrees with teacher on {agreed}/{len(records)} "
              f"({agreed / len(records):.0%})")
    if args.samples > 1:
        stable = sum(r["consistent"] for r in records)
        print(f"teacher self-consistent on {stable}/{len(records)}")
    print(f"\nwrote {args.output} and {args.review}")
    return 0


def write_review(records: list[dict], path: Path, compared: bool) -> None:
    """Write the human-editable ground-truth file.

    Three tab-separated columns — query, location, label — and everything the
    reviewer needs to judge a row (teacher agreement, what the student said)
    pushed into trailing `#` comments so that fixing a label is a one-field
    edit that can't break the file. This edited file, not the teacher's raw
    output, is what the dataset builder reads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ground truth for the widget predictor. Columns: query <TAB> location <TAB> label",
        "# Edit the third column where the teacher got it wrong. Everything after",
        "# a '#' is a note and is ignored by the reader.",
        "# ?? = teacher and student disagreed, or the teacher contradicted itself.",
        "",
    ]
    for rec in records:
        flag = "" if rec.get("match", True) and rec["consistent"] else "?? "
        note = f"{flag}teacher {rec['teacher_agreement']}"
        if compared and not rec.get("match", True):
            note += f" | student: {rec['student']}"
        lines.append(
            f"{rec['query']}\t{rec['user_location'] or ''}\t{rec['target']}\t# {note}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
