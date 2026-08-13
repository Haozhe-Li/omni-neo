"""Turn hand-written answers in `authored.yaml` into trace records.

    python finetune/pro_agent/author.py            # verify only, writes nothing
    python finetune/pro_agent/author.py --write    # append to dataset/traces.jsonl

Same output schema as `collect.py`, so `build_dataset.py` cannot tell the two
apart — and must not have to. What differs is only where the assistant text
came from.

## Why hand-write at all

For the translation slice of `write-rewrite` there is nothing to distil: no
retrieval, no reasoning chain, one deterministic deliverable. The teacher's
only contribution is the final text, and on that it is unreliable — 4 of 8
collected translations omitted the lead-in line that `_S_WRITING_FORMAT`
requires, and the category checks do not look for it. A 50%-compliant
demonstration teaches that the line is optional, which contradicts the prompt
the adapter is keyed to.

## Why this is not "just trust the author"

Everything around the answer is production's own code, not a hand-copy:

- the system prompt comes from `fingerprint.capture()`, i.e. the *assembled*
  prompt deepagents actually serves, so a record written here carries the same
  `system_prompt_sha` as every collected one;
- the user message comes from `collect.build_message`, which drives
  `core.stream.build_message_content`;
- the answer is then run through the identical gate + category checks, and a
  failing record is refused rather than written.

The gate's `no_prompt_leak` needs arming (`register_sensitive_prompts`) or it
fails closed — same as in `collect.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dotenv
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

from core.utils.citations import all_citations, reset_citation_registry  # noqa: E402
from evals import checks as checks_mod  # noqa: E402
from evals.config import CheckSpec  # noqa: E402
from evals.trace import RunTrace, TurnTrace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import build_message  # noqa: E402
from fingerprint import capture  # noqa: E402
from spec import Spec, load  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
AUTHORED = HERE / "authored.yaml"


def _record(spec: Spec, q, answer: str, system_prompt: str) -> tuple[dict, list[str]]:
    """Build one trace record and the list of checks it failed (empty = good)."""
    # Citation numbering is per-thread global state; a stale registry from the
    # previous query would make `citation_exists` grade this answer against
    # someone else's sources.
    reset_citation_registry(f"authored-{q.id}", 1)
    content, _doc_files, _sources = build_message(spec, q)

    turn = TurnTrace(index=0, query=q.text)
    turn.text = answer
    turn.n_llm_turns = 1
    trace = RunTrace(case_id=q.id, model_label="authored", status="ok")
    trace.turns.append(turn)
    trace.citations = [dict(c) for c in all_citations()]

    specs = [
        CheckSpec(key=c["key"], args=c.get("args") or {}, weight=1, turn="all")
        for c in spec.checks_for(q)
    ]
    failed = [
        f"{s.label}: {r.evidence or r.reason}"
        for s, r in checks_mod.run_checks(trace, specs)
        if not r.passed
    ]

    return {
        "id": q.id, "cat": q.cat, "lang": q.lang, "block": q.block,
        "text": q.text,
        "run_limit": spec.run_limit_for(q),
        "n_tool_calls": 0,
        "blocked_calls": 0,
        "trimmed_blocked": 0,
        # Not a sample index — these mark the record's provenance for anyone
        # reading dataset/traces.jsonl without this file open.
        "chose_sample": -1,
        "n_candidates_passing": 0,
        "authored": True,
        "system_prompt": system_prompt,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": answer},
        ],
    }, failed


def _lead_in_ok(answer: str) -> str:
    """The contract detail no existing check covers: prose before the block.

    Reported rather than enforced — `author.py` writes what `authored.yaml`
    says. It exists so a regression in the hand-written file is visible in the
    same run that produced it.
    """
    if "<textblock" not in answer:
        return "n/a"
    return "ok" if answer.split("<textblock")[0].strip() else "MISSING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="append to dataset/traces.jsonl")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--only", nargs="*", default=None, help="query ids")
    args = ap.parse_args()

    from core.agent import SYSTEM_PROMPTS
    from core.prompt_guard import register_sensitive_prompts

    register_sensitive_prompts(SYSTEM_PROMPTS)

    raw = yaml.safe_load(AUTHORED.read_text(encoding="utf-8"))
    answers: dict[str, str] = {k: v.strip() for k, v in (raw.get("answers") or {}).items()}
    if args.only:
        answers = {k: v for k, v in answers.items() if k in set(args.only)}
    if not answers:
        raise SystemExit("no authored answers matched")

    spec = load()
    by_id = {q.id: q for q in spec.queries}
    unknown = sorted(set(answers) - set(by_id))
    if unknown:
        raise SystemExit(f"authored.yaml has ids not in queries.yaml: {', '.join(unknown)}")

    system_prompt, _tools = capture()

    records, bad = [], 0
    for qid, answer in answers.items():
        q = by_id[qid]
        rec, failed = _record(spec, q, answer, system_prompt)
        status = "ok  " if not failed else "FAIL"
        if failed:
            bad += 1
        print(f"{status} {qid:<8} {q.cat:<15} lead-in={_lead_in_ok(answer):<8} "
              f"{'; '.join(failed[:2])}")
        if not failed:
            records.append(rec)

    print(f"\n{len(records)}/{len(answers)} passed all checks")
    if bad:
        print("refusing to write failing records — fix authored.yaml")
    if not args.write:
        print("(dry run — pass --write to append)")
        return 1 if bad else 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "traces.jsonl"
    existing = set()
    if path.exists():
        for line in path.open(encoding="utf-8"):
            existing.add(json.loads(line)["id"])
    dupes = [r["id"] for r in records if r["id"] in existing]
    if dupes:
        raise SystemExit(
            f"already in {path}: {', '.join(dupes)} — remove them first, "
            "appending would train on the same query twice"
        )
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(records)} -> {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
