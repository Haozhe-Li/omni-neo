"""Measure newly added cases and file them into the runs they belong with.

    python -m evals.backfill_cases --label rebase-2026-08-12 \
        --case language/follow-zh-pure language/follow-en-pure ...

    python -m evals.backfill_cases --label ... --case ... --dry-run
    python -m evals.backfill_cases --label ... --restore snapshot.json

Why this exists rather than just running the CLI: `eval_runs.score` is computed
per run, and `v_eval_model_leaderboard` groups by `run_id`. A plain
`--case <new ids>` invocation therefore produces a second run row per model
whose score covers only the new cases, and nothing in the schema knows how to
combine the two. Attaching instead makes the original run cover the whole
rubric, which is what a leaderboard reader assumes it already does.

## What it overwrites, and why that needs care

This mutates recorded measurements. `eval_cases` carries `rubric_version`
precisely so a rubric edit is *visible* rather than silently changing what old
scores meant, and rewriting a run's score works against that grain. Two things
make it defensible here, and neither is automatic:

  - the added cases are new coverage, not a re-grading of what was already
    measured, so no existing per-case score moves;
  - every touched run gets a `notes` line naming the cases, the date, and the
    fact that they were produced by a later checkout than `prompt_sha`.

It is still a one-way door for the aggregate, so a snapshot of every field this
script can change is written *before* the first write, and `--restore` puts
them all back. Take the snapshot seriously: it is the only copy.

## The number is not comparable to what it replaces

`scoring.summarize` scores a run as the **mean of suite means** — every suite
counts once regardless of how many cases it holds. Adding a suite therefore
reweights every existing suite (11 -> 12 suites moves each from 1/11 to 1/12),
so a backfilled score differs from the one it replaces for two independent
reasons: the new cases, and the reweighting. Adding cases to a suite that
already exists only moves that one suite's mean.

The script reports both effects separately so the two are not confused.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import date

from evals.cli import _load_env  # env first: core.* builds clients at import

_load_env()

from evals.cli import _arm_prompt_guard, parse_args as cli_parse_args, run_model  # noqa: E402
from evals.config import load_cases  # noqa: E402
from evals.models import resolve_models  # noqa: E402
from evals import store  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

# Fields this script can change. The snapshot captures exactly these, so a
# restore is guaranteed to be complete rather than approximately complete.
MUTATED_FIELDS = (
    "score",
    "pass_rate",
    "suite_scores",
    "n_cases",
    "n_errors",
    "total_latency_ms",
    "total_cost_usd",
    "notes",
)


def target_runs(label: str, models: list[str] | None) -> dict[str, dict]:
    """The newest finished run per model carrying `label`."""
    from core.database.supabase_client import supabase

    rows = (
        supabase.table("eval_runs")
        .select(",".join(("run_id", "model_label", "started_at") + MUTATED_FIELDS))
        .eq("label", label)
        .eq("status", "done")
        .order("started_at", desc=True)
        .execute()
        .data
        or []
    )
    newest: dict[str, dict] = {}
    for r in rows:  # ordered newest first, so first wins
        newest.setdefault(r["model_label"], r)
    if models:
        missing = [m for m in models if m not in newest]
        if missing:
            raise SystemExit(
                f"no finished run labelled {label!r} for: {', '.join(missing)}\n"
                f"available: {', '.join(sorted(newest))}"
            )
        newest = {m: newest[m] for m in models}
    return newest


def recompute(run_id: str) -> dict:
    """Rebuild a run's summary from every case score it now holds.

    Deliberately re-read from `eval_case_scores` rather than merged in memory:
    the old cases' numbers must come from what was actually recorded, not from
    anything this process computed.

    Mirrors `scoring.summarize` exactly — mean of suite means, and a pass_rate
    that is the flat mean over cases. If that function changes, this must too;
    they are two implementations of one definition, which is a real cost and
    the reason this is the only place that duplicates it.
    """
    from core.database.supabase_client import supabase

    rows = (
        supabase.table("eval_case_scores")
        .select("case_id,suite,score_mean,pass_rate,n_errors")
        .eq("run_id", run_id)
        .execute()
        .data
        or []
    )
    if not rows:
        raise SystemExit(f"run {run_id} has no case scores — refusing to write a summary")
    by_suite: dict[str, list[float]] = {}
    for r in rows:
        by_suite.setdefault(r["suite"], []).append(float(r["score_mean"]))
    suite_scores = {s: round(statistics.fmean(v), 4) for s, v in by_suite.items()}
    return {
        "score": round(statistics.fmean(list(suite_scores.values())), 4),
        "pass_rate": round(statistics.fmean([float(r["pass_rate"]) for r in rows]), 4),
        "suite_scores": suite_scores,
        "n_cases": len(rows),
        "n_errors": sum(int(r["n_errors"] or 0) for r in rows),
    }


def clear_prior_attempt(run_id: str, case_ids: list[str]) -> int:
    """Drop any result rows these cases already left in this run.

    `save_result` inserts, it does not upsert, so a second attempt at the same
    case in the same run appends rather than replaces. That is not hypothetical:
    an interrupted backfill leaves rows for whatever it got through, and
    re-running would leave the run claiming more results than it has cases —
    `v_eval_model_leaderboard` derives n_results, error_rate, the latency
    percentiles and the cost sum from exactly those rows.

    `eval_checks.result_id` is ON DELETE CASCADE, so the checks go with them.
    `eval_case_scores` needs no cleanup: it upserts on (run_id, case_id).
    """
    from core.database.supabase_client import supabase

    existing = (
        supabase.table("eval_results").select("result_id")
        .eq("run_id", run_id).in_("case_id", case_ids).execute().data or []
    )
    if existing:
        supabase.table("eval_results").delete().eq("run_id", run_id).in_(
            "case_id", case_ids
        ).execute()
    return len(existing)


def _runner_args(args, case_ids: list[str], run_id: str):
    """A CLI arg namespace for one attached model run."""
    ns = cli_parse_args([])
    ns.cases = case_ids
    ns.suites = None
    ns.smoke = False
    ns.repeats = args.repeats
    ns.concurrency = 1
    ns.case_delay = args.case_delay
    ns.tool_cache = args.tool_cache
    ns.judge = args.judge
    ns.judge_model = args.judge_model
    ns.no_supabase = False
    ns.label = None
    ns.attach_run_id = run_id
    return ns


async def main_async(args) -> int:
    suite = load_cases()
    cases = suite.filter(None, args.cases)
    if len(cases) != len(args.cases):
        found = {c.id for c in cases}
        raise SystemExit(f"unknown case id(s): {sorted(set(args.cases) - found)}")

    runs = target_runs(args.label, args.models)
    specs = resolve_models(list(runs))
    _arm_prompt_guard()

    new_suites = sorted({c.suite for c in cases})
    print(f"backfilling {len(cases)} case(s) into {len(runs)} run(s) labelled {args.label!r}")
    print(f"  cases:  {', '.join(c.id for c in cases)}")
    print(f"  suites touched: {', '.join(new_suites)}")
    for label, r in runs.items():
        print(f"  {label:<24} run={r['run_id'][:8]}  score={r['score']}  n_cases={r['n_cases']}")

    if args.dry_run:
        print("\n--dry-run: nothing was run and nothing was written")
        return 0

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    snap_path = os.path.join(
        SNAPSHOT_DIR, f"pre-backfill-{args.label}-{int(time.time())}.json"
    )
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(
            {"label": args.label, "cases": args.cases,
             "runs": {k: {f: v.get(f) for f in ("run_id",) + MUTATED_FIELDS}
                      for k, v in runs.items()}},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nsnapshot -> {snap_path}")
    print("  restore with: python -m evals.backfill_cases --restore " + snap_path)

    ctx = store.RunContext()
    store.upsert_cases(suite, ctx)  # the new cases must exist in the registry

    from evals.pricing import load_pricing as load_pricing_yaml

    pricing_version = load_pricing_yaml().version
    from core.database.supabase_client import supabase

    note = (
        f"{date.today().isoformat()}: backfilled {len(cases)} case(s) "
        f"({', '.join(c.id for c in cases)}) added to the suite after this run. "
        f"Those cases were produced by a later checkout than prompt_sha records; "
        f"every other case score is the original measurement."
    )

    changed = []
    for spec in specs:
        run = runs[spec.label]
        print(f"\n=== {spec.label} -> run {run['run_id'][:8]} ===")
        before = {f: run.get(f) for f in MUTATED_FIELDS}
        started = time.perf_counter()
        out = {}
        if not args.recompute_only:
            stale = clear_prior_attempt(run["run_id"], args.cases)
            if stale:
                print(f"  cleared {stale} result row(s) from an earlier attempt")
            out = await run_model(
                spec, cases, suite, _runner_args(args, args.cases, run["run_id"]), pricing_version
            )

        # run_model already called finish_run, which wrote a summary covering
        # only the new cases. Overwrite it with one covering the whole run.
        summary = recompute(run["run_id"])
        patch = dict(summary)
        # Written whenever the summary covers backfilled cases, including on a
        # `--recompute-only` re-apply after `--restore` — a run whose score
        # includes them but whose notes don't say so is the failure mode this
        # whole script is supposed to avoid. Appended only once, so repeated
        # runs don't stack duplicates.
        if note not in (before["notes"] or ""):
            patch["notes"] = ((before["notes"] + "\n") if before["notes"] else "") + note
        if not args.recompute_only:
            patch["total_latency_ms"] = (
                int(before["total_latency_ms"] or 0) + int((time.perf_counter() - started) * 1000)
            )
            if out.get("cost_usd") is not None:
                patch["total_cost_usd"] = round(
                    float(before["total_cost_usd"] or 0) + out["cost_usd"], 6
                )
        supabase.table("eval_runs").update(patch).eq("run_id", run["run_id"]).execute()

        print(f"  score  {before['score']} -> {summary['score']}   "
              f"n_cases {before['n_cases']} -> {summary['n_cases']}")
        for s, after_v in sorted(summary["suite_scores"].items()):
            before_v = (before["suite_scores"] or {}).get(s)
            if before_v == after_v:
                continue
            print(f"  suite {s:<12} {'(new)' if before_v is None else f'{before_v:.4f}'}"
                  f" -> {after_v:.4f}")
        changed.append((spec.label, before["score"], summary["score"],
                        before["n_cases"], summary["n_cases"]))

    try:
        from core.database.db_evals import bump_cache_epoch

        bump_cache_epoch()
    except Exception as e:
        print(f"  (cache epoch bump failed: {type(e).__name__}: {e})")

    # Union across every touched run, not one representative model: a model
    # whose run is missing a suite (a case errored out when it was measured)
    # would otherwise set the denominator for everyone and make this note say
    # something false about the reweighting.
    before_suites: set[str] = set()
    for r in runs.values():
        before_suites |= set(r["suite_scores"] or {})
    n_before = len(before_suites)
    n_after = len(before_suites | set(new_suites))
    print(f"\n{'model':<24}{'before':>9}{'after':>9}{'delta':>9}   cases")
    for label, b, a, cb, ca in changed:
        b = float(b or 0)
        print(f"{label:<24}{b:>9.4f}{a:>9.4f}{a - b:>+9.4f}   {cb} -> {ca}")
    if n_after != n_before:
        print(f"\nNOTE: suites {n_before} -> {n_after}. The overall score is a mean of "
              f"suite means, so every pre-existing suite's weight changed from "
              f"1/{n_before} to 1/{n_after}. Part of every delta above is that "
              f"reweighting, not model behaviour on the new cases.")
    print(f"\nsnapshot kept at {snap_path}")
    return 0


def restore(path: str) -> int:
    from core.database.supabase_client import supabase

    with open(path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    for label, row in snap["runs"].items():
        patch = {f: row.get(f) for f in MUTATED_FIELDS}
        supabase.table("eval_runs").update(patch).eq("run_id", row["run_id"]).execute()
        print(f"restored {label:<24} score={patch['score']} n_cases={patch['n_cases']}")
    print("\nNOTE: eval_results / eval_checks / eval_case_scores rows written by the "
          "backfill are NOT removed — only the run summary is restored. Delete them "
          "by case_id if you need the run's detail rows to match too.")
    try:
        from core.database.db_evals import bump_cache_epoch

        bump_cache_epoch()
    except Exception:
        pass
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", help="run label to attach to, e.g. rebase-2026-08-12")
    p.add_argument("--case", nargs="*", dest="cases", default=[], help="case ids to measure")
    p.add_argument("--models", nargs="*", default=None,
                   help="restrict to these model labels (default: every model with that label)")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--case-delay", type=float, default=10.0)
    p.add_argument("--tool-cache", dest="tool_cache", action="store_true", default=True)
    p.add_argument("--no-tool-cache", dest="tool_cache", action="store_false")
    p.add_argument("--judge", dest="judge", action="store_true", default=True)
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--dry-run", action="store_true", help="show the plan, write nothing")
    # Rebuilds run summaries from the case scores already stored, running no
    # model and spending nothing. Two uses: recovering after a crash between
    # `run_model` and the summary update (which would otherwise leave a run
    # scored on just the new cases), and re-applying a backfill after
    # `--restore` without paying for the measurements twice.
    p.add_argument("--recompute-only", action="store_true",
                   help="recompute run summaries from stored case scores; run no cases")
    p.add_argument("--restore", default=None, help="restore run summaries from a snapshot file")
    args = p.parse_args()

    if args.restore:
        return restore(args.restore)
    if not args.label or not args.cases:
        raise SystemExit("--label and --case are both required (or use --restore)")
    if args.judge_model is None:
        from evals.judge import DEFAULT_JUDGE_MODEL

        args.judge_model = DEFAULT_JUDGE_MODEL
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
