"""Per-metric upper bound across every measured rix version.

    python -m evals.oracle                 # print the table
    python -m evals.oracle --write         # also upsert into eval_oracle
    python -m evals.oracle --ddl           # print the table definition

## What this is

For each case, take the best result any rix version achieved, then aggregate
those bests the way `scoring.summarize` aggregates a real run. The result
answers one question: *if a single model matched the best version on every
case, where would it land?*

## What this is not

It is not a model and it is not a measurement. No model produced these numbers,
and nothing that behaves this way exists. It is deliberately kept out of
`eval_runs`, so it can never reach `v_eval_model_leaderboard` or any other view
that ranks models — the leaderboard's job is to say what was measured.

Read it as headroom, not as a score. Its value is diagnostic: when the oracle
sits far above the best single version, the gap is a *mixing* problem — every
point has already been reached by something, just never all at once — which is
a data-composition question. When the oracle sits close to the best version,
the remaining gap is genuine capability and no amount of reweighting will close
it.

## Why the number is softer than it looks

Two reasons it overstates, both structural:

1. **Max-of-repeats bias.** Each case's score is already a mean over repeats,
   but taking the max across versions still selects for luck. A version that
   got a favourable sample on one case is credited with it permanently.
2. **Cross-harness mixing.** v1/v3/v4 were measured with the old
   `<personalization>` field spelling; only v5 ran under production's. Any
   suite whose best comes from an older version is quoting a number produced
   under a different harness. `stale_harness` marks those rows — treat their
   headroom as provisional until that version is re-measured.

Speed and cost bests carry a third caveat: they are p50s over a run whose
provider load is not controlled, so a 20% latency difference between versions
is not necessarily attributable to the model.
"""
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from evals.cli import _load_env

_load_env()

# Runs measured before `evals/agent_factory.py` rendered `<personalization>`
# through production's own formatter. Their scores are not strictly comparable
# with v5's; see the module docstring.
STALE_HARNESS_LABELS = {"rebase-2026-08-12", "v4-2026-08-13"}

DDL = """\
-- Analysis artifact, not a measurement. Deliberately NOT in eval_runs: nothing
-- that ranks models should be able to reach it. See evals/oracle.py.
CREATE TABLE IF NOT EXISTS eval_oracle (
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    family         TEXT NOT NULL,          -- "rix"
    scope          TEXT NOT NULL,          -- "case" | "suite" | "overall"
    key            TEXT NOT NULL,          -- case_id, suite name, or "OVERALL"
    suite          TEXT,
    -- Best value and the run that produced it, per metric.
    best_score     NUMERIC,
    score_from     TEXT,
    best_pass_rate NUMERIC,
    pass_from      TEXT,
    best_ttft_ms   NUMERIC,
    ttft_from      TEXT,
    best_latency_ms NUMERIC,
    latency_from   TEXT,
    best_cost_usd  NUMERIC,
    cost_from      TEXT,
    -- TRUE when the winning run predates the personalization harness fix.
    stale_harness  BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (family, scope, key)
);
"""


def write_run(family, overall, suite_means, rows_out, newest, supabase) -> None:
    """File the oracle as a run so the dashboard's views can render it.

    Every value written here is a real measurement. For each case the
    best-scoring result rows are copied wholesale from whichever version
    produced them, so latency, tokens and cost stay internally consistent with
    the score beside them rather than being mixed across versions or invented.
    That is a weaker claim than the `eval_oracle` table makes — there each
    metric is independently best-of — and the difference is deliberate: a
    leaderboard row implies one model ran one way, so the only honest thing to
    put in it is one version's actual numbers per case.

    The row is not disguised. `model_label` says oracle, `provider` says
    oracle, `model_family` is NULL so `v_eval_family_grid` skips it, and
    `notes` states outright that no model produced this. Naming is the entire
    safeguard once the row is in a table that ranks things, so it is loud on
    purpose.
    """
    from core.database.supabase_client import utcnow_iso

    label = f"{family}-oracle-upper-bound"
    prior = (
        supabase.table("eval_runs").select("run_id")
        .eq("model_label", label).execute().data or []
    )
    for p in prior:  # eval_results/eval_case_scores cascade on delete
        supabase.table("eval_runs").delete().eq("run_id", p["run_id"]).execute()
    if prior:
        print(f"  replaced {len(prior)} earlier oracle run(s)")

    versions = ", ".join(f"{m}={r['score']}" for m, r in sorted(newest.items()))
    run = supabase.table("eval_runs").insert({
        "label": "ORACLE — synthetic upper bound, NOT a measurement",
        "mode": "pro",
        "model_label": label,
        "model_id": f"oracle:max-over-{family}-versions",
        "provider": "oracle",
        "model_family": None,
        "status": "done",
        "started_at": utcnow_iso(),
        "finished_at": utcnow_iso(),
        "score": overall,
        "suite_scores": suite_means,
        "n_cases": len([r for r in rows_out if r["scope"] == "case"]),
        "notes": (
            "SYNTHETIC. No model produced this run. For each case it carries the "
            "best result any rix version achieved, so the aggregate describes a "
            "model that does not exist. Read it as headroom, not as a score. "
            f"Composed from: {versions}. Rows whose best comes from v1/v3/v4 were "
            "measured under the pre-fix <personalization> spelling and are not "
            "strictly comparable with v5's. Generated by evals/oracle.py."
        ),
    }).execute().data[0]["run_id"]

    src_runs = {r["run_id"]: r["model_label"] for r in newest.values()}
    results = (
        supabase.table("eval_results").select("*")
        .in_("run_id", list(src_runs)).execute().data or []
    )
    best_of: dict[str, dict] = {}
    for r in results:
        if r["status"] != "ok" or r.get("score") is None:
            continue
        cur = best_of.get(r["case_id"])
        if cur is None or float(r["score"]) > float(cur["score"]):
            best_of[r["case_id"]] = r

    copied = []
    for cid, r in best_of.items():
        row = {k: v for k, v in r.items() if k not in ("result_id", "created_at")}
        row["run_id"] = run
        copied.append(row)
    for i in range(0, len(copied), 20):
        supabase.table("eval_results").insert(copied[i:i + 20]).execute()

    cs = (
        supabase.table("eval_case_scores").select("*")
        .in_("run_id", list(src_runs)).execute().data or []
    )
    best_cs: dict[str, dict] = {}
    for r in cs:
        cur = best_cs.get(r["case_id"])
        if cur is None or float(r["score_mean"]) > float(cur["score_mean"]):
            best_cs[r["case_id"]] = r
    cs_rows = []
    for cid, r in best_cs.items():
        row = dict(r)
        row["run_id"] = run
        row["model_label"] = label
        cs_rows.append(row)
    supabase.table("eval_case_scores").upsert(cs_rows, on_conflict="run_id,case_id").execute()

    try:
        from core.database.db_evals import bump_cache_epoch

        bump_cache_epoch()
    except Exception:
        pass
    print(f"\nfiled as eval_runs row {run[:8]} — model_label={label!r}")
    print(f"  {len(copied)} result rows + {len(cs_rows)} case scores copied from real runs")


def _best(rows, field, *, lower_is_better=False):
    """(value, model_label, label) of the best row for `field`, or (None,)*3."""
    have = [r for r in rows if r.get(field) is not None]
    if not have:
        return None, None, None
    pick = (min if lower_is_better else max)(have, key=lambda r: float(r[field]))
    return float(pick[field]), pick["model_label"], pick["_label"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="rix")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--ddl", action="store_true")
    # The table needs DDL that PostgREST cannot issue, so `--write` fails until
    # someone applies it by hand. This keeps the computed rows rather than
    # making the next person re-derive them.
    ap.add_argument("--out", default=None, help="also dump the rows to this JSON file")
    ap.add_argument("--write-run", action="store_true",
                    help="also file it as an eval_runs row so the dashboard can show it")
    args = ap.parse_args()

    if args.ddl:
        print(DDL)
        return 0

    from core.database.supabase_client import supabase

    runs = (
        supabase.table("eval_runs")
        .select("run_id,model_label,label,started_at,score,status")
        .eq("status", "done").order("started_at", desc=True).execute().data or []
    )
    newest: dict[str, dict] = {}
    for r in runs:
        if r["model_label"].startswith(args.family):
            newest.setdefault(r["model_label"], r)
    if not newest:
        raise SystemExit(f"no finished runs for family {args.family!r}")

    by_run = {r["run_id"]: r for r in newest.values()}
    print(f"versions in scope ({len(newest)}):")
    for m, r in sorted(newest.items()):
        stale = " [stale harness]" if r["label"] in STALE_HARNESS_LABELS else ""
        print(f"  {m:<24} score={r['score']}  label={r['label']}{stale}")

    scores = (
        supabase.table("eval_case_scores")
        .select("run_id,case_id,suite,model_label,score_mean,pass_rate,"
                "ttft_ms_p50,latency_ms_p50,cost_usd_mean")
        .in_("run_id", list(by_run)).execute().data or []
    )
    per_case: dict[str, list] = defaultdict(list)
    for s in scores:
        s["_label"] = by_run[s["run_id"]]["label"]
        per_case[s["case_id"]].append(s)

    rows_out, suite_of, case_best = [], {}, {}
    for cid, rows in sorted(per_case.items()):
        suite_of[cid] = rows[0]["suite"]
        sc, sc_m, sc_l = _best(rows, "score_mean")
        pr, pr_m, _ = _best(rows, "pass_rate")
        tt, tt_m, _ = _best(rows, "ttft_ms_p50", lower_is_better=True)
        lat, lat_m, _ = _best(rows, "latency_ms_p50", lower_is_better=True)
        cost, cost_m, _ = _best(rows, "cost_usd_mean", lower_is_better=True)
        case_best[cid] = sc
        rows_out.append({
            "family": args.family, "scope": "case", "key": cid, "suite": rows[0]["suite"],
            "best_score": sc, "score_from": sc_m,
            "best_pass_rate": pr, "pass_from": pr_m,
            "best_ttft_ms": tt, "ttft_from": tt_m,
            "best_latency_ms": lat, "latency_from": lat_m,
            "best_cost_usd": cost, "cost_from": cost_m,
            "stale_harness": sc_l in STALE_HARNESS_LABELS,
        })

    # Aggregate exactly as scoring.summarize does: suite means, then the mean
    # of those. Using per-case maxima rather than per-suite maxima — taking the
    # max of already-averaged suites would double-count the selection.
    by_suite: dict[str, list[float]] = defaultdict(list)
    for cid, v in case_best.items():
        if v is not None:
            by_suite[suite_of[cid]].append(v)
    suite_means = {s: round(statistics.fmean(v), 4) for s, v in by_suite.items()}
    overall = round(statistics.fmean(list(suite_means.values())), 4)

    for s, v in sorted(suite_means.items()):
        contributors = {r["score_from"] for r in rows_out if r["suite"] == s}
        stale = any(r["stale_harness"] for r in rows_out if r["suite"] == s)
        rows_out.append({
            "family": args.family, "scope": "suite", "key": s, "suite": s,
            "best_score": v, "score_from": "+".join(sorted(contributors)),
            "stale_harness": stale,
        })
    rows_out.append({
        "family": args.family, "scope": "overall", "key": "OVERALL", "suite": None,
        "best_score": overall, "score_from": "+".join(sorted(newest)),
        "stale_harness": any(r.get("stale_harness") for r in rows_out),
    })

    best_real = max(newest.values(), key=lambda r: r["score"] or 0)
    print(f"\n{'suite':<18}{'oracle':>9}{'best real':>11}{'headroom':>10}   贡献版本")
    real_suites = (
        supabase.table("eval_runs").select("suite_scores")
        .eq("run_id", best_real["run_id"]).execute().data[0]["suite_scores"] or {}
    )
    for s, v in sorted(suite_means.items()):
        rv = real_suites.get(s)
        who = "+".join(sorted({r["score_from"].replace("rix-30b-a3b-", "")
                               for r in rows_out if r["scope"] == "case" and r["suite"] == s}))
        gap = f"{v - rv:+.4f}" if rv is not None else "  -"
        print(f"{s:<18}{v:>9.4f}{(rv if rv is not None else 0):>11.4f}{gap:>10}   {who}")
    print(f"{'OVERALL':<18}{overall:>9.4f}{best_real['score']:>11.4f}"
          f"{overall - best_real['score']:>+10.4f}   vs {best_real['model_label']}")

    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"family": args.family, "overall": overall,
                       "suite_means": suite_means,
                       "versions": {m: r["score"] for m, r in newest.items()},
                       "rows": rows_out}, f, ensure_ascii=False, indent=2)
        print(f"\ndumped {len(rows_out)} rows -> {args.out}")

    if args.write_run:
        write_run(args.family, overall, suite_means, rows_out, newest, supabase)

    if not args.write:
        print("(dry run — pass --write to upsert into eval_oracle)")
        return 0

    try:
        supabase.table("eval_oracle").upsert(
            rows_out, on_conflict="family,scope,key"
        ).execute()
    except Exception as e:
        print(f"\nwrite failed: {type(e).__name__}: {e}")
        print("If the table does not exist yet, apply this in the Supabase SQL editor:\n")
        print(DDL)
        return 1
    print(f"\nwrote {len(rows_out)} rows -> eval_oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
