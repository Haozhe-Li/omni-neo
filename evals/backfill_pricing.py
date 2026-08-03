"""Recompute `cost_usd` on stored results from `evals/pricing.yaml`.

Run this after editing pricing.yaml (a price changed, a model gained a price
it didn't have before) to bring historical rows up to date without re-running
any model — the whole point of keeping token counts and prices as separate
columns. Also run once after this feature is first added, to backfill every
row that predates it.

    python -m evals.backfill_pricing            # recompute everything
    python -m evals.backfill_pricing --dry-run   # report without writing

Idempotent: safe to run repeatedly, and safe to run against a pricing.yaml
that hasn't changed (every row just gets rewritten to the same number).
"""
from __future__ import annotations

import argparse
import statistics

from evals.cli import _load_env  # env must be loaded before core.* imports build clients

_load_env()

from evals.models import discover_models  # noqa: E402
from evals.pricing import compute_cost, load_pricing  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument("--run-id", default=None, help="restrict to one run")
    args = p.parse_args()

    from core.database.supabase_client import supabase

    by_label = {m.label: m for m in discover_models()}
    table = load_pricing()
    print(f"pricing.yaml version={table.version}, cache_hit_ratio={table.cache_hit_ratio}\n")

    runs_q = supabase.table("eval_runs").select("run_id,model_label")
    if args.run_id:
        runs_q = runs_q.eq("run_id", args.run_id)
    runs = {r["run_id"]: r["model_label"] for r in runs_q.execute().data or []}

    results_q = (
        supabase.table("eval_results")
        .select("result_id,run_id,input_tokens,output_tokens,peak_context_tokens,cost_usd")
        .eq("status", "ok")
    )
    if args.run_id:
        results_q = results_q.eq("run_id", args.run_id)
    results = results_q.execute().data or []

    n_priced = n_unpriced = n_unchanged = n_updated = 0
    per_run_costs: dict[str, list[float]] = {}
    per_case_costs: dict[tuple[str, str], list[float]] = {}
    case_by_result: dict[str, str] = {}

    # A second pass needs case_id per result to rebuild eval_case_scores; fetch
    # it in the same query set rather than a second round trip per row.
    case_ids = {
        r["result_id"]: r["case_id"]
        for r in supabase.table("eval_results").select("result_id,case_id")
        .eq("status", "ok").execute().data or []
    }

    for row in results:
        model = by_label.get(runs.get(row["run_id"], ""))
        if model is None:
            n_unpriced += 1
            continue
        new_cost = compute_cost(
            model.provider,
            model.family,
            input_tokens=row["input_tokens"] or 0,
            output_tokens=row["output_tokens"] or 0,
            peak_context_tokens=row["peak_context_tokens"] or 0,
            table=table,
        )
        if new_cost is None:
            n_unpriced += 1
            continue
        n_priced += 1
        if row.get("cost_usd") == new_cost:
            n_unchanged += 1
        else:
            n_updated += 1
            if not args.dry_run:
                supabase.table("eval_results").update({"cost_usd": new_cost}).eq(
                    "result_id", row["result_id"]
                ).execute()

        per_run_costs.setdefault(row["run_id"], []).append(new_cost)
        case_id = case_ids.get(row["result_id"])
        if case_id:
            per_case_costs.setdefault((row["run_id"], case_id), []).append(new_cost)

    print(f"results: {n_priced} priced ({n_updated} changed, {n_unchanged} already correct), "
          f"{n_unpriced} still unpriced (no matching model or pricing.yaml entry)")

    if not args.dry_run:
        for run_id, costs in per_run_costs.items():
            supabase.table("eval_runs").update(
                {"total_cost_usd": round(sum(costs), 6), "pricing_version": table.version}
            ).eq("run_id", run_id).execute()
        for (run_id, case_id), costs in per_case_costs.items():
            supabase.table("eval_case_scores").update(
                {"cost_usd_mean": round(statistics.fmean(costs), 6)}
            ).eq("run_id", run_id).eq("case_id", case_id).execute()
        print(f"updated total_cost_usd on {len(per_run_costs)} run(s), "
              f"cost_usd_mean on {len(per_case_costs)} case row(s)")

        try:
            from core.database.db_evals import bump_cache_epoch

            bump_cache_epoch()
            print("bumped the dashboard's cache epoch")
        except Exception as e:
            print(f"! could not bump cache epoch: {e}")
    else:
        print("(dry run — nothing written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
