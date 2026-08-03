"""Read-only API over the evaluation results, for the eval dashboard.

Every endpoint is a GET and every query in `core/database/db_evals.py` is a
select — eval rows are written only by `python -m evals.cli`, never by a
request, so there is deliberately no handler here that can create or mutate
one.

## Auth

Public — no sign-in. The benchmark page is meant to be readable by anyone, the
way a published model leaderboard is.

Nothing here is user data. Every row describes a fixed, synthetic test case
from `evals/cases.yaml` being answered by a model: the prompts are authored by
us, the answers are about sea lions and Docker images, and the only identifiers
are model names and content hashes (`prompt_sha` / `skills_sha` expose no
prompt text). No thread, message, upload or account is reachable from any of
these endpoints.

What being public does expose is request volume, since anyone can now call
these. That is handled by the week-long Redis cache in `db_evals` rather than
by an auth wall: a cold query costs several PostgREST round trips, a warm one
costs a single Redis GET, and the data changes a few times a day at most.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from core.database import db_evals

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evals", tags=["evals"])


# ── runs ────────────────────────────────────────────────────────────────────
@router.get("/runs")
async def api_list_runs(
    model_label: str | None = None,
    label: str | None = None,
    status: str | None = None,
    since: str | None = Query(default=None, description="ISO-8601 lower bound on started_at"),
    limit: int = Query(default=50, ge=1, le=db_evals.MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    """Runs, newest first — one row per (model, batch)."""
    return {
        "runs": await db_evals.list_runs(
            model_label=model_label,
            label=label,
            status=status,
            since=since,
            limit=limit,
            offset=offset,
        )
    }


@router.get("/runs/{run_id}")
async def api_get_run(run_id: str):
    """One run with its per-suite rollup and per-case scores.

    Bundled into a single response because the run detail page needs all three
    to render anything at all, and three round trips from the browser to show
    one page is three chances to render half a screen.
    """
    run = await db_evals.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {
        "run": run,
        "suites": await db_evals.run_summary(run_id),
        "case_scores": await db_evals.get_case_scores(run_id),
    }


@router.get("/runs/{run_id}/results")
async def api_list_results(
    run_id: str,
    case_id: str | None = None,
    status: str | None = None,
):
    """Individual executions for a run.

    Omits `trace`, `final_texts` and `report_md` — those run to hundreds of
    kilobytes per row and are only needed on the detail view. Fetch one result
    by id to get them.
    """
    return {"results": await db_evals.list_results(run_id, case_id=case_id, status=status)}


# ── a single execution ──────────────────────────────────────────────────────
@router.get("/results/{result_id}")
async def api_get_result(
    result_id: str,
    include_trace: bool = True,
):
    """One execution in full: the answer, the report, the tool trace, and the
    rubric checklist that graded it."""
    result = await db_evals.get_result(result_id, include_trace=include_trace)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")
    return {"result": result, "checks": await db_evals.get_checks(result_id)}


# ── cross-run views ─────────────────────────────────────────────────────────
@router.get("/leaderboard")
async def api_leaderboard(
    label: str | None = None,
    since: str | None = None,
):
    """Quality beside latency percentiles, tokens and cost, one row per run.

    Filter by `label` or `since` when the table has accumulated smoke runs
    alongside real ones — the underlying view spans every run ever recorded.
    """
    return {"rows": await db_evals.leaderboard(label=label, since=since)}


@router.get("/family-grid")
async def api_family_grid(
    family: str | None = Query(default=None, description='e.g. "gpt-oss-120b"'),
):
    """One row per family x provider x reasoning_effort.

    Reading along a row shows what extra reasoning effort bought; reading down
    a column shows what the provider changed at identical effort — which
    doubles as a sanity check on the harness, since identical weights at
    identical effort should score the same and differ only in latency and cost.
    """
    return {"rows": await db_evals.family_grid(family)}


@router.get("/check-failures")
async def api_check_failures(
    min_evaluated: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=db_evals.MAX_LIMIT),
):
    """Rubric items ranked by failure rate across every run — the "what to fix
    next" list. Raise `min_evaluated` to drop items with too little data."""
    return {"rows": await db_evals.check_failures(min_evaluated=min_evaluated, limit=limit)}


@router.get("/matrix")
async def api_matrix(
    label: str | None = None,
    run_ids: str | None = Query(default=None, description="comma-separated run ids"),
    latest_per_model: bool = True,
):
    """Case x model score matrix, pivoted server-side.

    `latest_per_model` defaults on because runs accumulate — a smoke run and a
    full matrix run of the same model both persist, and showing both would put
    one model in two columns with different case coverage.
    """
    ids = [p.strip() for p in run_ids.split(",") if p.strip()] if run_ids else None
    return await db_evals.case_matrix(
        label=label, run_ids=ids, latest_per_model=latest_per_model
    )


# ── cache control ───────────────────────────────────────────────────────────
@router.post("/refresh")
async def api_refresh():
    """Force the cached queries to be recomputed from Supabase.

    The only non-GET route here, and it still writes no eval data — it bumps a
    cache generation counter, nothing more. It exists because every read is
    cached for a week, so the dashboard's refresh button would otherwise
    re-request rows it has already been given and appear to do nothing.

    Rate-limited server-side to one real invalidation per cooldown window (see
    `db_evals.request_refresh`). A public button that drops a shared cache is
    an amplifier otherwise: one click per viewer becomes a burst of uncached
    queries against Supabase.

    `refreshed: false` is a normal, successful response — it means another
    caller invalidated moments ago and the data is already fresh.
    """
    return await db_evals.request_refresh()


# ── case registry ───────────────────────────────────────────────────────────
@router.get("/cases")
async def api_list_cases(suite: str | None = None):
    """The case registry as authored in `evals/cases.yaml` — prompts, rubric
    definitions and weights — re-upserted on every run so the dashboard can
    render a case's expectations without reading the config file."""
    return {"cases": await db_evals.list_cases(suite)}
