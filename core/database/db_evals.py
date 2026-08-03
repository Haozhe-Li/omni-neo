"""Read-only access to the evaluation tables (see evals/schema_evals.sql).

Write access deliberately lives nowhere near the API: eval rows are produced
only by `python -m evals.cli`, never by a request, so this module exposes
selects and nothing else. Even if the dashboard is compromised there is no
handler here that can mutate a result.

All access is over PostgREST via the async client, since every caller is an
async FastAPI handler on the event loop (the sync client would block it for a
full round trip — see supabase_client.py).

## Column projection

`eval_results` rows are big: `trace`, `final_texts` and `report_md` together
run to hundreds of kilobytes for a deep-research case. Every list query here
names its columns explicitly and omits those three; only the single-result
detail endpoint pulls them. A `select("*")` on a run's result list would ship
tens of megabytes to render a table of scores.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from upstash_redis.asyncio import Redis as AsyncRedis

from core.database.supabase_client import get_async_supabase
from core.utils.redis_cache import l1cache

logger = logging.getLogger(__name__)

# ── caching ─────────────────────────────────────────────────────────────────
# Eval rows are written by a CLI a few times a day and read by a dashboard on
# every page view, so almost every request would otherwise re-run the same
# PostgREST queries against unchanged data. A week is safe as a TTL because the
# rows are append-only: a finished run's results and checks never change.
#
# What a long TTL cannot handle on its own is a *new* run landing — the
# leaderboard and matrix would keep serving last week's model list. Hence the
# epoch below: it is part of every cache key, and `bump_cache_epoch()` (called
# by the eval CLI when a run finishes) invalidates the whole namespace in one
# INCR. Without it a 7-day TTL would mean a fresh benchmark is invisible for
# 7 days, which is worse than not caching at all.
CACHE_TTL = 3600 * 24 * 7

_EPOCH_KEY = "evals:cache_epoch"
# The epoch itself is memoised in-process so the Redis GET happens at most
# every 30s rather than on every request — otherwise the invalidation
# mechanism would add back a round trip per request to save several.
_EPOCH_MEMO_SECONDS = 30
_epoch_value: int = 0
_epoch_read_at: float = 0.0

_async_redis: AsyncRedis | None = None


def _redis() -> AsyncRedis:
    global _async_redis
    if _async_redis is None:
        _async_redis = AsyncRedis.from_env()
    return _async_redis


async def cache_epoch() -> int:
    """Current cache generation. Bumped whenever a run finishes."""
    global _epoch_value, _epoch_read_at
    now = time.monotonic()
    if now - _epoch_read_at < _EPOCH_MEMO_SECONDS:
        return _epoch_value
    try:
        raw = await _redis().get(_EPOCH_KEY)
        _epoch_value = int(raw) if raw else 0
    except Exception as e:
        # A Redis hiccup must not take the dashboard down: fall through on the
        # last known epoch, which at worst serves slightly stale rows.
        logger.warning("[db_evals] epoch read failed, reusing %s: %s", _epoch_value, e)
    _epoch_read_at = now
    return _epoch_value


def bump_cache_epoch() -> None:
    """Invalidate every cached eval query. Called after a run is written.

    Synchronous and best-effort: it runs from the eval CLI, not a request
    handler, and a failure here only means the dashboard serves stale data
    until the TTL expires.
    """
    try:
        from core.utils.redis_cache import r as sync_redis

        sync_redis.incr(_EPOCH_KEY)
    except Exception as e:
        logger.warning("[db_evals] could not bump cache epoch: %s", e)


# How often a viewer-triggered refresh may actually invalidate the cache.
# The dashboard is public, so its refresh button is reachable by anyone; without
# a cooldown, holding it down would turn one click into an unbounded stream of
# uncached Supabase queries. 30s is short enough that a human waiting on a run
# they just finished still feels the button work.
_REFRESH_COOLDOWN_SECONDS = 30
_REFRESH_LOCK_KEY = "evals:refresh_lock"


async def request_refresh() -> dict[str, Any]:
    """Force-invalidate the cache on a viewer's request, at most once per
    cooldown window.

    Returns `{"refreshed": bool, "epoch": int, "retry_after": int}` —
    `refreshed=False` means someone else refreshed moments ago and the caller
    already has that data, which is a success from the user's point of view,
    not an error.
    """
    global _epoch_read_at
    redis = _redis()
    try:
        # SET NX EX is the whole rate limiter: exactly one caller per window
        # wins the key, everyone else is told to read the cache that winner
        # just repopulated.
        acquired = await redis.set(_REFRESH_LOCK_KEY, "1", nx=True, ex=_REFRESH_COOLDOWN_SECONDS)
        if not acquired:
            return {
                "refreshed": False,
                "epoch": await cache_epoch(),
                "retry_after": _REFRESH_COOLDOWN_SECONDS,
            }
        epoch = int(await redis.incr(_EPOCH_KEY))
        # Drop the in-process memo too, or this very process would keep serving
        # the old epoch — and therefore the old cache — for up to 30 more
        # seconds after the refresh it just performed.
        _epoch_read_at = 0.0
        return {"refreshed": True, "epoch": epoch, "retry_after": 0}
    except Exception as e:
        logger.warning("[db_evals] refresh failed: %s", e)
        return {"refreshed": False, "epoch": _epoch_value, "retry_after": 0, "error": str(e)[:200]}

# Everything except the heavy payload columns (trace / final_texts / report_md).
_RESULT_LIST_COLUMNS = (
    "result_id,run_id,case_id,repeat_idx,status,error,score,passed_hard,"
    "n_tool_calls,n_searches,n_pages_read,n_charts,n_maps,has_report,has_question,"
    "word_count,skills_loaded,hit_run_limit,ttft_ms,ttft_answer_ms,ttft_report_ms,"
    "latency_ms,per_turn_latency_ms,n_llm_turns,input_tokens,output_tokens,"
    "cached_input_tokens,reasoning_tokens,peak_context_tokens,cost_usd,created_at"
)

_RUN_COLUMNS = (
    "run_id,label,mode,model_label,model_id,provider,reasoning_effort,model_family,"
    "judge_model,git_sha,prompt_sha,skills_sha,repeats,tool_cache,pricing_version,"
    "suites,status,score,pass_rate,suite_scores,n_cases,n_errors,total_latency_ms,"
    "total_cost_usd,notes,started_at,finished_at"
)

MAX_LIMIT = 500


def _clamp(limit: int | None, default: int = 50) -> int:
    if not limit or limit < 1:
        return default
    return min(limit, MAX_LIMIT)


@l1cache(ttl=CACHE_TTL)
async def _list_runs_cached(_epoch: int, *,
    model_label: str | None = None,
    label: str | None = None,
    status: str | None = None,
    since: str | None = None,
    limit: int = 50,
    offset: int = 0,) -> list[dict]:
    """Runs, newest first. One row per (model, batch)."""
    sb = await get_async_supabase()
    q = sb.table("eval_runs").select(_RUN_COLUMNS)
    if model_label:
        q = q.eq("model_label", model_label)
    if label:
        q = q.eq("label", label)
    if status:
        q = q.eq("status", status)
    if since:
        q = q.gte("started_at", since)
    res = await (
        q.order("started_at", desc=True)
        .range(offset, offset + _clamp(limit) - 1)
        .execute()
    )
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _get_run_cached(_epoch: int, run_id: str) -> dict | None:
    sb = await get_async_supabase()
    res = await sb.table("eval_runs").select(_RUN_COLUMNS).eq("run_id", run_id).limit(1).execute()
    return res.data[0] if res.data else None


@l1cache(ttl=CACHE_TTL)
async def _get_case_scores_cached(_epoch: int, run_id: str) -> list[dict]:
    """Per-case rollups across repeats for one run."""
    sb = await get_async_supabase()
    res = await (
        sb.table("eval_case_scores")
        .select("*")
        .eq("run_id", run_id)
        .order("suite")
        .order("case_id")
        .execute()
    )
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _list_results_cached(_epoch: int, run_id: str, *, case_id: str | None = None, status: str | None = None) -> list[dict]:
    """Individual executions for a run, without the heavy payload columns."""
    sb = await get_async_supabase()
    q = sb.table("eval_results").select(_RESULT_LIST_COLUMNS).eq("run_id", run_id)
    if case_id:
        q = q.eq("case_id", case_id)
    if status:
        q = q.eq("status", status)
    res = await q.order("case_id").order("repeat_idx").execute()
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _get_result_cached(_epoch: int, result_id: str, *, include_trace: bool = True) -> dict | None:
    """One execution in full — this is the only place the big columns load."""
    sb = await get_async_supabase()
    columns = "*" if include_trace else _RESULT_LIST_COLUMNS
    res = await sb.table("eval_results").select(columns).eq("result_id", result_id).limit(1).execute()
    return res.data[0] if res.data else None


@l1cache(ttl=CACHE_TTL)
async def _get_checks_cached(_epoch: int, result_id: str) -> list[dict]:
    """The rubric checklist for one result — both layers, deterministic first.

    Ordering puts failures at the top within each layer, since that is what
    anyone opening a result is looking for.
    """
    sb = await get_async_supabase()
    res = await (
        sb.table("eval_checks")
        .select("*")
        .eq("result_id", result_id)
        .order("kind")
        .order("passed")
        .order("weight", desc=True)
        .execute()
    )
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _list_cases_cached(_epoch: int, suite: str | None = None) -> list[dict]:
    """The case registry: prompts and rubric definitions, mirrored from
    cases.yaml on every run so the dashboard never parses the config itself."""
    sb = await get_async_supabase()
    q = sb.table("eval_cases").select("*")
    if suite:
        q = q.eq("suite", suite)
    res = await q.order("suite").order("case_id").execute()
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _leaderboard_cached(_epoch: int, *, label: str | None = None, since: str | None = None) -> list[dict]:
    """Quality next to latency and cost, one row per run."""
    sb = await get_async_supabase()
    q = sb.table("v_eval_model_leaderboard").select("*")
    if label:
        q = q.eq("label", label)
    if since:
        q = q.gte("started_at", since)
    res = await q.order("score", desc=True).execute()
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _family_grid_cached(_epoch: int, family: str | None = None) -> list[dict]:
    """provider x reasoning_effort cells, for the gpt-oss-120b 2x3 and gemma pair."""
    sb = await get_async_supabase()
    q = sb.table("v_eval_family_grid").select("*")
    if family:
        q = q.eq("model_family", family)
    res = await q.order("model_family").order("provider").order("reasoning_effort").execute()
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _check_failures_cached(_epoch: int, *, min_evaluated: int = 1, limit: int = 50) -> list[dict]:
    """Which rubric items fail most often, across every run."""
    sb = await get_async_supabase()
    res = await (
        sb.table("v_eval_check_failures")
        .select("*")
        .gte("n_evaluated", min_evaluated)
        .order("failure_rate", desc=True)
        .limit(_clamp(limit))
        .execute()
    )
    return res.data or []


@l1cache(ttl=CACHE_TTL)
async def _run_summary_cached(_epoch: int, run_id: str) -> list[dict]:
    sb = await get_async_supabase()
    res = await sb.table("v_eval_run_summary").select("*").eq("run_id", run_id).execute()
    return res.data or []


async def resolve_run_ids(
    *, label: str | None = None, run_ids: list[str] | None = None, latest_per_model: bool = False
) -> list[dict]:
    """Pick the runs a comparison should cover.

    `latest_per_model` exists because runs accumulate: a smoke run and a full
    matrix run of the same model both live in `eval_runs` forever, and a matrix
    built over "all runs" would show the same model twice with different case
    coverage. Collapsing to the newest run per model is what a dashboard almost
    always wants, and is done here rather than in SQL so the same helper can
    also serve an explicit run-id selection.
    """
    if run_ids:
        sb = await get_async_supabase()
        res = await sb.table("eval_runs").select(_RUN_COLUMNS).in_("run_id", run_ids).execute()
        return res.data or []

    runs = await list_runs(label=label, status="done", limit=MAX_LIMIT)
    if not latest_per_model:
        return runs
    newest: dict[str, dict] = {}
    for run in runs:  # already newest-first
        newest.setdefault(run["model_label"], run)
    return list(newest.values())


@l1cache(ttl=CACHE_TTL)
async def _case_scores_for_runs(_epoch: int, run_ids: tuple[str, ...]) -> list[dict]:
    sb = await get_async_supabase()
    res = await sb.table("eval_case_scores").select("*").in_("run_id", list(run_ids)).execute()
    return res.data or []


async def case_matrix(
    *, label: str | None = None, run_ids: list[str] | None = None, latest_per_model: bool = True
) -> dict[str, Any]:
    """The headline view: case x model scores, pivoted server-side.

    Two queries regardless of matrix size — the runs, then every case score for
    those runs — instead of one per cell. Pivoting here rather than in the
    frontend keeps the shape stable if the dashboard is ever reimplemented.
    """
    runs = await resolve_run_ids(label=label, run_ids=run_ids, latest_per_model=latest_per_model)
    if not runs:
        return {"models": [], "cases": [], "cells": {}, "runs": []}

    by_run = {r["run_id"]: r for r in runs}
    # Sorted so the cache key is stable: `resolve_run_ids` returns rows in
    # whatever order PostgREST produced, and an unsorted list would mint a new
    # cache entry for the same set of runs on every call.
    rows = await _case_scores_for_runs(await cache_epoch(), tuple(sorted(by_run)))

    cells: dict[str, dict[str, Any]] = {}
    cases: dict[str, str] = {}
    for row in rows:
        case_id, model = row["case_id"], row["model_label"]
        cases[case_id] = row.get("suite") or ""
        cells.setdefault(case_id, {})[model] = {
            "run_id": row["run_id"],
            "score_mean": row.get("score_mean"),
            "score_stdev": row.get("score_stdev"),
            "pass_rate": row.get("pass_rate"),
            "n_errors": row.get("n_errors"),
            "n_repeats": row.get("n_repeats"),
            "latency_ms_p50": row.get("latency_ms_p50"),
            "ttft_ms_p50": row.get("ttft_ms_p50"),
            "cost_usd_mean": row.get("cost_usd_mean"),
        }

    models = sorted({r["model_label"] for r in runs})
    return {
        "models": models,
        "cases": [{"case_id": c, "suite": s} for c, s in sorted(cases.items(), key=lambda kv: (kv[1], kv[0]))],
        "cells": cells,
        "runs": [
            {
                "run_id": r["run_id"],
                "model_label": r["model_label"],
                "provider": r.get("provider"),
                "reasoning_effort": r.get("reasoning_effort"),
                "model_family": r.get("model_family"),
                "label": r.get("label"),
                "started_at": r.get("started_at"),
                "score": r.get("score"),
                "pass_rate": r.get("pass_rate"),
            }
            for r in runs
        ],
    }


# ── public API (epoch-keyed cache in front of every query) ──────────────────

async def list_runs(*, model_label: str | None = None, label: str | None = None, status: str | None = None, since: str | None = None, limit: int = 50, offset: int = 0):
    return await _list_runs_cached(await cache_epoch(), model_label=model_label, label=label, status=status, since=since, limit=limit, offset=offset)

async def get_run(run_id: str):
    return await _get_run_cached(await cache_epoch(), run_id=run_id)

async def get_case_scores(run_id: str):
    return await _get_case_scores_cached(await cache_epoch(), run_id=run_id)

async def list_results(run_id: str, *, case_id: str | None = None, status: str | None = None):
    return await _list_results_cached(await cache_epoch(), run_id=run_id, case_id=case_id, status=status)

async def get_result(result_id: str, *, include_trace: bool = True):
    return await _get_result_cached(await cache_epoch(), result_id=result_id, include_trace=include_trace)

async def get_checks(result_id: str):
    return await _get_checks_cached(await cache_epoch(), result_id=result_id)

async def list_cases(suite: str | None = None):
    return await _list_cases_cached(await cache_epoch(), suite=suite)

async def leaderboard(*, label: str | None = None, since: str | None = None):
    return await _leaderboard_cached(await cache_epoch(), label=label, since=since)

async def family_grid(family: str | None = None):
    return await _family_grid_cached(await cache_epoch(), family=family)

async def check_failures(*, min_evaluated: int = 1, limit: int = 50):
    return await _check_failures_cached(await cache_epoch(), min_evaluated=min_evaluated, limit=limit)

async def run_summary(run_id: str):
    return await _run_summary_cached(await cache_epoch(), run_id=run_id)
