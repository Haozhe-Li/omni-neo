"""Persist eval results to Supabase over PostgREST.

Reuses the app's existing sync client (`core/database/supabase_client.py`),
which needs a service_role key — these tables are written by a CLI, never by a
request handler, so there is no user JWT to scope them with.

Every write is best-effort in the sense that a Supabase outage must not destroy
a run that already cost real money in model calls: failures are reported and
the run continues, and `--out` gives a local JSON fallback of the same data.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from core.database.supabase_client import supabase, utcnow_iso
from evals.config import Case, Suite
from evals.models import ModelSpec
from evals.scoring import CaseScore, RepeatRollup, RunSummary, ScoredCheck
from evals.trace import RunTrace

log = logging.getLogger("evals.store")

# Trace rows are capped before they hit PostgREST: a 30-step pro run with full
# page bodies is megabytes, and the dashboard only ever renders the heads.
_MAX_TRACE_STEPS = 60


@dataclass
class RunContext:
    run_id: str | None = None
    model: ModelSpec | None = None
    enabled: bool = True
    failures: list[str] = field(default_factory=list)

    def note(self, what: str, err: Exception) -> None:
        msg = f"{what}: {type(err).__name__}: {err}"
        self.failures.append(msg)
        log.warning("supabase write failed — %s", msg)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def prompt_fingerprints() -> tuple[str, str]:
    """sha256 of the system prompt and of every SKILL.md, so a score can be
    attributed to the exact prompt that produced it. A prompt edit that moves
    the numbers is otherwise indistinguishable from a model regression."""
    import hashlib

    from core.agent import SKILL_FILES, SYSTEM_PROMPT

    prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    blob = "".join(
        f"{path}:{json.dumps(data, sort_keys=True, default=str)}"
        for path, data in sorted(SKILL_FILES.items())
    )
    return prompt_sha, hashlib.sha256(blob.encode()).hexdigest()


def upsert_cases(suite: Suite, ctx: RunContext) -> None:
    """Mirror cases.yaml into `eval_cases` so the dashboard can render rubrics
    without importing Python or parsing YAML."""
    if not ctx.enabled:
        return
    rows = [
        {
            "case_id": c.id,
            "suite": c.suite,
            "skill": c.skill,
            "title": c.title,
            "lang": c.lang,
            "turns": [t.text for t in c.turns],
            "rubric": c.as_rubric_json(),
            "rubric_version": c.rubric_version,
            "is_negative": c.is_negative,
            "weight": c.weight,
            "enabled": True,
            "updated_at": utcnow_iso(),
        }
        for c in suite.cases
    ]
    try:
        supabase.table("eval_cases").upsert(rows, on_conflict="case_id").execute()
    except Exception as e:
        ctx.note("upsert eval_cases", e)


def start_run(
    model: ModelSpec,
    *,
    suites: list[str],
    repeats: int,
    tool_cache: bool,
    judge_model: str | None,
    label: str | None,
    mode: str,
    ctx: RunContext,
    attach_run_id: str | None = None,
) -> None:
    """Open a run, or attach to one that already exists.

    `attach_run_id` is for `evals.backfill_cases`: when cases are added to the
    suite, the new ones are measured and filed into the *original* run so its
    score covers the whole rubric instead of being split across two rows that
    no view knows how to combine. Attaching skips the insert entirely — none of
    the columns below are re-derived, so the run keeps the git sha, prompt sha
    and started_at of the measurement it belongs to rather than silently
    acquiring today's.

    That also means the attached rows were produced by a *different* checkout
    than `prompt_sha` claims. It is the caller's job to say so; the backfill
    script writes it into `eval_runs.notes`.
    """
    ctx.model = model
    if not ctx.enabled:
        return
    if attach_run_id:
        ctx.run_id = attach_run_id
        return
    prompt_sha, skills_sha = prompt_fingerprints()
    row = {
        "label": label,
        "mode": mode,
        "model_label": model.label,
        "model_id": model.model_id,
        # Split out rather than left for the dashboard to parse back out of the
        # label: gpt-oss-120b is a fully crossed provider x effort grid and
        # pivoting it is the point of running all six variants.
        #
        # `display_provider`, not `provider` — our own fine-tunes are hosted by
        # W&B but belong to us, so they show as `omni`. Pricing still keys off
        # `provider` (see compute_cost); only the column the dashboard renders
        # changes.
        "provider": model.display_provider,
        "reasoning_effort": model.reasoning_effort,
        "model_family": model.family,
        "judge_model": judge_model,
        "git_sha": git_sha(),
        "prompt_sha": prompt_sha,
        "skills_sha": skills_sha,
        "repeats": repeats,
        "tool_cache": tool_cache,
        "suites": suites,
        "status": "running",
        "started_at": utcnow_iso(),
    }
    try:
        res = supabase.table("eval_runs").insert(row).execute()
        ctx.run_id = res.data[0]["run_id"]
    except Exception as e:
        ctx.note("insert eval_runs", e)
        ctx.enabled = False


def save_result(
    case: Case,
    trace: RunTrace,
    scored: CaseScore | None,
    repeat_idx: int,
    ctx: RunContext,
    *,
    cost_usd: float | None,
) -> None:
    if not ctx.enabled or not ctx.run_id:
        return
    from evals import parsers

    last_text = trace.turns[-1].text if trace.turns else ""
    reports = [r for t in trace.turns for r in parsers.extract_reports(t.text)]
    report = max(reports, key=lambda r: r.words) if reports else None
    usage = trace.usage

    row = {
        "run_id": ctx.run_id,
        "case_id": case.id,
        "repeat_idx": repeat_idx,
        "status": trace.status,
        "error": trace.error,
        "score": scored.score if scored else None,
        "passed_hard": scored.passed_hard if scored else None,
        "final_texts": [t.text for t in trace.turns],
        "report_md": report.body if report else None,
        "report_title": report.title if report else None,
        "n_tool_calls": trace.n_tool_calls,
        "n_searches": sum(t.n_searches for t in trace.turns),
        "n_pages_read": sum(t.n_pages_read for t in trace.turns),
        "n_charts": sum(len(parsers.extract_charts(t.text)) for t in trace.turns),
        "n_maps": sum(len(parsers.extract_maps(t.text)) for t in trace.turns),
        "has_report": bool(report),
        "has_question": any(parsers.extract_question(t.text) for t in trace.turns),
        "word_count": parsers.count_words(parsers.prose_only(last_text)),
        "skills_loaded": trace.skills_loaded,
        "hit_run_limit": trace.hit_run_limit,
        "ttft_ms": trace.turns[0].ttft_ms if trace.turns else None,
        "ttft_answer_ms": trace.turns[-1].ttft_answer_ms if trace.turns else None,
        "ttft_report_ms": next(
            (t.ttft_report_ms for t in trace.turns if t.ttft_report_ms is not None), None
        ),
        "latency_ms": trace.latency_ms,
        "per_turn_latency_ms": trace.per_turn_latency_ms,
        "n_llm_turns": trace.n_llm_turns,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "peak_context_tokens": usage.peak_context_tokens,
        "cost_usd": cost_usd,
        "trace": trace.as_trace_json()[:_MAX_TRACE_STEPS],
    }
    try:
        res = supabase.table("eval_results").insert(row).execute()
        result_id = res.data[0]["result_id"]
    except Exception as e:
        ctx.note(f"insert eval_results ({case.id})", e)
        return

    if scored:
        _save_checks(result_id, case, scored.checks, ctx)


def _save_checks(result_id: str, case: Case, checks: list[ScoredCheck], ctx: RunContext) -> None:
    rows = [
        {
            "result_id": result_id,
            "run_id": ctx.run_id,
            "case_id": case.id,
            "kind": c.kind,
            "key": c.key,
            "label": c.label,
            "turn": c.turn,
            "passed": c.passed,
            "score": c.score,
            "max_score": c.max_score,
            "weight": c.weight,
            "evidence": (c.evidence or "")[:2000],
            "reason": (c.reason or "")[:2000],
            "detail": _jsonable(c.detail),
        }
        for c in checks
    ]
    if not rows:
        return
    try:
        supabase.table("eval_checks").insert(rows).execute()
    except Exception as e:
        ctx.note(f"insert eval_checks ({case.id})", e)


def save_case_scores(rollups: list[RepeatRollup], metrics: dict[str, dict], ctx: RunContext) -> None:
    if not ctx.enabled or not ctx.run_id or not ctx.model:
        return
    rows = []
    for r in rollups:
        m = metrics.get(r.case_id, {})
        rows.append(
            {
                "run_id": ctx.run_id,
                "case_id": r.case_id,
                "suite": r.suite,
                "model_label": ctx.model.label,
                "n_repeats": r.n_repeats,
                "score_mean": r.score_mean,
                "score_min": r.score_min,
                "score_max": r.score_max,
                "score_stdev": r.score_stdev,
                "pass_rate": r.pass_rate,
                "n_errors": r.n_errors,
                **m,
            }
        )
    try:
        supabase.table("eval_case_scores").upsert(rows, on_conflict="run_id,case_id").execute()
    except Exception as e:
        ctx.note("upsert eval_case_scores", e)


def finish_run(
    summary: RunSummary,
    ctx: RunContext,
    *,
    total_latency_ms: int,
    total_cost_usd: float | None,
    pricing_version: int | None,
    notes: str | None = None,
) -> None:
    if not ctx.enabled or not ctx.run_id:
        return
    try:
        supabase.table("eval_runs").update(
            {
                "status": "done",
                "score": summary.score,
                "pass_rate": summary.pass_rate,
                "suite_scores": summary.suite_scores,
                "n_cases": summary.n_cases,
                "n_errors": summary.n_errors,
                "total_latency_ms": total_latency_ms,
                "total_cost_usd": total_cost_usd,
                "pricing_version": pricing_version,
                "notes": notes,
                "finished_at": utcnow_iso(),
            }
        ).eq("run_id", ctx.run_id).execute()
    except Exception as e:
        ctx.note("finish eval_runs", e)
        return

    # The dashboard API caches every query for a week, keyed on this epoch.
    # Without the bump a finished run would stay invisible until the TTL
    # expired, which for a benchmark you just spent two hours producing is the
    # same as not having written it.
    try:
        from core.database.db_evals import bump_cache_epoch

        bump_cache_epoch()
    except Exception as e:
        ctx.note("bump eval cache epoch", e)


# ── pricing ─────────────────────────────────────────────────────────────────
# The oracle is `evals/pricing.yaml`, read by `evals.pricing.compute_cost` —
# not this module. What lives here is purely the DB mirror: `eval_pricing`
# gets re-upserted from the YAML on every CLI run so the numbers are
# inspectable over plain SQL (joins, the dashboard's own queries) without
# anyone needing to open the YAML — but it is a courtesy copy, never consulted
# for the actual dollar figure written to `eval_results.cost_usd`.
#
# The mirror necessarily loses the YAML's short/long-context tiering (the
# table has one input/output/cached triple per model, not two) — every OpenAI
# row here is its short_context price. `evals.pricing.compute_cost` is what
# actually picks a tier per run, using `peak_context_tokens`; this table is
# reference only.
def upsert_pricing_mirror(ctx: RunContext) -> int | None:
    """Push pricing.yaml into `eval_pricing`. Returns the YAML's version int,
    used as `eval_runs.pricing_version` regardless of whether the mirror write
    succeeds — the actual cost calculation never depends on this table.

    Rows are keyed by `ModelSpec.label` (`gpt-oss-120b-low`,
    `gpt-oss-120b-low-groq`, ...) — the table's original `model_label` == every
    other table's `model_label` contract — not by `(provider, family)`.
    Keying by family directly would collide: `gpt-oss-120b` is one family
    priced once in the YAML but is TWO labels here (Cerebras and Groq both
    serve it), and `eval_pricing`'s unique constraint is `(version,
    model_label)` alone, so the upsert would try to write two different
    provider rows under the same key in one batch and fail outright.
    """
    from evals.models import discover_models
    from evals.pricing import load_pricing

    table = load_pricing()
    if not ctx.enabled:
        return table.version

    rows = []
    for model in discover_models():
        price = table.get(model.provider, model.family)
        if price is None:
            continue
        tier = price.tiers.get("short_context") or price.tiers.get("single")
        if tier is None:
            continue
        rows.append(
            {
                "version": table.version,
                "model_label": model.label,
                "provider": model.display_provider,
                "usd_per_1m_input": tier.input,
                "usd_per_1m_output": tier.output,
                "usd_per_1m_cached_input": tier.cached_input,
                "source": "evals/pricing.yaml (short_context tier; see that file for long_context)",
            }
        )
    try:
        supabase.table("eval_pricing").upsert(rows, on_conflict="version,model_label").execute()
    except Exception as e:
        ctx.note("upsert eval_pricing mirror", e)
    return table.version


def compute_cost(model, usage, *, peak_context_tokens: int = 0) -> float | None:
    """USD for one run, or None when the model has no price on file.

    `model` is an `evals.models.ModelSpec` (needs `.provider` / `.family` to
    look up pricing.yaml — a bare label string can't reconstruct those, e.g.
    `gpt-oss-120b-medium-groq` doesn't parse back into provider=groq without
    the table `evals.models` already built).

    None rather than 0.0 on purpose: an unpriced model must read as "unknown"
    in the dashboard, not as "free", or it wins every cost comparison it is in.
    """
    from evals.pricing import compute_cost as _compute_cost

    return _compute_cost(
        model.provider,
        model.family,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        peak_context_tokens=peak_context_tokens,
    )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=str))


def dump_local(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
