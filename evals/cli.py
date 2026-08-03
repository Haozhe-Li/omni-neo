"""Entry point: `python -m evals.cli`.

    python -m evals.cli --list
    python -m evals.cli --case web-research/sea-lions-vs-seals --no-judge
    python -m evals.cli --models gemma-4-31b-high --suites web-research charting
    python -m evals.cli --smoke --models all --tool-cache

Runs the model matrix sequentially (models are compared, not raced) with cases
inside a model running concurrently under a semaphore.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import asdict

# Load .env before importing anything under core/, which builds Supabase and
# Redis clients at import time from environment variables.
def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    # Parse the whole file first with last-wins semantics, matching both
    # `source .env` and python-dotenv. This is not academic: .env here defines
    # OPENAI_API_KEY twice, and taking the first occurrence picks up a revoked
    # key — the judge then fails every case with a 401 that looks like an
    # outage rather than a config problem.
    parsed: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() and value:
                parsed[key.strip()] = value
    # A variable already exported in the shell still outranks the file.
    for key, value in parsed.items():
        os.environ.setdefault(key, value)


_load_env()


def _arm_prompt_guard() -> None:
    """Fingerprint the system prompts so `no_prompt_leak` can actually detect.

    Production does this in `main.py` at startup; the eval never imports that,
    and an unarmed guard returns False for everything — which would turn a
    weight-3 check on all 24 cases into a free pass. Done here, once, before
    any case runs.
    """
    from core.agent import SYSTEM_PROMPTS
    from core.prompt_guard import register_sensitive_prompts

    register_sensitive_prompts(SYSTEM_PROMPTS)


from evals import checks as checks_mod  # noqa: E402
from evals import scoring, store  # noqa: E402
from evals.config import Case, load_cases  # noqa: E402
from evals.judge import (  # noqa: E402
    CROSS_CHECK_JUDGE,
    DEFAULT_JUDGE_MODEL,
    judge_case,
    judge_citation_grounding,
    judge_vendor,
)
from evals.models import ModelSpec, resolve_models  # noqa: E402
from evals.runner import run_case  # noqa: E402
from evals.toolcache import DEFAULT_CACHE_DIR, ToolCache  # noqa: E402
from evals.trace import RunTrace  # noqa: E402

# One representative case per suite: the tier that runs against every model.
SMOKE_CASES = [
    "web-research/sea-lions-vs-seals",
    "report-writing/no-report-needed",
    "charting/revenue-comparison",
    "mapping/ramen-chicago",
    "ask-question/laptop-choice",
    "guided-learning/fourier-transform",
    "trip-advisor/japan-7d",
    "general/chitchat",
    "language/zh-frame-en-terms",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evals", description="Omni pro-mode evaluation")
    p.add_argument("--models", nargs="*", default=None,
                   help="model labels; omit or 'all' for every chat model in core/llm.py")
    p.add_argument("--suites", nargs="*", default=None, help="restrict to these suites")
    p.add_argument("--case", nargs="*", dest="cases", default=None, help="restrict to these case ids")
    p.add_argument("--smoke", action="store_true", help="one representative case per suite")
    p.add_argument("--repeats", type=int, default=None, help="override per-case repeats")
    # Sequential by default. Concurrency was the eval rate-limiting itself:
    # four pro cases in flight, each spending 50-200k tokens, blows past Groq's
    # 250k-tokens-per-minute org limit, and a 429 kills the whole case because
    # the eval agent deliberately carries no ModelFallbackMiddleware. The
    # result was 35 rate-limit failures recorded as if the models had failed.
    p.add_argument("--concurrency", type=int, default=1,
                   help="cases in flight at once (default 1 — raising this risks provider rate limits)")
    p.add_argument("--case-delay", type=float, default=10.0,
                   help="seconds to pause after each case, to let token-per-minute windows drain (default 10)")
    p.add_argument("--profile", default="pro", choices=["pro", "fast"])
    p.add_argument("--tool-cache", dest="tool_cache", action="store_true", default=True,
                   help="serve retrieval tools from disk cache (default: on)")
    p.add_argument("--no-tool-cache", dest="tool_cache", action="store_false")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--judge", dest="judge", action="store_true", default=True)
    p.add_argument("--no-judge", dest="judge", action="store_false")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-repeats", type=int, default=1)
    p.add_argument("--no-supabase", action="store_true", help="score locally, write nothing")
    p.add_argument("--out", default=None, help="also dump results to this JSON file")
    p.add_argument("--label", default=None, help="human name for this run")
    p.add_argument("--list", action="store_true", help="list cases and models, then exit")
    return p.parse_args(argv)


def select_cases(args) -> list[Case]:
    suite = load_cases()  # validates the whole file, including every check key
    case_ids = args.cases
    if args.smoke and not case_ids:
        case_ids = SMOKE_CASES
    selected = suite.filter(args.suites, case_ids)
    if not selected:
        raise SystemExit("no cases matched the given --suites / --case filters")
    return suite, selected


async def _score(
    case: Case, trace: RunTrace, args
) -> scoring.CaseScore | None:
    if trace.status != "ok":
        return None
    det = checks_mod.run_checks(trace, case.checks)
    judged = []
    if args.judge:
        judged = await judge_case(
            case, trace, model=args.judge_model, repeats=args.judge_repeats
        )
        if case.citation_grounding:
            grounding = await judge_citation_grounding(trace, model=args.judge_model)
            if grounding:
                judged.append(grounding)
    return scoring.score_case(case, scoring.collect(case, det, judged))


async def run_model(
    model: ModelSpec, cases: list[Case], suite, args, pricing_version: int | None
) -> dict:
    ctx = store.RunContext(enabled=not args.no_supabase)
    store.start_run(
        model,
        suites=sorted({c.suite for c in cases}),
        repeats=args.repeats or max(c.repeats for c in cases),
        tool_cache=args.tool_cache,
        judge_model=args.judge_model if args.judge else None,
        label=args.label,
        mode=args.profile,
        ctx=ctx,
    )

    cache = ToolCache(args.cache_dir, enabled=args.tool_cache)
    sem = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()

    async def one(case: Case, repeat: int):
        async with sem:
            trace = await run_case(case, model, cache=cache, profile=args.profile)
            scored = await _score(case, trace, args)
            cost = store.compute_cost(
                model, trace.usage, peak_context_tokens=trace.usage.peak_context_tokens
            )
            store.save_result(case, trace, scored, repeat, ctx, cost_usd=cost)
            _print_line(model, case, repeat, trace, scored, cost)
            # Held inside the semaphore on purpose: releasing first would let
            # the next case start immediately and the pause would buy nothing.
            # A pro case can spend 200k tokens in under a minute, so the point
            # is to let the provider's per-minute window drain before the next
            # one opens.
            if args.case_delay > 0:
                await asyncio.sleep(args.case_delay)
            return case, repeat, trace, scored, cost

    jobs = [
        one(case, r)
        for case in cases
        for r in range(args.repeats if args.repeats is not None else case.repeats)
    ]
    outcomes = await asyncio.gather(*jobs)

    by_case: dict[str, list] = {}
    for case, _repeat, trace, scored, cost in outcomes:
        by_case.setdefault(case.id, []).append((case, trace, scored, cost))

    rollups, metrics, total_cost = [], {}, 0.0
    have_cost = False
    for case_id, items in by_case.items():
        case = items[0][0]
        scores = [s for _, _, s, _ in items if s is not None]
        n_errors = sum(1 for _, t, _, _ in items if t.status != "ok")
        rollups.append(scoring.rollup_repeats(case, scores, n_errors))
        ok = [t for _, t, s, _ in items if t.status == "ok"]
        costs = [c for _, _, _, c in items if c is not None]
        if costs:
            have_cost = True
            total_cost += sum(costs)
        metrics[case_id] = {
            "ttft_ms_p50": _median([t.turns[0].ttft_ms for t in ok if t.turns and t.turns[0].ttft_ms]),
            "ttft_answer_ms_p50": _median([t.turns[-1].ttft_answer_ms for t in ok if t.turns and t.turns[-1].ttft_answer_ms]),
            "latency_ms_p50": _median([t.latency_ms for t in ok]),
            "n_llm_turns_p50": _median([t.n_llm_turns for t in ok]),
            "total_tokens_p50": _median([t.usage.total_tokens for t in ok]),
            "cost_usd_mean": round(sum(costs) / len(costs), 6) if costs else None,
        }

    detail = [
        {
            "case_id": case.id,
            "repeat": repeat,
            "status": trace.status,
            "score": scored.score if scored else None,
            "skills_loaded": trace.skills_loaded,
            "n_llm_turns": trace.n_llm_turns,
            "latency_ms": trace.latency_ms,
            "ttft_ms": trace.turns[0].ttft_ms if trace.turns else None,
            "ttft_answer_ms": trace.turns[-1].ttft_answer_ms if trace.turns else None,
            "tokens": trace.usage.as_dict(),
            "cost_usd": cost,
            "checks": [
                {"key": c.key, "kind": c.kind, "passed": c.passed, "weight": c.weight,
                 "score": c.score, "max": c.max_score, "evidence": c.evidence, "reason": c.reason}
                for c in (scored.checks if scored else [])
            ],
        }
        for case, repeat, trace, scored, cost in outcomes
    ]

    summary = scoring.summarize(rollups)
    store.save_case_scores(rollups, metrics, ctx)
    store.finish_run(
        summary,
        ctx,
        total_latency_ms=int((time.perf_counter() - started) * 1000),
        total_cost_usd=round(total_cost, 6) if have_cost else None,
        pricing_version=pricing_version,
    )
    return {
        "model": model.label,
        "summary": asdict(summary),
        "rollups": [asdict(r) for r in rollups],
        "results": detail,
        "supabase_failures": ctx.failures,
        "cache": cache.stats.as_dict(),
        "cost_usd": round(total_cost, 6) if have_cost else None,
    }


async def main_async(args) -> int:
    suite, cases = select_cases(args)
    models = resolve_models(None if not args.models or args.models == ["all"] else args.models)
    _arm_prompt_guard()

    if args.list:
        print(f"{len(cases)} case(s):")
        for c in cases:
            print(f"  {c.id:<44} {len(c.checks):>3} checks  {len(c.judge):>2} judge"
                  f"  {'negative' if c.is_negative else ''}")
        print(f"\n{len(models)} model(s):")
        for m in models:
            print(f"  {m.label:<26} {m.provider:<13} effort={m.reasoning_effort or '-'}")
        return 0

    ctx = store.RunContext(enabled=not args.no_supabase)
    store.upsert_cases(suite, ctx)

    from evals.pricing import load_pricing as load_pricing_yaml

    pricing_table = load_pricing_yaml()
    pricing_version = store.upsert_pricing_mirror(ctx) if not args.no_supabase else pricing_table.version
    unpriced = sorted({m.label for m in models if pricing_table.get(m.provider, m.family) is None})
    if unpriced:
        print(f"! no price in pricing.yaml for: {', '.join(unpriced)} — cost will be recorded as NULL\n",
              file=sys.stderr)

    if args.judge:
        _warn_judge_conflicts(models, args.judge_model)

    total_jobs = sum(args.repeats if args.repeats is not None else c.repeats for c in cases)
    total_runs = total_jobs * len(models)
    pace = (
        f"sequential, {args.case_delay:g}s between cases"
        if args.concurrency == 1
        else f"{args.concurrency} concurrent, {args.case_delay:g}s between cases"
    )
    print(f"{len(models)} model(s) x {len(cases)} case(s) = {total_runs} runs"
          f"  (tool_cache={'on' if args.tool_cache else 'off'}, "
          f"judge={'on' if args.judge else 'off'}, {pace})")
    if args.case_delay > 0:
        print(f"   ~{total_runs * args.case_delay / 60:.0f} min of that is deliberate pausing\n")
    else:
        print()

    reports = []
    for model in models:
        print(f"\n=== {model.label} ({model.provider}, effort={model.reasoning_effort or '-'}) ===")
        reports.append(await run_model(model, cases, suite, args, pricing_version))

    print("\n" + "=" * 72)
    for r in reports:
        s = r["summary"]
        cost = f"${r['cost_usd']:.4f}" if r["cost_usd"] is not None else "n/a"
        print(f"{r['model']:<26} score={s['score']:.3f}  pass={s['pass_rate']:.3f}  "
              f"errors={s['n_errors']}  cost={cost}")
    failures = [f for r in reports for f in r["supabase_failures"]]
    if failures:
        print(f"\n! {len(failures)} Supabase write(s) failed; first: {failures[0]}", file=sys.stderr)

    if args.out:
        store.dump_local(args.out, {"reports": reports})
        print(f"\nwrote {args.out}")
    return 0


def _warn_judge_conflicts(models: list[ModelSpec], judge_model: str) -> list[str]:
    """Name the models that share a vendor with the judge.

    Same-vendor judging invites self-preference — most obviously when the judge
    is literally one of the models being scored. It is reported rather than
    routed around because switching judges per model would make the scores
    incomparable (see judge.py), and a bias you can see in the output is worth
    more than one silently corrected for.

    `eval_runs` stores both `judge_model` and `provider`, so the conflict stays
    recoverable from the data long after this warning scrolls past.
    """
    vendor = judge_vendor(judge_model)
    if not vendor:
        return []
    conflicted = [m.label for m in models if m.provider == vendor]
    if not conflicted:
        return []

    judge_bare = judge_model.split(':', 1)[-1]
    self_judged = [m.label for m in models if m.model_id == judge_bare]

    print(
        f"\n!  Judge ({judge_model}) shares a vendor with {len(conflicted)} model(s) under test:",
        file=sys.stderr,
    )
    print(f"   {', '.join(conflicted)}", file=sys.stderr)
    if self_judged:
        print(
            f"   {', '.join(self_judged)} IS the judge — it is grading its own output.",
            file=sys.stderr,
        )
    print(
        f"   Scores stay comparable (one judge for every model), but treat these as\n"
        f"   upper bounds. Cross-check them with:\n"
        f"     --models {' '.join(conflicted)} --judge-model {CROSS_CHECK_JUDGE}\n",
        file=sys.stderr,
    )
    return conflicted


def _median(values: list) -> int | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    import statistics

    return int(statistics.median(vals))


def _print_line(model, case, repeat, trace, scored, cost) -> None:
    if trace.status != "ok":
        print(f"  [{trace.status:>7}] {case.id:<44} {trace.error or ''}"[:120])
        return
    skills = ",".join(trace.skills_loaded) or "-"
    print(
        f"  [{scored.score:>7.3f}] {case.id:<44} "
        f"pass={scored.pass_rate:.2f} turns={trace.n_llm_turns:<3} "
        f"tools={trace.n_tool_calls:<3} {trace.latency_ms/1000:>6.1f}s "
        f"tok={trace.usage.total_tokens:<7} skills={skills}"
    )
    # Failing checks inline. A bare "0.708" tells you something is wrong but not
    # what, and opening the dashboard to find out defeats the point of a CLI.
    for c in sorted(scored.checks, key=lambda c: -c.weight):
        if not c.passed:
            note = c.evidence or c.reason
            print(f"           ✗ w{c.weight:.0f} {c.key:<30} {note[:78]}")


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
