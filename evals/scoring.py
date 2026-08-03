"""Aggregate check results into scores.

Two numbers come out of a run and they answer different questions:

- **score** — weighted mean of every normalised check. A smooth quality signal,
  good for tracking drift and comparing models.
- **pass_rate** — fraction of the *load-bearing* checks (weight >= 2) that
  passed. Blunt, and the one to look at first: a model can hold a respectable
  0.82 while failing the one check that says it never wrote the report.

Suite means, not case means, roll up into the run score — otherwise `general`
(5 cases) would outweigh `trip-advisor` (2) purely by headcount, and the score
would quietly track how many cases each suite happens to have.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from evals.checks import CheckResult
from evals.config import Case, CheckSpec
from evals.judge import JudgeResult


@dataclass
class ScoredCheck:
    key: str
    label: str
    kind: str            # deterministic | judge
    turn: int | str | None
    passed: bool
    score: float         # raw
    max_score: float
    weight: float
    evidence: str = ""
    reason: str = ""
    detail: dict = field(default_factory=dict)
    # The judge failed to produce a score. Recorded so the row still shows up
    # in the dashboard, but excluded from every aggregate below.
    errored: bool = False

    @property
    def normalized(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0

    @property
    def is_hard(self) -> bool:
        return self.weight >= 2 and not self.errored

    @property
    def counts_toward_score(self) -> bool:
        return not self.errored


@dataclass
class CaseScore:
    case_id: str
    checks: list[ScoredCheck]
    score: float
    pass_rate: float
    passed_hard: bool


def collect(
    case: Case,
    det: list[tuple[CheckSpec, CheckResult]],
    judged: list[JudgeResult],
) -> list[ScoredCheck]:
    out: list[ScoredCheck] = []
    for spec, result in det:
        out.append(
            ScoredCheck(
                key=spec.label,
                label=_describe(spec),
                kind="deterministic",
                turn=spec.turn if isinstance(spec.turn, int) else None,
                passed=result.passed,
                score=result.score,
                max_score=result.max_score,
                weight=spec.weight,
                evidence=result.evidence,
                detail=result.detail,
            )
        )
    for j in judged:
        out.append(
            ScoredCheck(
                key=j.key,
                label=j.key.replace("judge:", "").replace("_", " "),
                kind="judge",
                turn=None,
                passed=j.passed,
                score=j.score,
                max_score=j.max_score,
                weight=j.weight,
                evidence=j.evidence,
                reason=j.reason,
                detail=j.detail,
                errored=j.errored,
            )
        )
    return out


def score_case(case: Case, checks: list[ScoredCheck]) -> CaseScore:
    if not checks:
        return CaseScore(case.id, [], 0.0, 0.0, False)
    # Rubrics the judge couldn't score are dropped from the denominator, not
    # counted as zero. Scoring them as failures would attribute a grader
    # outage to the model — and since the judge shares an API account with
    # some of the models under test, an outage hits both at once and would
    # look exactly like a model collapsing.
    scored = [c for c in checks if c.counts_toward_score]
    if not scored:
        return CaseScore(case.id, checks, 0.0, 0.0, False)
    total_weight = sum(c.weight for c in scored) or 1.0
    score = sum(c.normalized * c.weight for c in scored) / total_weight
    hard = [c for c in scored if c.is_hard]
    pass_rate = (sum(1 for c in hard if c.passed) / len(hard)) if hard else 1.0
    return CaseScore(
        case_id=case.id,
        checks=checks,
        score=round(score, 4),
        pass_rate=round(pass_rate, 4),
        passed_hard=all(c.passed for c in hard),
    )


@dataclass
class RepeatRollup:
    case_id: str
    suite: str
    n_repeats: int
    score_mean: float
    score_min: float
    score_max: float
    score_stdev: float
    pass_rate: float
    n_errors: int


def rollup_repeats(case: Case, scores: list[CaseScore], n_errors: int) -> RepeatRollup:
    """Collapse a case's repeats.

    `score_stdev` is the reason repeats exist: a case whose score swings between
    identical runs of the same model is telling you the rubric or the prompt is
    unstable, which is a different problem from the model being worse — and one
    a single run cannot distinguish.
    """
    values = [s.score for s in scores] or [0.0]
    return RepeatRollup(
        case_id=case.id,
        suite=case.suite,
        n_repeats=len(scores),
        score_mean=round(statistics.fmean(values), 4),
        score_min=round(min(values), 4),
        score_max=round(max(values), 4),
        score_stdev=round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        pass_rate=round(statistics.fmean([s.pass_rate for s in scores] or [0.0]), 4),
        n_errors=n_errors,
    )


@dataclass
class RunSummary:
    score: float
    pass_rate: float
    suite_scores: dict[str, float]
    n_cases: int
    n_errors: int


def summarize(rollups: list[RepeatRollup]) -> RunSummary:
    by_suite: dict[str, list[RepeatRollup]] = {}
    for r in rollups:
        by_suite.setdefault(r.suite, []).append(r)
    suite_scores = {
        suite: round(statistics.fmean([r.score_mean for r in rs]), 4)
        for suite, rs in by_suite.items()
    }
    return RunSummary(
        # Mean of suite means: every suite counts once regardless of how many
        # cases it happens to contain.
        score=round(statistics.fmean(list(suite_scores.values())), 4) if suite_scores else 0.0,
        pass_rate=round(statistics.fmean([r.pass_rate for r in rollups]), 4) if rollups else 0.0,
        suite_scores=suite_scores,
        n_cases=len(rollups),
        n_errors=sum(r.n_errors for r in rollups),
    )


def _describe(spec: CheckSpec) -> str:
    """Human-readable label for the dashboard checklist."""
    from evals.checks import CHECK_REGISTRY

    entry = CHECK_REGISTRY.get(spec.key)
    base = (entry.label if entry else spec.key).replace("_", " ")
    bits = [f"{k}={v}" for k, v in spec.args.items()]
    return f"{base} ({', '.join(bits)})" if bits else base
