"""Layer B — LLM-as-judge.

Scores only what a regex cannot: did it answer, is the substance real, did it
follow the skill's workflow, are the numbers the ones the tools returned.
Anything decidable deterministically stays in `checks.py` — handing "count the
charts" to a judge would add variance and cost to a question with an exact
answer.

Two design choices do the heavy lifting on reliability:

- **Evidence is required.** Every score must quote the span it is scoring.
  A judge that has to point at the text before scoring it produces far fewer
  confidently-wrong 2s than one that only emits a number.
- **The judge sees the trace, not just the answer.** The most valuable rubrics
  here (`number_from_tool`, `chart_data_real`, `no_mental_math`) are all
  variants of "is this figure the one the tool actually returned", which is
  unanswerable from the prose alone.
"""
from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from evals.config import Case, JudgeSpec
from evals.trace import RunTrace

# The judge is no longer outside the set under test.
#
# It used to be: when the roster was Cerebras/Groq/Gemini only, an OpenAI judge
# shared a vendor with nothing it graded. Adding gpt-5.6-luna / gpt-5.6-terra /
# gpt-5.4-mini / gpt-5.4-nano put OpenAI models on both sides, and with
# Cerebras, Groq, Google and OpenAI all now represented in the roster there is
# no vendor left to judge from that isn't also being judged.
#
# Kept as a single fixed judge anyway, because for a leaderboard comparability
# beats neutrality: rotating the judge per model would remove self-preference
# but make the scores incomparable, since two judges differ in strictness and
# every cross-model gap would then mix "better answer" with "easier grader" —
# a worse failure than the bias it fixes.
#
# The bias is handled by being visible rather than hidden: `eval_runs` records
# both `judge_model` and the model's `provider`, `_warn_judge_conflicts` in
# cli.py flags same-vendor pairs at startup, and `--judge-model` re-runs the
# affected models against a second judge as a cross-check.
DEFAULT_JUDGE_MODEL = "openai:gpt-5.6-terra"

# Used only to flag same-vendor judging; never to pick a judge automatically.
CROSS_CHECK_JUDGE = "google_genai:gemini-3.6-flash"


def judge_vendor(model_str: str) -> str:
    """Provider prefix of a `provider:model` judge string."""
    return model_str.split(":", 1)[0] if ":" in model_str else ""

_MAX_TRACE_STEPS = 40
_MAX_RESULT_HEAD = 400
_MAX_ANSWER_CHARS = 24000


class RubricScore(BaseModel):
    key: str = Field(description="The rubric key being scored, copied verbatim.")
    score: int = Field(description="0 = fails the rubric, 1 = partially meets it, 2 = fully meets it.")
    evidence: str = Field(
        description="A short verbatim quote from the answer, report, or trace that justifies "
        "this score. Required. If nothing in the output is relevant, say so explicitly."
    )
    reason: str = Field(description="One sentence explaining the score.")


class JudgeVerdict(BaseModel):
    scores: list[RubricScore] = Field(description="One entry per rubric, in the order given.")


@dataclass
class JudgeResult:
    key: str
    score: float
    max_score: float
    weight: float
    passed: bool
    evidence: str = ""
    reason: str = ""
    detail: dict = field(default_factory=dict)
    # True when the judge itself failed (API error, malformed output, no score
    # for this rubric) rather than the model earning a zero. Scoring drops
    # these instead of counting them, because a judge outage is not evidence
    # about the model — and the two are easy to confuse when the judge runs on
    # the same account as the models under test, where running out of credits
    # takes down grader and subject together.
    errored: bool = False


_SYSTEM = """You are grading the output of an AI assistant against a rubric. You are a
strict, fair examiner: you reward substance and penalise padding, hedging, and
answers that restate the question instead of answering it.

Rules:
- Score each rubric 0, 1, or 2. Use the full range. A 2 means the rubric is
  fully satisfied, not merely "not violated".
- Quote real text as evidence. Never invent a quote. If the output contains
  nothing relevant to a rubric, score 0 and say the output has nothing relevant.
- Judge only the rubric you are given. Do not deduct for things another rubric
  covers, and do not reward length.
- The TOOL TRACE shows what the assistant retrieved. When a rubric asks whether
  a figure or name is real, compare it against the trace: a number that appears
  nowhere in the tool results was fabricated, however plausible it looks.
"""


def _render_trace(trace: RunTrace) -> str:
    lines = []
    steps = trace.as_trace_json()[:_MAX_TRACE_STEPS]
    for step in steps:
        args = ", ".join(f"{k}={v!r}" for k, v in list(step["args"].items())[:3])
        head = (step.get("result_head") or "").replace("\n", " ")[:_MAX_RESULT_HEAD]
        lines.append(f"[{step['i']}] {step['name']}({args}) -> {head}")
    if len(trace.as_trace_json()) > _MAX_TRACE_STEPS:
        lines.append(f"… ({len(trace.as_trace_json()) - _MAX_TRACE_STEPS} more steps omitted)")
    return "\n".join(lines) or "(no tool calls)"


def _render_output(trace: RunTrace) -> str:
    from evals import parsers

    parts = []
    for turn in trace.turns:
        parts.append(f"--- USER (turn {turn.index}) ---\n{turn.query}")
        parts.append(f"--- ASSISTANT (turn {turn.index}) ---\n{turn.text}")
        for report in parsers.extract_reports(turn.text):
            parts.append(
                f"--- REPORT in turn {turn.index} (title={report.title!r}, "
                f"{report.words} words) ---\n{report.body}"
            )
    text = "\n\n".join(parts)
    if len(text) > _MAX_ANSWER_CHARS:
        # Keep both ends: rubrics about openings and about conclusions both exist.
        half = _MAX_ANSWER_CHARS // 2
        text = text[:half] + "\n\n…[middle omitted]…\n\n" + text[-half:]
    return text


def _build_prompt(case: Case, trace: RunTrace, rubrics: list[JudgeSpec]) -> str:
    rubric_lines = "\n".join(
        f"- {r.key}: {r.prompt}" for r in rubrics
    )
    return f"""ORIGINAL USER REQUEST
{case.turns[0].text}

TOOL TRACE (what the assistant retrieved)
{_render_trace(trace)}

ASSISTANT OUTPUT
{_render_output(trace)}

RUBRICS TO SCORE
{rubric_lines}

Score every rubric listed above, in order, using its exact key."""


async def judge_case(
    case: Case,
    trace: RunTrace,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    repeats: int = 1,
) -> list[JudgeResult]:
    """Score a case's judge rubrics.

    `repeats > 1` scores the same output several times and takes the **median**
    — a cheap read on the judge's own variance, which is worth knowing before
    trusting a 0.05 gap between two models.
    """
    if not case.judge:
        return []

    llm = _get_judge_llm(model)
    prompt = _build_prompt(case, trace, case.judge)
    by_key: dict[str, list[RubricScore]] = {r.key: [] for r in case.judge}

    verdicts = await asyncio.gather(
        *[_invoke(llm, prompt) for _ in range(max(1, repeats))],
        return_exceptions=True,
    )
    errors = [v for v in verdicts if isinstance(v, Exception)]
    for verdict in verdicts:
        if isinstance(verdict, Exception):
            continue
        for item in verdict.scores:
            if item.key in by_key:
                by_key[item.key].append(item)

    results = []
    for spec in case.judge:
        got = by_key.get(spec.key) or []
        if not got:
            detail = {"judge_error": str(errors[0])[:300]} if errors else {"judge_error": "no score returned"}
            results.append(
                JudgeResult(
                    key=spec.label, score=0.0, max_score=2.0, weight=spec.weight,
                    passed=False, evidence="", reason="judge returned no score for this rubric",
                    detail=detail, errored=True,
                )
            )
            continue
        score = statistics.median(s.score for s in got)
        best = max(got, key=lambda s: s.score)
        results.append(
            JudgeResult(
                key=spec.label,
                score=float(score),
                max_score=2.0,
                weight=spec.weight,
                passed=score >= spec.pass_at,
                evidence=best.evidence[:1000],
                reason=best.reason[:500],
                detail={"scores": [s.score for s in got]} if len(got) > 1 else {},
            )
        )
    return results


_JUDGE_CACHE: dict[str, object] = {}


def _get_judge_llm(model: str):
    if model not in _JUDGE_CACHE:
        from langchain.chat_models import init_chat_model

        _JUDGE_CACHE[model] = init_chat_model(model, temperature=0)
    return _JUDGE_CACHE[model]


async def _invoke(llm, prompt: str) -> JudgeVerdict:
    structured = llm.with_structured_output(JudgeVerdict)
    return await structured.ainvoke(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    )
