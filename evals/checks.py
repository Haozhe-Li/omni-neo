"""Layer A — deterministic checks.

Each check is a pure function of `(RunTrace, CheckSpec)` returning a
`CheckResult`. No LLM, no network, no randomness: run the same trace twice and
get the same verdict, which is what lets these carry the load-bearing weights.

The registry holds check *types* only. Every threshold, tool name and skill
name comes from `cases.yaml` — a check here never knows which case it is
serving.

`evidence` is mandatory in spirit on every failure: a red row in the dashboard
that doesn't say *what* was wrong costs more time than it saves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from evals import parsers
from evals.config import CheckSpec
from evals.trace import RunTrace, TurnTrace

# The charting skill's mandated series colours. A model that ignores the skill
# and lets ECharts pick its defaults produces a chart that renders but visibly
# doesn't belong to the product — worth catching, but only at weight 1, since
# it's cosmetic next to "the JSON doesn't parse".
PALETTE = {"#20B2AA", "#005A5A", "#7B9E9E", "#C4A882", "#8B7D6B", "#5B8FA8"}


@dataclass
class CheckResult:
    passed: bool
    score: float
    max_score: float = 1.0
    evidence: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, evidence: str = "", **detail: Any) -> "CheckResult":
        return cls(True, 1.0, 1.0, evidence, detail)

    @classmethod
    def fail(cls, evidence: str, **detail: Any) -> "CheckResult":
        return cls(False, 0.0, 1.0, evidence, detail)

    @classmethod
    def partial(cls, score: float, evidence: str, **detail: Any) -> "CheckResult":
        return cls(score >= 1.0, max(0.0, min(1.0, score)), 1.0, evidence, detail)


@dataclass
class RegistryEntry:
    fn: Callable[[RunTrace, CheckSpec], CheckResult]
    arg_names: tuple[str, ...] = ()
    required_args: tuple[str, ...] = ()
    label: str = ""


CHECK_REGISTRY: dict[str, RegistryEntry] = {}


def check(key: str, *, args: tuple[str, ...] = (), required: tuple[str, ...] = (), label: str = ""):
    def deco(fn):
        CHECK_REGISTRY[key] = RegistryEntry(fn, args, required, label or key)
        return fn
    return deco


# ── helpers ─────────────────────────────────────────────────────────────────
def _turns(trace: RunTrace, spec: CheckSpec) -> list[TurnTrace]:
    return trace.select(spec.turn)


def _text(trace: RunTrace, spec: CheckSpec) -> str:
    return "\n\n".join(t.text for t in _turns(trace, spec))


def _all_calls(trace: RunTrace, spec: CheckSpec):
    return [c for t in _turns(trace, spec) for c in t.tool_calls]


def _in_range(value: int, args: dict, default_min: int = 0) -> bool:
    lo = args.get("min", default_min)
    hi = args.get("max")
    return value >= lo and (hi is None or value <= hi)


def _range_str(args: dict, default_min: int = 0) -> str:
    lo, hi = args.get("min", default_min), args.get("max")
    return f"[{lo}, {'∞' if hi is None else hi}]"


# ═══════════════════════════════════════════════════════════════════════════
# 3.1 Skill triggering
# ═══════════════════════════════════════════════════════════════════════════
@check("skill_loaded", args=("skill",), required=("skill",))
def _skill_loaded(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    want = spec.args["skill"]
    loaded = [s for t in _turns(trace, spec) for s in t.skills_loaded]
    if want in loaded:
        return CheckResult.ok(f"read /skills/{want}/SKILL.md", loaded=loaded)
    return CheckResult.fail(
        f"never read /skills/{want}/SKILL.md; loaded: {loaded or 'none'}", loaded=loaded
    )


@check("skill_not_loaded", args=("skill",), required=("skill",))
def _skill_not_loaded(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    want = spec.args["skill"]
    loaded = [s for t in _turns(trace, spec) for s in t.skills_loaded]
    if want in loaded:
        return CheckResult.fail(f"loaded {want} when it should not have", loaded=loaded)
    return CheckResult.ok(f"did not load {want}", loaded=loaded)


@check("skill_load_order", args=("before", "after"), required=("before", "after"))
def _skill_load_order(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    before, after = spec.args["before"], spec.args["after"]
    turns = _turns(trace, spec)
    i_before = next((t.skill_load_index(before) for t in turns if t.skill_load_index(before) is not None), None)
    i_after = next((t.skill_load_index(after) for t in turns if t.skill_load_index(after) is not None), None)
    if i_before is None or i_after is None:
        return CheckResult.fail(f"missing one of the skills ({before}={i_before}, {after}={i_after})")
    if i_before < i_after:
        return CheckResult.ok(f"{before}@{i_before} before {after}@{i_after}")
    return CheckResult.fail(f"{after}@{i_after} came before {before}@{i_before}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.2 Tool usage
# ═══════════════════════════════════════════════════════════════════════════
@check("tool_called", args=("tool", "min", "max"), required=("tool",))
def _tool_called(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    name = spec.args["tool"]
    n = sum(len(t.tools_named(name)) for t in _turns(trace, spec))
    if _in_range(n, spec.args, default_min=1):
        return CheckResult.ok(f"{name} called {n}x", count=n)
    return CheckResult.fail(
        f"{name} called {n}x, expected {_range_str(spec.args, 1)}", count=n
    )


@check("no_tool_calls")
def _no_tool_calls(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    calls = _all_calls(trace, spec)
    if not calls:
        return CheckResult.ok("no tool calls")
    names = sorted({c.name for c in calls})
    return CheckResult.fail(f"called {len(calls)} tool(s): {names}", tools=names)


@check("distinct_queries", args=("tool", "min"), required=("tool",))
def _distinct_queries(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    tool = spec.args["tool"]
    want = int(spec.args.get("min", 1))
    queries = set()
    for t in _turns(trace, spec):
        queries |= t.distinct_queries(tool)
    total = sum(len(t.tools_named(tool)) for t in _turns(trace, spec))
    if len(queries) >= want:
        return CheckResult.ok(f"{len(queries)} distinct of {total} calls", queries=sorted(queries))
    return CheckResult.fail(
        f"only {len(queries)} distinct {tool} queries across {total} calls "
        f"(want >= {want}) — repeated searching, not broader coverage",
        queries=sorted(queries),
    )


@check("distinct_domains", args=("min",))
def _distinct_domains(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    want = int(spec.args.get("min", 1))
    domains: set[str] = set()
    for t in _turns(trace, spec):
        domains |= t.distinct_domains
    if len(domains) >= want:
        return CheckResult.ok(f"{len(domains)} domains", domains=sorted(domains))
    return CheckResult.fail(
        f"only {len(domains)} distinct domain(s), want >= {want}", domains=sorted(domains)
    )


@check("search_discipline", args=("max_per_topic",))
def _search_discipline(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """SYSTEM_PROMPT caps searches per sub-topic. Sub-topics aren't labelled in the
    trace, so this approximates one by near-identical queries: the same query
    re-run more than the cap is the behaviour the rule exists to stop."""
    cap = int(spec.args.get("max_per_topic", 2))
    counts: dict[str, int] = {}
    for t in _turns(trace, spec):
        for call in t.tools_named("google_search"):
            q = str(call.args.get("query") or "").strip().lower()
            key = " ".join(sorted(q.split()))  # order-insensitive fingerprint
            counts[key] = counts.get(key, 0) + 1
    offenders = {q: n for q, n in counts.items() if n > cap}
    if not offenders:
        return CheckResult.ok(f"no query repeated more than {cap}x")
    return CheckResult.fail(f"repeated searches beyond cap {cap}: {offenders}", offenders=offenders)


# ═══════════════════════════════════════════════════════════════════════════
# 3.3 Output contracts
# ═══════════════════════════════════════════════════════════════════════════
@check("has_report", args=("min_words", "max_words", "require_title"))
def _has_report(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    reports = parsers.extract_reports(_text(trace, spec))
    if not reports:
        return CheckResult.fail("no <report> block")
    report = max(reports, key=lambda r: r.words)
    problems = []
    if spec.args.get("require_title") and not report.title:
        problems.append("missing title attribute")
    lo = spec.args.get("min_words")
    hi = spec.args.get("max_words")
    if lo is not None and report.words < lo:
        problems.append(f"{report.words} words < {lo}")
    if hi is not None and report.words > hi:
        problems.append(f"{report.words} words > {hi}")
    detail = {"words": report.words, "title": report.title, "n_reports": len(reports)}
    if problems:
        return CheckResult.fail("; ".join(problems), **detail)
    return CheckResult.ok(f"report '{report.title}' — {report.words} words", **detail)


@check("no_report")
def _no_report(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    reports = parsers.extract_reports(_text(trace, spec))
    if not reports:
        return CheckResult.ok("no report, as expected")
    return CheckResult.fail(
        f"wrote a {reports[0].words}-word report when none was warranted",
        words=reports[0].words,
    )


@check("chart_count", args=("min", "max"))
def _chart_count(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    n = len(parsers.extract_charts(_text(trace, spec)))
    if _in_range(n, spec.args):
        return CheckResult.ok(f"{n} chart(s)", count=n)
    return CheckResult.fail(f"{n} chart(s), expected {_range_str(spec.args)}", count=n)


@check("charts_valid", args=("require_palette", "min_series"))
def _charts_valid(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    charts = parsers.extract_charts(_text(trace, spec))
    if not charts:
        return CheckResult.fail("no ```echarts fences to validate")
    min_series = int(spec.args.get("min_series", 1))
    problems: list[str] = []
    off_palette = 0
    for i, block in enumerate(charts):
        if not block.ok:
            problems.append(f"chart {i}: {block.error}")
            continue
        option = block.data
        if not isinstance(option, dict):
            problems.append(f"chart {i}: option is {type(option).__name__}, not an object")
            continue
        series = option.get("series")
        series = series if isinstance(series, list) else ([series] if series else [])
        if len(series) < min_series:
            problems.append(f"chart {i}: {len(series)} series, want >= {min_series}")
        if spec.args.get("require_palette"):
            colors = option.get("color")
            if not (isinstance(colors, list) and colors and set(colors[: len(PALETTE)]) <= PALETTE):
                off_palette += 1
    if problems:
        return CheckResult.fail("; ".join(problems[:3]), n_charts=len(charts))
    if off_palette:
        # Cosmetic next to a JSON error, so partial rather than a hard fail.
        return CheckResult.partial(
            0.5,
            f"{off_palette}/{len(charts)} chart(s) not using the skill's palette",
            n_charts=len(charts),
        )
    return CheckResult.ok(f"{len(charts)} chart(s) valid", n_charts=len(charts))


@check("map_fence", args=("min", "max_pins"))
def _map_fence(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    maps = parsers.extract_maps(_text(trace, spec))
    want = int(spec.args.get("min", 1))
    if len(maps) < want:
        return CheckResult.fail(f"{len(maps)} ```map fence(s), want >= {want}")
    max_pins = int(spec.args.get("max_pins", 8))
    problems: list[str] = []
    total_pins = 0
    for i, block in enumerate(maps):
        if not block.ok:
            problems.append(f"map {i}: {block.error}")
            continue
        data = block.data
        if not isinstance(data, dict):
            problems.append(f"map {i}: not an object")
            continue
        if not data.get("title"):
            problems.append(f"map {i}: missing title")
        pins = data.get("pins")
        if not isinstance(pins, list) or not pins:
            problems.append(f"map {i}: no pins")
            continue
        total_pins += len(pins)
        if len(pins) > max_pins:
            problems.append(f"map {i}: {len(pins)} pins > {max_pins}")
        if any(not isinstance(p, dict) or not p.get("name") for p in pins):
            problems.append(f"map {i}: a pin is missing `name`")
    if problems:
        return CheckResult.fail("; ".join(problems[:3]), n_maps=len(maps), pins=total_pins)
    return CheckResult.ok(f"{len(maps)} map(s), {total_pins} pins", n_maps=len(maps), pins=total_pins)


@check("no_map")
def _no_map(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    maps = parsers.extract_maps(_text(trace, spec))
    if maps:
        return CheckResult.fail(f"emitted {len(maps)} map fence(s) unnecessarily")
    return CheckResult.ok("no map, as expected")


_QUESTION_TYPES = {"single", "multiple", "text"}


@check("question_block", args=("min_q", "max_q", "must_be_last"))
def _question_block(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    block = parsers.extract_question(_text(trace, spec))
    if block is None:
        return CheckResult.fail("no <question> block")
    if not block.parsed.ok:
        return CheckResult.fail(f"<question> body {block.parsed.error}")
    data = block.parsed.data
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        return CheckResult.fail("<question> JSON has no `questions` array")
    questions = data["questions"]
    problems: list[str] = []
    lo = int(spec.args.get("min_q", 1))
    hi = int(spec.args.get("max_q", 8))
    if not (lo <= len(questions) <= hi):
        problems.append(f"{len(questions)} questions, expected {lo}-{hi}")
    ids = [q.get("id") for q in questions if isinstance(q, dict)]
    if len(set(ids)) != len(ids):
        problems.append(f"duplicate question ids: {ids}")
    for q in questions:
        if not isinstance(q, dict):
            problems.append("a question is not an object")
            continue
        if q.get("type") not in _QUESTION_TYPES:
            problems.append(f"bad type {q.get('type')!r}")
        if not q.get("prompt"):
            problems.append(f"question {q.get('id')} has no prompt")
        if q.get("type") in {"single", "multiple"} and not q.get("options"):
            problems.append(f"question {q.get('id')} is {q.get('type')} with no options")
    if spec.args.get("must_be_last", True) and not block.is_last:
        # The frontend renders the block as a form; anything after it is
        # unreachable, so trailing prose is a real rendering bug, not a nit.
        problems.append(f"content after the block: {block.trailing[:80]!r}")
    if problems:
        return CheckResult.fail("; ".join(problems[:3]), n_questions=len(questions))
    return CheckResult.ok(f"{len(questions)} well-formed question(s)", n_questions=len(questions))


@check("no_question_block")
def _no_question_block(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    if parsers.extract_question(_text(trace, spec)) is not None:
        return CheckResult.fail("asked a clarifying question when the request was already clear")
    return CheckResult.ok("no question block, as expected")


@check(
    "textblock",
    args=("min", "max", "require_type", "require_subject", "min_words", "max_words"),
)
def _textblock(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """The writing path's output contract, per SYSTEM_PROMPT's "Writing and
    Rewrites" section: the finished deliverable — and nothing else — goes in a
    `<textblock>…</textblock>`.

    Replaces `has_delimiters`, which looked for `---` horizontal rules. That
    convention was retired from the prompt, and `_S_FORMATTING_PRO` now says to
    use headers *instead of* horizontal rules — so the old check scored a model
    down for following the current instructions, on a weight-3 item. Nothing
    caught it because the case it sits on is outside the smoke set.

    `require_type: email` also demands the `subject="…"` attribute, since the
    prompt pairs them; a drafted email without a subject line is not a usable
    deliverable.
    """
    blocks = parsers.extract_textblocks(_text(trace, spec))
    args = spec.args
    if not blocks:
        return CheckResult.fail("no <textblock> deliverable")
    if not _in_range(len(blocks), args):
        return CheckResult.fail(
            f"{len(blocks)} <textblock>(s), expected {_range_str(args)}", n=len(blocks)
        )

    problems: list[str] = []
    want_type = args.get("require_type")
    if want_type:
        wrong = [b.type for b in blocks if (b.type or "") != want_type]
        if wrong:
            problems.append(f'type={wrong[0]!r}, expected "{want_type}"')
    if args.get("require_subject") or want_type == "email":
        if any(not b.subject for b in blocks):
            problems.append("missing subject=\"…\" on the opening tag")
    decorated = [d for b in blocks for d in b.decoration]
    if decorated:
        problems.append(f"markdown/citations inside the block: {decorated[:2]}")

    words = sum(b.words for b in blocks)
    lo, hi = args.get("min_words"), args.get("max_words")
    if (lo is not None and words < lo) or (hi is not None and words > hi):
        problems.append(f"{words} words in the deliverable, expected [{lo}, {hi}]")

    if problems:
        return CheckResult.fail("; ".join(problems[:3]), n=len(blocks), words=words)
    return CheckResult.ok(f"{len(blocks)} well-formed <textblock>", n=len(blocks), words=words)


@check("no_textblock")
def _no_textblock(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    blocks = parsers.extract_textblocks(_text(trace, spec))
    if blocks:
        return CheckResult.fail(f"{len(blocks)} <textblock> on a non-writing task")
    return CheckResult.ok("no textblock, as expected")


@check("followup_question")
def _followup_question(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    if parsers.ends_with_question(parsers.prose_only(_text(trace, spec))):
        return CheckResult.ok("ends with a follow-up question")
    return CheckResult.fail("no follow-up question after the writing task")


# ═══════════════════════════════════════════════════════════════════════════
# 3.4 Format compliance
# ═══════════════════════════════════════════════════════════════════════════
@check("word_count", args=("min", "max"))
def _word_count(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Counts prose only.

    Reports, charts and question blocks are stripped first: a `max` on a chat
    answer is about how much the model *said*, and a 400-line ECharts option
    counting toward it would make every threshold meaningless. Reports get
    their own budget via `has_report(min_words=...)`.
    """
    text = parsers.prose_only(_text(trace, spec))
    n = parsers.count_words(text)
    if _in_range(n, spec.args):
        return CheckResult.ok(f"{n} words", words=n)
    return CheckResult.fail(f"{n} words, expected {_range_str(spec.args)}", words=n)


@check("citation_count", args=("min",))
def _citation_count(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    want = int(spec.args.get("min", 1))
    found = parsers.find_citations(_text(trace, spec)).distinct
    if len(found) >= want:
        return CheckResult.ok(f"{len(found)} distinct citations", cited=sorted(found))
    return CheckResult.fail(
        f"only {len(found)} distinct citation(s), want >= {want}", cited=sorted(found)
    )


@check("citation_exists")
def _citation_exists(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Every [n] must correspond to a source the tools actually registered.

    The registry is built by the retrieval tools themselves, before their
    results ever reach the model, so it is exactly the set of numbers the model
    is entitled to use. A citation outside it is fabricated — and this is a
    quiet, high-cost failure: the frontend renders a footnote marker that opens
    nothing or points at the wrong source, so the answer *looks* sourced while
    the claim is unbacked.
    """
    cited = parsers.find_citations(_text(trace, spec)).distinct
    if not cited:
        return CheckResult.ok("no citations to verify")
    registered = trace.citation_numbers()
    invented = sorted(cited - registered)
    if invented:
        return CheckResult.fail(
            f"citation(s) {invented} were never registered by any tool "
            f"(registry has 1..{max(registered) if registered else 0})",
            invented=invented,
            registered=sorted(registered),
        )
    return CheckResult.ok(f"all {len(cited)} citations resolve", cited=sorted(cited))


@check("citation_required")
def _citation_required(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Fail when the tools handed the model citable sources and it cited none.

    This exists because `citation_exists` and `citation_format` both pass
    *vacuously*: an answer with zero `[n]` markers has no fabricated citation
    and no malformed one, so both return ok. A model that simply never cites
    therefore scored full marks on two of the three citation checks, and the
    only one that required citations at all (`citation_count`) is configured on
    four cases. Measured on the sft-v1 run, 31 of its 42 `citation_exists`
    passes were this vacuum — the aggregate read 95% while the model was
    actually silent three quarters of the time.

    Self-configuring, so it can sit in `common_checks` without a per-case
    argument: the trigger is the citation registry, which the retrieval tools
    populate themselves. An empty registry means nothing citable was retrieved
    (a `run_python` answer, a pure rewrite, a clarifying question, a
    `get_stock_data` lookup — that tool registers no sources), and the check
    goes inert rather than punishing correct restraint.

    Deliberately a separate key rather than a stricter `citation_exists`:
    "cited something that does not exist" and "cited nothing at all" are
    different failures with different fixes, and collapsing them into one
    number would hide whichever one is currently dominant.
    """
    registered = trace.citation_numbers()
    if not registered:
        return CheckResult.ok("no citable sources registered — nothing to cite")
    cited = parsers.find_citations(_text(trace, spec)).distinct
    if not cited:
        return CheckResult.fail(
            f"{len(registered)} source(s) were available and the answer cites none",
            registered=sorted(registered),
        )
    return CheckResult.ok(
        f"cites {len(cited)} of {len(registered)} available source(s)",
        cited=sorted(cited),
    )


@check("citation_format")
def _citation_format(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    cites = parsers.find_citations(_text(trace, spec))
    problems = []
    if cites.bad_format:
        problems.append(f"non-ASCII citation markers: {cites.bad_format[:2]}")
    if cites.midsentence:
        problems.append(f"mid-sentence citation: …{cites.midsentence[0]}…")
    if problems:
        return CheckResult.fail("; ".join(problems))
    return CheckResult.ok("citation markers well-formed")


@check("citation_coverage", args=("min_ratio",))
def _citation_coverage(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Catch under-citing: paragraphs that clearly carry retrieved facts but
    have no [n] at all.

    Necessarily heuristic — "looks like a retrieved fact" is approximated by
    numbers, years and percentages — so it scores as a ratio with a low weight
    rather than pass/fail, and false positives cost a fraction of a point
    instead of failing the case.
    """
    want = float(spec.args.get("min_ratio", 0.7))
    text = parsers.prose_only(_text(trace, spec))
    for report in parsers.extract_reports(_text(trace, spec)):
        text += "\n\n" + report.body
    factual = []
    for para in parsers.paragraphs(text):
        if len(para) < 60:
            continue
        if re.search(r"\b(19|20)\d{2}\b|\d+(\.\d+)?\s*%|\d{3,}", para):
            factual.append(para)
    if not factual:
        return CheckResult.ok("no fact-bearing paragraphs to check")
    cited = sum(1 for p in factual if parsers.find_citations(p).numbers)
    ratio = cited / len(factual)
    if ratio >= want:
        return CheckResult.ok(f"{cited}/{len(factual)} fact paragraphs cited", ratio=round(ratio, 2))
    return CheckResult.partial(
        ratio / want,
        f"only {cited}/{len(factual)} fact-bearing paragraphs carry a citation "
        f"(ratio {ratio:.2f} < {want})",
        ratio=round(ratio, 2),
    )


@check("no_hyperlinks")
def _no_hyperlinks(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    links = parsers.find_hyperlinks(_text(trace, spec))
    if links:
        return CheckResult.fail(f"{len(links)} hyperlink(s): {links[:2]}", links=links)
    return CheckResult.ok("no hyperlinks")


@check("no_ascii_art")
def _no_ascii_art(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    art = parsers.find_ascii_art(_text(trace, spec))
    if art:
        return CheckResult.fail(f"text-art block: {art[0][:80]!r}")
    return CheckResult.ok("no text art")


@check("tool_discipline")
def _tool_discipline(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """SYSTEM_PROMPT: a turn is 100% tool calls or 100% final text, never both.

    Violations are what produce "Let me search for that…" narration leaking
    into the answer stream ahead of the real reply.
    """
    offenders: list[str] = []
    for turn in _turns(trace, spec):
        offenders.extend(turn.mixed_messages)
    if offenders:
        return CheckResult.fail(
            f"{len(offenders)} message(s) mixed prose with tool calls: {offenders[0][:100]!r}"
        )
    return CheckResult.ok("no mixed tool/text messages")


@check("no_leading_header")
def _no_leading_header(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    for turn in _turns(trace, spec):
        if parsers.starts_with_header(turn.text):
            first = next((l for l in turn.text.splitlines() if l.strip()), "")
            return CheckResult.fail(f"answer opens with a header: {first[:60]!r}")
    return CheckResult.ok("opens with content")


@check("latex_sanity")
def _latex_sanity(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    problems = parsers.find_latex_problems(_text(trace, spec))
    if problems:
        return CheckResult.fail("; ".join(problems[:3]))
    return CheckResult.ok("LaTeX usage conforms")


@check("response_language", args=("expect",))
def _response_language(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Did the answer come back in the language it was supposed to?

    Two rules are in play, and which one applies is decided by the
    `<personalization>` block, exactly as in production:

    - When personalization states a response language, SYSTEM_PROMPT says to
      honour it silently — so that language wins even if the query was written
      in another one.
    - When it states none, the answer must follow the language of the query.

    The interesting cases are the mixed ones. "你可以告诉我the current valuation
    of langchain吗?" is a Chinese question that happens to contain an English
    noun phrase, and answering it in English is wrong; "could you please explain
    what does 你好 mean?" is the mirror image. Models routinely flip to whichever
    language the *keywords* were in, which reads to the user as the assistant
    switching languages mid-conversation for no reason.

    `expect` comes from the case, not from auto-detecting the query, because
    these examples are chosen precisely to defeat naive detection.
    """
    expect = spec.args.get("expect")
    if not expect:
        return CheckResult.ok("no expected language declared")
    # Everything the user is *told*, not just the prose: an answer that is
    # entirely a `<question>` block is still written in a language. But NOT the
    # `<textblock>` deliverable — "把这段翻译成英文" is correctly answered with
    # Chinese commentary around an English block, and counting the block made
    # every cross-language writing task read as `mixed`. See
    # `parsers.conversational_text`.
    verdict = parsers.detect_language(parsers.conversational_text(_text(trace, spec)))
    # An answer can be too small to have a language at all. "240 的 15% 是多少?"
    # is correctly answered with "36", which lands at 2 CJK chars and 3 Latin
    # words and scored "mixed" — the check demanding a commitment the content
    # cannot make. Terse replies are the *point* on the restraint and
    # no-X-needed cases, so below this floor there is nothing to grade.
    if verdict.cjk_chars + verdict.latin_words < 8:
        return CheckResult.ok("too short to carry a language", **_lang_detail(verdict))
    if verdict.lang == "unknown":
        return CheckResult.fail("answer too short to classify", **_lang_detail(verdict))
    if verdict.lang == expect:
        return CheckResult.ok(f"answered in {expect} — {verdict}", **_lang_detail(verdict))
    if verdict.lang == "mixed":
        # Half-and-half is its own failure, distinct from answering in the
        # wrong language, so it scores partially rather than zero.
        return CheckResult.partial(
            0.5,
            f"answer mixes languages instead of committing to {expect} — {verdict}",
            **_lang_detail(verdict),
        )
    return CheckResult.fail(
        f"answered in {verdict.lang}, expected {expect} — {verdict}", **_lang_detail(verdict)
    )


def _lang_detail(verdict) -> dict:
    return {
        "detected": verdict.lang,
        "ratio": round(verdict.ratio, 3),
        "cjk_chars": verdict.cjk_chars,
        "latin_words": verdict.latin_words,
    }


@check("no_prompt_leak")
def _no_prompt_leak(trace: RunTrace, spec: CheckSpec) -> CheckResult:
    """Reuses the production guard rather than a lookalike.

    If this check and `core/prompt_guard.py` disagreed about what counts as a
    leak, the eval would be grading a policy the product doesn't enforce.

    The armed-guard assertion is not defensive padding. `prompt_guard` starts
    with an *empty* fingerprint set and is only armed by
    `register_sensitive_prompts`, which lives in `main.py` — code the eval
    never imports. Unarmed, `has_prompt_leakage` returns False for every input,
    so this weight-3 check applied to all 24 cases would pass unconditionally
    and silently inflate every score. A check that cannot do its job has to
    fail loudly rather than wave everything through.
    """
    from core.prompt_guard import _DEFAULT_LEAK_GUARD, has_prompt_leakage

    if not getattr(_DEFAULT_LEAK_GUARD, "fingerprint_size", 0):
        return CheckResult.fail(
            "prompt-leak guard is unarmed (no prompts registered) — this check "
            "cannot detect anything; call register_sensitive_prompts at startup"
        )
    for turn in _turns(trace, spec):
        if turn.text and has_prompt_leakage(turn.text):
            return CheckResult.fail("output tripped the production prompt-leakage guard")
    return CheckResult.ok("no prompt leakage")


def run_checks(trace: RunTrace, specs: list[CheckSpec]) -> list[tuple[CheckSpec, CheckResult]]:
    out = []
    for spec in specs:
        entry = CHECK_REGISTRY.get(spec.key)
        if entry is None:  # unreachable after config validation, kept as a guard
            out.append((spec, CheckResult.fail(f"unknown check {spec.key!r}")))
            continue
        if not trace.select(spec.turn):
            out.append((spec, CheckResult.fail(f"turn {spec.turn} produced no output")))
            continue
        try:
            out.append((spec, entry.fn(trace, spec)))
        except Exception as e:  # a broken check must not void the whole case
            out.append((spec, CheckResult.fail(f"check raised {type(e).__name__}: {e}")))
    return out
