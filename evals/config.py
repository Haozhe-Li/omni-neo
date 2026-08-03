"""Load and validate `cases.yaml`.

The YAML is the single source of truth for every query, threshold, weight and
judge prompt. This module turns it into typed objects and — importantly —
fails loudly on anything it doesn't recognise.

That validation is the whole reason this file exists rather than a bare
`yaml.safe_load` at the call site. The failure mode of a config-driven rubric
is silent: misspell `has_report` as `has_reprot` and the check simply never
runs, the case scores 1.0 on a smaller denominator, and nothing anywhere says
so. So every check key is resolved against `checks.CHECK_REGISTRY` and every
`args` key against that check's declared signature *before* a single model is
called.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULT_CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.yaml")

# `turn` selectors a check may carry.
_TURN_SELECTORS = ("last", "any", "all")


@dataclass
class CheckSpec:
    key: str
    args: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    turn: int | str = "last"

    @property
    def is_hard(self) -> bool:
        """Weight >= 2 marks a check as load-bearing; only these count toward
        `pass_rate`. Keeps the headline "did anything actually break" number
        from being diluted by a dozen cosmetic checks."""
        return self.weight >= 2

    @property
    def label(self) -> str:
        """Stable, human-readable identity, also the `eval_checks.key` column.

        Includes the discriminating arg (skill name / tool name) so that two
        `skill_loaded` checks on one case don't collapse into one row — and so
        "which rubric item fails across every model" stays answerable.
        """
        for arg in ("skill", "tool"):
            if arg in self.args:
                return f"{self.key}:{self.args[arg]}"
        return self.key


@dataclass
class JudgeSpec:
    key: str
    prompt: str
    weight: float = 1.0
    pass_at: int = 2

    @property
    def label(self) -> str:
        return f"judge:{self.key}"


@dataclass
class Turn:
    text: str


@dataclass
class Case:
    id: str
    suite: str
    title: str
    turns: list[Turn]
    checks: list[CheckSpec]
    judge: list[JudgeSpec]
    skill: str | None = None
    lang: str = "zh"
    is_negative: bool = False
    weight: float = 1.0
    timeout_s: int = 300
    repeats: int = 2
    citation_grounding: bool = False
    fixture: str | None = None
    personalization: dict[str, str] = field(default_factory=dict)
    expect_lang: str | None = None
    rubric_version: int = 1

    def as_rubric_json(self) -> list[dict]:
        """The rubric as stored in `eval_cases.rubric`, so the dashboard can
        render a case's expectations without importing any Python."""
        return [
            {
                "layer": "deterministic",
                "key": c.label,
                "check": c.key,
                "args": c.args,
                "weight": c.weight,
                "turn": c.turn,
            }
            for c in self.checks
        ] + [
            {
                "layer": "judge",
                "key": j.label,
                "prompt": j.prompt,
                "weight": j.weight,
                "pass_at": j.pass_at,
            }
            for j in self.judge
        ]


@dataclass
class Suite:
    cases: list[Case]
    version: int
    defaults: dict[str, Any]

    def filter(self, suites: list[str] | None, case_ids: list[str] | None) -> list[Case]:
        out = self.cases
        if suites:
            out = [c for c in out if c.suite in suites]
        if case_ids:
            wanted = set(case_ids)
            out = [c for c in out if c.id in wanted]
        return out


class ConfigError(Exception):
    """Raised for any malformed case file. Always fatal — a rubric that only
    half-loaded produces scores that look fine and mean nothing."""


def _coerce_checks(raw: list[dict], where: str) -> list[CheckSpec]:
    out = []
    for i, item in enumerate(raw or []):
        if not isinstance(item, dict) or "key" not in item:
            raise ConfigError(f"{where}: check #{i} must be a mapping with a `key`")
        turn = item.get("turn", "last")
        if not isinstance(turn, int) and turn not in _TURN_SELECTORS:
            raise ConfigError(
                f"{where}: check `{item['key']}` has turn={turn!r}; "
                f"expected an int or one of {_TURN_SELECTORS}"
            )
        out.append(
            CheckSpec(
                key=item["key"],
                args=item.get("args") or {},
                weight=float(item.get("weight", 1)),
                turn=turn,
            )
        )
    return out


def _coerce_judge(raw: list[dict], where: str, default_pass_at: int) -> list[JudgeSpec]:
    out = []
    for i, item in enumerate(raw or []):
        if not isinstance(item, dict) or "key" not in item or "prompt" not in item:
            raise ConfigError(f"{where}: judge #{i} needs both `key` and `prompt`")
        out.append(
            JudgeSpec(
                key=item["key"],
                prompt=item["prompt"].strip(),
                weight=float(item.get("weight", 1)),
                pass_at=int(item.get("pass_at", default_pass_at)),
            )
        )
    return out


def _merge_common_checks(
    common: list[CheckSpec], own: list[CheckSpec], disabled: set[str]
) -> list[CheckSpec]:
    """Fold the shared format-compliance checks into a case.

    A case's own spec wins on collision (so a case can tighten `word_count`
    without the default fighting it), and `disable:` drops a common check
    outright — which cases like the resignation-email and chitchat ones need,
    since demanding citations from an answer that never touched a tool would
    fail them for doing the right thing.
    """
    own_keys = {c.key for c in own}
    # Copied, not shared. The `common` specs are parsed once and would otherwise
    # be the same objects on all 28 cases, so any per-case mutation of one —
    # `_bind_expected_language` filling in `expect`, for instance — would write
    # through to every case and leave the last one processed deciding what all
    # the others check for.
    kept = [
        CheckSpec(key=c.key, args=dict(c.args), weight=c.weight, turn=c.turn)
        for c in common
        if c.key not in disabled and c.key not in own_keys
    ]
    return kept + own


def load_cases(path: str = DEFAULT_CASES_PATH, *, validate: bool = True) -> Suite:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "cases" not in raw:
        raise ConfigError(f"{path}: expected a mapping with a top-level `cases` list")

    defaults = raw.get("defaults") or {}
    common = _coerce_checks(defaults.get("common_checks"), "defaults.common_checks")
    default_pass_at = int(defaults.get("judge_pass_at", 2))

    seen_ids: set[str] = set()
    cases: list[Case] = []
    for raw_case in raw["cases"]:
        cid = raw_case.get("id")
        if not cid:
            raise ConfigError(f"{path}: a case is missing `id`")
        if cid in seen_ids:
            raise ConfigError(f"{path}: duplicate case id {cid!r}")
        seen_ids.add(cid)
        where = f"{path}:{cid}"

        turns = [Turn(text=t["text"].strip()) for t in raw_case.get("turns") or []]
        if not turns:
            raise ConfigError(f"{where}: needs at least one turn")

        # Personalization merges over the defaults rather than replacing them,
        # so a case can drop just the response language (to test that the model
        # follows the query's language) without also losing the pinned datetime
        # that keeps "tomorrow" reproducible. An explicit `language: null` in a
        # case is meaningful — it removes the key.
        personalization = {
            **(defaults.get("personalization") or {}),
            **(raw_case.get("personalization") or {}),
        }
        personalization = {k: v for k, v in personalization.items() if v is not None}

        expect_lang = raw_case.get("expect_lang") or _implied_lang(
            personalization.get("language"), raw_case.get("lang", defaults.get("lang", "zh"))
        )

        own = _coerce_checks(raw_case.get("checks"), where)
        disabled = set(raw_case.get("disable") or [])
        unknown_disable = disabled - {c.key for c in common}
        if unknown_disable:
            raise ConfigError(
                f"{where}: `disable` lists {sorted(unknown_disable)}, which are not "
                f"in defaults.common_checks — nothing would be disabled"
            )

        cases.append(
            Case(
                id=cid,
                suite=raw_case.get("suite") or cid.split("/")[0],
                title=raw_case.get("title") or cid,
                skill=raw_case.get("skill"),
                lang=raw_case.get("lang", defaults.get("lang", "zh")),
                turns=turns,
                checks=_merge_common_checks(common, own, disabled),
                judge=_coerce_judge(raw_case.get("judge"), where, default_pass_at),
                is_negative=bool(raw_case.get("is_negative", False)),
                weight=float(raw_case.get("weight", 1)),
                timeout_s=int(raw_case.get("timeout_s", defaults.get("timeout_s", 300))),
                repeats=int(raw_case.get("repeats", defaults.get("repeats", 2))),
                citation_grounding=bool(raw_case.get("citation_grounding", False)),
                fixture=raw_case.get("fixture"),
                personalization=personalization,
                expect_lang=expect_lang,
                rubric_version=int(raw.get("version", 1)),
            )
        )

    for case in cases:
        _bind_expected_language(case)

    suite = Suite(cases=cases, version=int(raw.get("version", 1)), defaults=defaults)
    if validate:
        validate_suite(suite)
    return suite


def validate_suite(suite: Suite) -> None:
    """Resolve every check key and arg against the registry.

    Imported lazily: `checks` imports `parsers`, which is cheap, but keeping
    the dependency one-directional means `config` stays importable from
    anywhere (including `checks` itself, for the arg spec).
    """
    from evals.checks import CHECK_REGISTRY

    problems: list[str] = []
    for case in suite.cases:
        n_turns = len(case.turns)
        for spec in case.checks:
            entry = CHECK_REGISTRY.get(spec.key)
            if entry is None:
                near = _did_you_mean(spec.key, CHECK_REGISTRY)
                problems.append(f"{case.id}: unknown check {spec.key!r}{near}")
                continue
            unknown_args = set(spec.args) - set(entry.arg_names)
            if unknown_args:
                problems.append(
                    f"{case.id}: check {spec.key!r} got unknown args "
                    f"{sorted(unknown_args)}; accepts {sorted(entry.arg_names)}"
                )
            missing = [a for a in entry.required_args if a not in spec.args]
            if missing:
                problems.append(
                    f"{case.id}: check {spec.key!r} is missing required args {missing}"
                )
            if isinstance(spec.turn, int) and not (0 <= spec.turn < n_turns):
                problems.append(
                    f"{case.id}: check {spec.key!r} targets turn {spec.turn} but the "
                    f"case only has {n_turns} turn(s)"
                )
        judge_keys = [j.key for j in case.judge]
        dupes = {k for k in judge_keys if judge_keys.count(k) > 1}
        if dupes:
            problems.append(f"{case.id}: duplicate judge keys {sorted(dupes)}")

    if problems:
        raise ConfigError(
            "cases.yaml failed validation:\n  - " + "\n  - ".join(problems)
        )


# Maps a `<personalization>` "Response language" value onto the code the
# language check compares against.
_LANG_CODES = {
    "zh": "zh", "简体中文": "zh", "繁體中文": "zh", "中文": "zh",
    "chinese": "zh", "simplified chinese": "zh",
    "en": "en", "english": "en",
}


def _implied_lang(personalization_language: str | None, case_lang: str) -> str | None:
    """Which language the answer is expected to be in.

    Personalization wins when it states one — PRO_PROMPT tells the model to
    honour it silently, so that is the behaviour under test. With nothing
    stated, the expectation falls back to the case's own `lang`, which is the
    language the *query* is written in: no instruction means follow the user.
    """
    stated = (personalization_language or "").strip().lower()
    if stated:
        return _LANG_CODES.get(stated)
    return _LANG_CODES.get((case_lang or "").strip().lower())


def _bind_expected_language(case: Case) -> None:
    """Fill `expect` on any response_language check that didn't specify one.

    Lets the shared `common_checks` entry stay a bare `{key: response_language}`
    while still comparing against each case's own expectation.
    """
    for spec in case.checks:
        if spec.key == "response_language" and "expect" not in spec.args:
            if case.expect_lang:
                spec.args = {**spec.args, "expect": case.expect_lang}


def _did_you_mean(key: str, registry: dict) -> str:
    import difflib

    close = difflib.get_close_matches(key, list(registry), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""
