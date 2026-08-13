"""Load and validate `queries.yaml`.

Same philosophy as `evals/config.py`: fail loudly at load time rather than
silently collect the wrong data. A misspelled check key here would mean a
category whose acceptance criteria never run — every trajectory "passes", the
bad ones get trained on, and nothing anywhere says so.

Run it directly to see the roll-up:

    python finetune/pro_agent/spec.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.utils.data_model import Personalization  # noqa: E402

HERE = Path(__file__).resolve().parent
QUERIES = HERE / "queries.yaml"
DOCS = HERE / "docs"


def doc_path(q: "Query") -> Path:
    """The synthetic attachment for a query, by convention `docs/<id>.md`.

    Convention rather than an explicit `doc:` field so the two can never drift
    apart — there is no second place to forget to update.
    """
    return DOCS / f"{q.id}.md"

# Blocks the production user message can carry (core/stream.py). The three the
# eval harness already emits are excluded — they need no deliberate coverage.
OPTIONAL_BLOCKS = {
    "user_memory": 12,
    "attached_files": 8,
    "priority_sources": 6,
    "follow_up_selection": 6,
    "requested_skill": 6,
}

DEFAULT_RUN_LIMIT = 30

# What production puts in `Response Language:` when the user has set no
# preference — an instruction, not a language name. Read off the model rather
# than retyped: `core/utils/utils.py::format_personalization` renders whatever
# this default holds, so a copy here would silently drift the day it changes.
#
# It appeared in 0 of the first 147 trajectories (every one pinned 简体中文 or
# English), while being the string most production turns actually carry.
FOLLOW_QUERY: str = Personalization.model_fields["response_language"].default


@dataclass
class Category:
    name: str
    n: int
    why: str
    checks: list[dict]
    run_limit: int = DEFAULT_RUN_LIMIT


@dataclass
class Query:
    id: str
    cat: str
    lang: str
    text: str
    block: str | None = None
    personalization: dict[str, str] = field(default_factory=dict)
    # Payloads for the two blocks whose content cannot be derived. A
    # `<follow_up_selection>` is a passage from the *previous* answer, and
    # `<priority_sources>` is a URL the user pinned — neither follows from the
    # query, so both are authored per query and validated below.
    follow_up: str = ""
    source_url: list[str] = field(default_factory=list)


@dataclass
class Spec:
    gate: list[str]
    categories: dict[str, Category]
    queries: list[Query]
    personalization_pool: list[dict]

    def checks_for(self, q: Query) -> list[dict]:
        """Gate + category checks for one query, as CheckSpec-shaped dicts.

        `response_language` needs its `expect` filled in here. Handed empty
        args it returns "no expected language declared" — an unconditional
        pass, so it would sit in the gate looking like a check while grading
        nothing. A personalization that names a language wins over the query's
        own, per SYSTEM_PROMPT, so that is normally the expectation.

        `FOLLOW_QUERY` is the exception and must be special-cased: it is not a
        language, it is an instruction to use the query's. Falling through to
        the `"中文" in …` test would score it as English and fail every Chinese
        query in the follow-query group — the exact group added to cover
        production's default.
        """
        stated = self.personalization_for(q).get("language") or ""
        if stated.strip() == FOLLOW_QUERY:
            expect = q.lang
        else:
            expect = "zh" if "中文" in stated else "en"
        out = []
        for key in self.gate:
            args = {"expect": expect} if key == "response_language" else {}
            out.append({"key": key, "args": args})
        out += [dict(c) for c in self.categories[q.cat].checks]
        return out

    def run_limit_for(self, q: Query) -> int:
        return self.categories[q.cat].run_limit

    # Queries that deliberately state a response language *different* from the
    # one they are written in. SYSTEM_PROMPT says a stated language wins, and the
    # base model is measurably bad at it (`language/personalization-overrides-
    # query`, 0.758), so the data has to contain the behaviour.
    #
    # Kept to a deliberate handful rather than left to chance. Rotating a
    # 3-zh/3-en pool against queries in file order produced a mismatch on 57 of
    # 130 — 44% of the set riding on a rule the *teacher* is itself unreliable
    # at: gpt-5.6-luna answered be-01 in Chinese on all three samples despite
    # an English personalization. What the teacher cannot do reliably cannot be
    # distilled, so the rest now agree and only these carry the override.
    # Deliberately excludes `write-rewrite`. Those answers carry a *deliverable*
    # whose language is set by its recipient — an apology email to an
    # English-speaking customer stays English however the reply is framed — so
    # a language override there produces a genuinely ambiguous target rather
    # than a testable one. wr-10 came back 39 CJK / 60 latin and scored
    # "mixed", which is the check being right about a badly-posed query.
    OVERRIDE_LANGUAGE = {
        "dr-06", "dr-12", "be-11", "sf-05", "st-06", "ch-02",
        "tc-04", "rs-08", "pl-02", "cp-06", "aq-04", "ab-03",
    }

    def personalization_for(self, q: Query) -> dict:
        if q.personalization:
            return q.personalization
        want = q.lang
        if q.id in self.OVERRIDE_LANGUAGE:
            want = "en" if q.lang == "zh" else "zh"
        pool = [
            p for p in self.personalization_pool
            if ("zh" if "中文" in (p.get("language") or "") else "en") == want
        ] or self.personalization_pool
        return pool[self.queries.index(q) % len(pool)]


def _normalize(text: str) -> str:
    """Fold a query for duplicate detection.

    NFKC first: the YAML mixes full-width and half-width punctuation, and
    `,` vs `,` would otherwise make two identical queries look distinct.
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"[^\w\s]", "", t).strip()


def _benchmark_queries() -> set[str]:
    import dotenv

    dotenv.load_dotenv(ROOT / ".env")
    from evals.config import load_cases

    return {
        _normalize(turn.text)
        for case in load_cases().cases
        for turn in case.turns
    }


def load() -> Spec:
    raw = yaml.safe_load(QUERIES.read_text())
    from evals.checks import CHECK_REGISTRY

    problems: list[str] = []

    gate = list(raw.get("gate") or [])
    for key in gate:
        if key not in CHECK_REGISTRY:
            problems.append(f"gate: unknown check {key!r}")

    cats: dict[str, Category] = {}
    for name, body in (raw.get("categories") or {}).items():
        checks = body.get("checks") or []
        for c in checks:
            key = c.get("key")
            spec = CHECK_REGISTRY.get(key)
            if spec is None:
                problems.append(f"{name}: unknown check {key!r}")
                continue
            unknown = set(c.get("args") or {}) - set(spec.arg_names)
            if unknown:
                problems.append(
                    f"{name}.{key}: unknown arg(s) {sorted(unknown)}; "
                    f"allowed {sorted(spec.arg_names)}"
                )
            missing = set(spec.required_args) - set(c.get("args") or {})
            if missing:
                problems.append(f"{name}.{key}: missing required arg(s) {sorted(missing)}")
        cats[name] = Category(
            name=name,
            n=int(body.get("n", 0)),
            why=(body.get("why") or "").strip(),
            checks=checks,
            run_limit=int(body.get("run_limit", DEFAULT_RUN_LIMIT)),
        )

    queries: list[Query] = []
    seen_ids: set[str] = set()
    for q in raw.get("queries") or []:
        qid = q.get("id", "?")
        if qid in seen_ids:
            problems.append(f"duplicate query id {qid!r}")
        seen_ids.add(qid)
        if q.get("cat") not in cats:
            problems.append(f"{qid}: unknown category {q.get('cat')!r}")
        if q.get("lang") not in ("zh", "en"):
            problems.append(f"{qid}: lang must be zh or en, got {q.get('lang')!r}")
        blk = q.get("block")
        if blk is not None and blk not in OPTIONAL_BLOCKS:
            problems.append(f"{qid}: unknown block {blk!r}")
        queries.append(
            Query(id=qid, cat=q.get("cat"), lang=q.get("lang"),
                  text=(q.get("text") or "").strip(), block=blk,
                  personalization=q.get("personalization") or {},
                  follow_up=(q.get("follow_up") or "").strip(),
                  source_url=list(q.get("source_url") or []))
        )

    # Declared counts must match reality, or the mix silently drifts.
    actual = Counter(q.cat for q in queries)
    for name, cat in cats.items():
        if actual[name] != cat.n:
            problems.append(f"{name}: declares n={cat.n} but has {actual[name]} queries")

    # Zero overlap with the test set. Enforced here, never by eye — an overlap
    # turns the benchmark from a measurement into a memorisation check.
    bench = _benchmark_queries()
    for q in queries:
        if _normalize(q.text) in bench:
            problems.append(f"{q.id}: query duplicates an evals/cases.yaml test query")

    # Self-duplicates.
    norm = Counter(_normalize(q.text) for q in queries)
    for q in queries:
        if norm[_normalize(q.text)] > 1:
            problems.append(f"{q.id}: duplicated within queries.yaml")

    # Non-CJK script leaking into a zh query is almost always a typo.
    for q in queries:
        if q.lang == "zh" and re.search(r"[Ѐ-ӿ؀-ۿ฀-๿]", q.text):
            problems.append(f"{q.id}: unexpected non-CJK script in a zh query")

    # A block that forces a tool call cannot ride on a category that forbids
    # them. `<requested_skill>` means "load it before anything else" and
    # `<attached_files>` mounts a document — both are honoured with `read_file`,
    # so pairing either with `no_tool_calls` makes the category check
    # unsatisfiable and would silently reject every candidate for that query.
    TOOL_FORCING = {"requested_skill", "attached_files"}
    for q in queries:
        keys = {c["key"] for c in cats[q.cat].checks} if q.cat in cats else set()
        if "no_tool_calls" in keys and q.block in TOOL_FORCING:
            problems.append(
                f"{q.id}: block {q.block!r} forces a read_file, but category "
                f"{q.cat!r} requires no_tool_calls — unsatisfiable"
            )

    # Every `attached_files` query needs its synthetic document on disk. Missing
    # one would silently collect a trace with an empty `<attached_files>` note,
    # teaching the model that the block means nothing.
    for q in queries:
        if q.block == "attached_files" and not doc_path(q).exists():
            problems.append(f"{q.id}: block is attached_files but {doc_path(q).name} is missing")
        if q.block == "follow_up_selection" and not q.follow_up:
            problems.append(f"{q.id}: block is follow_up_selection but follow_up: is empty")
        if q.block == "priority_sources" and not q.source_url:
            problems.append(f"{q.id}: block is priority_sources but source_url: is empty")
        if q.follow_up and q.block != "follow_up_selection":
            problems.append(f"{q.id}: has follow_up: but block is {q.block!r}")
        if q.source_url and q.block != "priority_sources":
            problems.append(f"{q.id}: has source_url: but block is {q.block!r}")

    if problems:
        raise SystemExit(
            "queries.yaml is invalid:\n" + "\n".join(f"  - {p}" for p in sorted(set(problems)))
        )

    return Spec(gate=gate, categories=cats, queries=queries,
                personalization_pool=raw.get("personalization_pool") or [])


def report(spec: Spec) -> None:
    n = len(spec.queries)
    print(f"{n} queries, {len(spec.categories)} categories, gate of {len(spec.gate)} checks\n")

    print(f"{'category':<20}{'n':>4}{'run_limit':>11}  extra checks")
    for name, cat in spec.categories.items():
        keys = ", ".join(c["key"] for c in cat.checks)
        print(f"{name:<20}{cat.n:>4}{cat.run_limit:>11}  {keys}")

    langs = Counter(q.lang for q in spec.queries)
    print(f"\nlanguage:  zh {langs['zh']} ({langs['zh']/n:.0%})   en {langs['en']} ({langs['en']/n:.0%})")

    blocks = Counter(q.block for q in spec.queries if q.block)
    print("\noptional-block coverage (production emits 7, eval harness only 3):")
    short = False
    for blk, want in OPTIONAL_BLOCKS.items():
        got = blocks.get(blk, 0)
        flag = "" if got >= want else f"   ← short by {want - got}"
        short |= got < want
        print(f"  {blk:<22}{got:>3} / {want}{flag}")
    if short:
        print("\n! block coverage below target — those blocks stay unseen in training")

    print(f"\nno overlap with the {len(_benchmark_queries())} benchmark queries: OK")


if __name__ == "__main__":
    report(load())
