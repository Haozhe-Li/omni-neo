"""Roll out the teacher, filter by the rubric, keep the tightest survivor.

    python finetune/pro_agent/collect.py --samples 3
    python finetune/pro_agent/collect.py --only dr-01 tc-02 --samples 1   # smoke

Writes `dataset/traces.jsonl` (one record per accepted query) and
`dataset/rejected.jsonl` (everything that failed, with the reason). Both are
gitignored — `finetune/**/dataset` — and neither is reproducible, so keep a
copy outside the repo.

## Why this does not reuse `evals/runner.py`

It nearly does, and the one difference is the point. `evals/agent_factory.py::
build_user_message` emits three of the seven blocks `core/stream.py::
build_message_content` can produce, because no eval case has an attachment.
That is a fair test harness and a bad *training* harness: the four blocks it
omits would then never appear in the data, and the model would meet
`<attached_files>` for the first time in production.

So collection drives the production builder directly, with `get_file_record`
patched to serve the synthetic documents in `docs/`. Everything else — tag
order, `_tagged_block` spacing, document mounting, citation registration —
is production's own code, not a copy of it.

## Selection

`--samples N` rollouts per query. Among those that clear the gate *and* their
category's checks, the one with the **fewest tool calls** wins. The student's
failure is two-sided — it either calls nothing or burns the whole budget — and
picking the tightest passing trace is how the data argues for restraint rather
than merely permitting it.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

from deepagents import create_deep_agent  # noqa: E402
from langchain.agents.middleware import (  # noqa: E402
    AgentMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import core.context_enrichment as enrich_mod  # noqa: E402
import core.stream as stream_mod  # noqa: E402
from core.agent import (  # noqa: E402
    SYSTEM_PROMPT,
    SKILL_FILES,
    AGENT_TOOLS,
    SKILLS_SOURCE,
    _register_harness_profiles,
)
from core.tools.adapters import _UNAVAILABLE  # noqa: E402
from core.utils.citations import all_citations, reset_citation_registry  # noqa: E402
from core.context_enrichment import Enrichment, enrich_context  # noqa: E402
from core.utils.data_model import Personalization  # noqa: E402
from core.utils.utils import format_system_reminder  # noqa: E402
from evals import checks as checks_mod  # noqa: E402
from evals.config import CheckSpec  # noqa: E402
from evals.runner import _normalize_stream_item, _text_of  # noqa: E402
from evals.toolcache import DEFAULT_CACHE_DIR, ToolCache, wrap_tools  # noqa: E402
from evals.trace import RunTrace, ToolCallRecord, TurnTrace  # noqa: E402

from spec import (  # noqa: E402
    FOLLOW_QUERY,
    SOURCE_URL_BLOCKS,
    Query,
    Spec,
    doc_path,
    load,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"

# Read off the adapter rather than retyped, so a reworded sentinel cannot
# silently stop being detected here.
_SEARCH_UNAVAILABLE = _UNAVAILABLE[0]["title"]

# Which skill a `<requested_skill>` block should name, by category. Derived
# rather than authored per query: the mapping is what the category already
# means, and a second place to state it is a second place to get it wrong.
REQUESTED_SKILL_BY_CAT = {
    "deep-research": "web-research",
    "budget-exhausted": "web-research",
    "chart": "charting",
    "teach": "guided-learning",
    "places": "mapping",
    "search-fact": "web-research",
}

# Production memory is about the *user*, not the query, so a rotating pool is
# realistic rather than a shortcut. Kept short: real memory blocks are terse.
USER_MEMORY_POOL = [
    "Works as a backend engineer at a mid-size fintech. Prefers concrete "
    "numbers over qualitative summaries. Has a toddler, so weekends are tight.",
    "在读研究生,方向是材料science。习惯先看结论再看论证。对图表要求比较高。",
    "Runs a two-person consulting shop. Travels most weeks. Vegetarian. "
    "Reads on a phone more often than a laptop.",
    "产品经理,负责一个 to B 的 SaaS。不喜欢太长的回答,喜欢先给判断再给理由。",
    "Former academic, now works in policy. Cares a lot about whether a claim "
    "is actually sourced. Based in the UK.",
    "自由职业设计师,住在成都。英文阅读没问题但更习惯中文回复。",
]


# ── Context variants ────────────────────────────────────────────────────────
# One query, several *envelopes*. Production wraps the same question in wildly
# varying context — a memory block or none, a pinned response language or the
# follow-the-query default, a location and a clock that are never the same
# twice — and the adapter has to be invariant to all of it. One fixed envelope
# per query teaches the opposite: every one of the 147 traces before this
# carried a pinned language and a datetime inside a two-week window, so
# "2026-08" and "简体中文" were as much a part of each example as the task was.
#
# Variant 0 is always the query's *authored* envelope, so each query keeps one
# canonical row comparable with earlier rounds and the randomisation only ever
# adds. Variants 1+ are seeded on (seed, query id) so a re-run reproduces them
# exactly — a collection you cannot reproduce is one you cannot debug.
#
# What is randomised and what is not:
#
#   randomised   personalization (including "absent entirely"), user_memory
#   authored     source_url, follow_up_selection, attached_files, requested_skill
#
# The split is not arbitrary. Memory and personalization are *about the user*
# and could accompany any question, so varying them varies nothing about the
# task. The other four are *about this query* — "read this page and summarise
# it" without its URL is a different request — so dropping them at random
# would quietly change what the row teaches.

# How often a randomised variant drops personalization entirely. Not a rare
# edge: a client that sends no `personalization` object at all still gets a
# `<system_reminder>` (the identity line survives — see
# core/utils/utils.py::format_system_reminder), and the model has never once
# been trained on that shape.
_P_NO_PERSONALIZATION = 0.15
# How often a pool entry's pinned language is replaced by production's default,
# `Follow User's Query Language`. The pool pins a language; production mostly
# does not.
_P_FOLLOW_QUERY = 0.40
# How often a randomised variant carries a memory block, when the query does
# not already demand one.
_P_USER_MEMORY = 0.35


@dataclass
class Variant:
    """One randomised context envelope for a query."""

    index: int
    # None means the client sent no personalization object at all — a real
    # production shape, and distinct from "sent one with empty fields".
    personalization: dict[str, str] | None
    user_memory: str = ""
    skill: str | None = None
    follow_up: str = ""
    source_url: list[str] = field(default_factory=list)
    attach_doc: bool = False

    @property
    def blocks(self) -> list[str]:
        """Which optional blocks this envelope will actually produce."""
        out = []
        if self.user_memory:
            out.append("user_memory")
        if self.personalization is not None:
            out.append("system_reminder")
        if self.attach_doc:
            out.append("attached_files")
        if self.skill:
            out.append("requested_skill")
        if self.follow_up:
            out.append("follow_up_selection")
        if self.source_url:
            out.append("source_url")
        return out


def _authored_variant(spec: Spec, q: Query) -> Variant:
    """Variant 0 — exactly what the query file asks for, nothing random."""
    return Variant(
        index=0,
        personalization=spec.personalization_for(q),
        user_memory=(
            USER_MEMORY_POOL[int(q.id.split("-")[1]) % len(USER_MEMORY_POOL)]
            if q.block == "user_memory" else ""
        ),
        skill=REQUESTED_SKILL_BY_CAT.get(q.cat, "web-research") if q.block == "requested_skill" else None,
        follow_up=q.follow_up if q.block == "follow_up_selection" else "",
        source_url=list(q.source_url or []) if q.block in SOURCE_URL_BLOCKS else [],
        attach_doc=q.block == "attached_files",
    )


def _wanted_language(spec: Spec, q: Query) -> str:
    """The language this query's personalization should state.

    Mirrors `Spec.personalization_for`: the query's own language, flipped for
    the queries that deliberately carry an override.
    """
    if q.id in spec.OVERRIDE_LANGUAGE:
        return "en" if q.lang == "zh" else "zh"
    return q.lang


def _random_variant(spec: Spec, q: Query, index: int, rng: random.Random) -> Variant:
    base = _authored_variant(spec, q)

    if rng.random() < _P_NO_PERSONALIZATION:
        personalization = None
    else:
        # Draw from the same language-matched slice `Spec.personalization_for`
        # uses, and only randomise *within* it. Drawing from the whole pool
        # instead re-creates a mismatch the spec removed on purpose: a Chinese
        # query under an English `Response Language:` is a rule the teacher
        # itself is unreliable at (spec.py::OVERRIDE_LANGUAGE documents luna
        # answering in Chinese on all three samples despite an English
        # personalization), so those rows teach "ignore the stated language"
        # rather than the rule. Measured on the first 42 rows of this run,
        # before the fix: 8 of them, 19%.
        #
        # The 12 designated override queries are the exception and keep their
        # deliberate mismatch — `personalization_for` already flips them.
        pool = spec.personalization_pool
        want = _wanted_language(spec, q)
        matched = [
            p for p in pool
            if ("zh" if "中文" in (p.get("language") or "") else "en") == want
        ] or pool
        picked = dict(rng.choice(matched))
        # Production's default is an instruction ("follow the query"), not a
        # language, so substituting it never creates a mismatch — but it would
        # erase an override query's whole point.
        if rng.random() < _P_FOLLOW_QUERY and q.id not in spec.OVERRIDE_LANGUAGE:
            picked["language"] = FOLLOW_QUERY
        personalization = picked

    memory = base.user_memory
    if not memory and rng.random() < _P_USER_MEMORY:
        memory = rng.choice(USER_MEMORY_POOL)

    return Variant(
        index=index,
        personalization=personalization,
        user_memory=memory,
        skill=base.skill,
        follow_up=base.follow_up,
        source_url=base.source_url,
        attach_doc=base.attach_doc,
    )


def variants_for(spec: Spec, q: Query, n: int, seed: int = 0) -> list[Variant]:
    # Seeded from a hash digest, not from the string itself. `random.Random(str)`
    # leaves the *first* draw correlated across similar short seeds: over ten
    # query ids, `random.Random(f"0:{qid}").random()` came back under 0.15 five
    # times, so variant 1 would have dropped personalization on half the set
    # instead of the configured 15%. blake2b avalanches, so neighbouring ids
    # get uncorrelated streams.
    digest = hashlib.blake2b(f"{seed}:{q.id}".encode(), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    out = [_authored_variant(spec, q)]
    while len(out) < n:
        out.append(_random_variant(spec, q, len(out), rng))
    return out[:n]


@dataclass
class Rollout:
    sample: int
    ok: bool
    reason: str = ""
    n_tool_calls: int = 0
    latency_ms: int = 0
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    blocked_calls: int = 0
    variant: "Variant | None" = None
    enrichment: "Enrichment | None" = None


class _CaptureSystem(AgentMiddleware):
    """Record the assembled system prompt, then let the call proceed.

    The adapter is trained on this exact string, so it is stored per rollout
    rather than assumed constant — if it ever differs from
    `fingerprint.json`, that is something the data should show.
    """

    def __init__(self) -> None:
        super().__init__()
        self.system = ""

    def _grab(self, request) -> None:
        if self.system:
            return
        text = getattr(request, "system_prompt", None)
        if not text:
            msgs = list(getattr(request, "messages", []) or [])
            if msgs and getattr(msgs[0], "type", "") == "system":
                text = msgs[0].content
        self.system = text or ""

    def wrap_model_call(self, request, handler):
        self._grab(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._grab(request)
        return await handler(request)


def _patch_file_records(q: Query) -> None:
    """Serve `docs/<id>.md` through the production attachment path.

    `build_message_content` resolves attachments via `get_file_record`, mounts
    the text, and calls `register_document_citation` to assign the `[n]` the
    model is allowed to cite. Patching at that seam keeps every one of those
    steps production's, so `citation_exists` means the same thing here as it
    does in the benchmark.
    """
    text = doc_path(q).read_text()

    def fake_record(file_id: str):
        return {
            "file_id": file_id,
            "category": "document",
            "status": "ready",
            "extracted_text": text,
            "created_at": "2026-01-01T00:00:00Z",
        }

    stream_mod.get_file_record = fake_record


def _cache_scout_tools(cache: ToolCache) -> None:
    """Point the scout's tools at the same disk cache the agent's tools use.

    `core/context_enrichment.py` imports the adapters directly, so its calls go
    around `wrap_tools` and would hit the live providers on every rollout —
    real money per row, and a re-run of the same seed would come back with
    different search results, destroying the reproducibility the cache exists
    to provide. The module looks these names up at call time, so rebinding them
    on the module is enough.

    The wrapper also replays the citations a cached call originally registered
    (`ToolCache.save` snapshots them), which is what keeps `[n]` numbering
    identical between a cold rollout and a replayed one.
    """
    if not cache.enabled:
        return
    wrapped = {
        getattr(t, "__name__", None) or getattr(t, "name", ""): t
        for t in wrap_tools(AGENT_TOOLS, cache)
    }
    for name in ("web_search", "weather_forecast", "stock_search", "currency_convert"):
        if name in wrapped:
            setattr(enrich_mod, name, wrapped[name])


async def build_message(
    spec: Spec, q: Query, v: Variant
) -> tuple[Any, dict, list[dict], Enrichment]:
    """Production's user message for one query under one context envelope.

    Everything here is production's own code, not a copy of it — the
    `<system_reminder>` body goes through `format_system_reminder`, the blocks
    through `build_message_content`, and the pre-flight scout through
    `enrich_context`. That matters more than it sounds: the training row and
    the production request have to be the same string, and every past drift
    between them (the `Response language:` / `Response Language:` capital that
    confounded v4's whole language story) came from a hand-rolled copy of a
    thing production already did.

    The scout runs for real. It is not simulated and its output is not
    synthesised: it calls the same `enrich_context` production calls, which
    issues a live search or lookup, registers the resulting citations in this
    rollout's registry, and hands back the `<context_enrichment>` body. The
    alternative — collecting without it — would train an adapter that meets
    that block for the first time in production, which is the exact failure
    the `<attached_files>` coverage exists to avoid.

    Precedence matches production: a pinned URL wins, and the scout is skipped
    entirely for that turn (core/stream.py::_stream_agent). Every collected row
    is a first turn, which is the only turn the scout runs on at all.
    """
    p = v.personalization
    system_reminder = format_system_reminder(
        Personalization(
            response_language=(p.get("language") or "") if p else "",
            user_location=(p.get("location") or "Unknown") if p else None,
            user_local_datetime=p["datetime"] if p else None,
        )
        if p is not None
        else None
    )

    kwargs: dict[str, Any] = {}
    attached: list[dict[str, str]] | None = None
    if v.user_memory:
        kwargs["user_memory"] = v.user_memory
    if v.skill:
        kwargs["skill"] = v.skill
    if v.follow_up:
        kwargs["follow_up_content"] = v.follow_up
    if v.source_url:
        kwargs["source_url"] = list(v.source_url)
    if v.attach_doc:
        _patch_file_records(q)
        attached = [{f"synthetic-{q.id}": f"{q.id}.md"}]

    enrichment = Enrichment()
    if not v.source_url:
        enrichment = await enrich_context(
            q.text,
            user_location=(p or {}).get("location"),
            user_local_datetime=(p or {}).get("datetime"),
        )

    content, files, doc_sources = stream_mod.build_message_content(
        q.text, system_reminder, attached, thread_id=f"collect-{q.id}",
        enrichment_text=enrichment.text, **kwargs,
    )
    return content, files, doc_sources, enrichment


async def rollout(spec: Spec, q: Query, v: Variant, cache: ToolCache) -> Rollout:
    from core.llm import gpt_5_6_luna

    _register_harness_profiles()
    cap = _CaptureSystem()
    agent = create_deep_agent(
        name="omni-eval-pro",
        model=gpt_5_6_luna,
        tools=wrap_tools(AGENT_TOOLS, cache),
        system_prompt=SYSTEM_PROMPT,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        middleware=[
            cap,
            ToolRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
            ToolCallLimitMiddleware(run_limit=spec.run_limit_for(q)),
        ],
    )

    # Before build_message, not after: the scout inside it registers the
    # citations its search produced, and those have to land in this rollout's
    # registry so the agent's own [n] numbering continues from them exactly as
    # it does in production.
    reset_citation_registry(f"collect-{q.id}-{v.index}", 1)
    content, doc_files, _doc_sources, enrichment = await build_message(spec, q, v)
    files = {**SKILL_FILES, **doc_files}
    state = {"messages": [{"role": "user", "content": content}], "files": files}
    cfg = {"configurable": {"thread_id": f"collect-{q.id}-{v.index}-{int(time.time()*1000)}"}}

    turn = TurnTrace(index=0, query=q.text)
    raw_messages: list[dict] = [{"role": "user", "content": content}]
    by_id: dict[str, ToolCallRecord] = {}
    blocked = 0
    t0 = time.perf_counter()

    async for item in agent.astream(
        state, config=cfg, stream_mode=["updates"], subgraphs=True
    ):
        mode, data = _normalize_stream_item(item)
        if mode != "updates":
            continue
        for node in (data or {}).values():
            if not isinstance(node, dict):
                continue
            for m in node.get("messages") or []:
                if isinstance(m, AIMessage):
                    turn.n_llm_turns += 1
                    turn.usage.add_usage(getattr(m, "usage_metadata", None))
                    body = _text_of(m.content).strip()
                    msg: dict[str, Any] = {"role": "assistant", "content": body}
                    if m.tool_calls:
                        if body:
                            turn.mixed_messages.append(body[:300])
                        msg["tool_calls"] = [
                            {
                                "id": c.get("id"),
                                "type": "function",
                                "function": {
                                    "name": c.get("name"),
                                    "arguments": json.dumps(
                                        c.get("args") or {}, ensure_ascii=False
                                    ),
                                },
                            }
                            for c in m.tool_calls
                        ]
                        for c in m.tool_calls:
                            rec = ToolCallRecord(
                                index=len(turn.tool_calls) + 1,
                                name=c.get("name", "?"),
                                args=c.get("args") or {},
                            )
                            turn.tool_calls.append(rec)
                            if c.get("id"):
                                by_id[c["id"]] = rec
                    elif body:
                        turn.text = body
                    raw_messages.append(msg)
                elif isinstance(m, ToolMessage):
                    body = _text_of(m.content)
                    rec = by_id.get(m.tool_call_id)
                    if rec is not None:
                        rec.result_full = body
                        rec.result_head = body[:600]
                    if "limit exceeded" in body.lower():
                        blocked += 1
                        turn.hit_run_limit = True
                    raw_messages.append(
                        {"role": "tool", "tool_call_id": m.tool_call_id, "content": body}
                    )

    turn.latency_ms = int((time.perf_counter() - t0) * 1000)

    # A turn where the search backend was down is not a training example. The
    # sentinel `web_search` returns in that case ("Search unavailable …", see
    # core/tools/adapters.py) tells the agent to answer unverified and say so,
    # and it does — producing a fluent, plausible trajectory that teaches
    # exactly the habit the whole retrieval prompt exists to prevent. It has to
    # be caught here rather than left to the rubric, because the rubric is not
    # gating this round. SearXNG 4xx/5xx are retried inside the client first;
    # this only fires when all of those attempts failed.
    unavailable = sum(
        1 for m in raw_messages
        if m.get("role") == "tool" and _SEARCH_UNAVAILABLE in (m.get("content") or "")
    )
    trace = RunTrace(case_id=q.id, model_label="gpt-5-6-luna", status="ok")
    trace.turns.append(turn)
    trace.citations = [dict(c) for c in all_citations()]

    specs = [
        CheckSpec(key=c["key"], args=c.get("args") or {}, weight=1, turn="all")
        for c in spec.checks_for(q)
    ]
    # run_checks returns (CheckSpec, CheckResult) pairs — the label lives on the
    # spec, the verdict on the result.
    results = checks_mod.run_checks(trace, specs)
    failed = [
        f"{s.label}: {r.evidence or r.reason}" for s, r in results if not r.passed
    ]

    return Rollout(
        sample=v.index,
        # A rollout is usable if it produced an answer at all and its tools
        # actually worked. The rubric still runs — `failed` below is recorded
        # on every row — but it no longer decides acceptance: see the note on
        # `--gate` in main().
        ok=bool(turn.text) and not unavailable,
        reason=(
            "" if turn.text and not unavailable
            else "no final answer" if not turn.text
            else f"search backend unavailable on {unavailable} call(s)"
        ),
        n_tool_calls=len(turn.tool_calls),
        latency_ms=turn.latency_ms,
        system_prompt=cap.system,
        messages=raw_messages,
        failed_checks=failed,
        blocked_calls=blocked,
        variant=v,
        enrichment=enrichment,
    )


def trim_blocked_retries(messages: list[dict]) -> tuple[list[dict], int]:
    """Keep one budget-exceeded call, drop the rest.

    At `run_limit=6` the teacher retried eight times before writing its answer.
    Cloned verbatim that teaches "retry eight times, then finish"; the lesson
    wanted is "budget hit, write the answer". Assistant/tool pairs are removed
    together so every `tool_call_id` still resolves.
    """
    blocked_ids = {
        m["tool_call_id"]
        for m in messages
        if m.get("role") == "tool" and "limit exceeded" in (m.get("content") or "").lower()
    }
    if len(blocked_ids) <= 1:
        return messages, 0

    keep_id = next(
        m["tool_call_id"]
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") in blocked_ids
    )
    drop = blocked_ids - {keep_id}
    out, dropped = [], 0
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") in drop:
            dropped += 1
            continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            kept = [c for c in m["tool_calls"] if c["id"] not in drop]
            if not kept:
                continue
            m = {**m, "tool_calls": kept}
        out.append(m)
    return out, dropped


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2,
                    help="context envelopes per query; variant 0 is the authored one")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the randomised envelopes; same seed reproduces a run")
    ap.add_argument("--gate", action="store_true",
                    help="drop rows that fail the rubric instead of only recording it")
    ap.add_argument("--only", nargs="*", default=None, help="query ids")
    ap.add_argument("--cat", nargs="*", default=None, help="categories")
    ap.add_argument("--out", default=str(DATA))
    args = ap.parse_args()

    # Without this the `no_prompt_leak` gate check has nothing to compare
    # against. It fails loudly when unarmed rather than passing everything,
    # which is the only reason this was caught here rather than in the data.
    from core.agent import SYSTEM_PROMPTS
    from core.prompt_guard import register_sensitive_prompts

    register_sensitive_prompts(SYSTEM_PROMPTS)

    spec = load()
    queries = spec.queries
    if args.only:
        queries = [q for q in queries if q.id in set(args.only)]
    if args.cat:
        queries = [q for q in queries if q.cat in set(args.cat)]
    if not queries:
        raise SystemExit("no queries matched")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_f = (out_dir / "traces.jsonl").open("a", encoding="utf-8")
    rejected_f = (out_dir / "rejected.jsonl").open("a", encoding="utf-8")
    cache = ToolCache(DEFAULT_CACHE_DIR, enabled=True)
    _cache_scout_tools(cache)

    n_ok = 0
    n_gate_fail = 0
    total = len(queries) * args.variants
    i = 0
    for q in queries:
        for v in variants_for(spec, q, args.variants, args.seed):
            i += 1
            try:
                r = await rollout(spec, q, v, cache)
            except Exception as e:
                r = Rollout(sample=v.index, ok=False,
                            reason=f"{type(e).__name__}: {e}", variant=v)

            # Two different reasons to drop a rollout, and only one of them is
            # about quality. "No answer" means there is no training target at
            # all, so it is always dropped. A rubric failure is dropped only
            # under --gate; by default it is recorded on the row and left for
            # build_dataset.py to filter or not. The rubric was written against
            # an older prompt, and the contracts it grades (the ```text fence,
            # the draft-email skill, citations carrying `n`) changed under it —
            # so a failure here is currently as likely to be a stale check as a
            # bad trace, and throwing the trace away destroys the evidence
            # needed to tell which.
            gate_failed = bool(r.failed_checks)
            if not r.ok or (args.gate and gate_failed):
                print(f"[{i:>3}/{total}] {q.id:<8} v{v.index} {q.cat:<17} REJECTED  "
                      f"{(r.failed_checks or [r.reason])[:1]}")
                rejected_f.write(json.dumps({
                    "id": q.id, "cat": q.cat, "variant": v.index, "text": q.text,
                    "reason": r.reason, "failed": r.failed_checks,
                    "n_tool_calls": r.n_tool_calls,
                    # The answer, not just the verdict. A rejection is as often
                    # a wrong threshold as a bad trace, and the two are only
                    # distinguishable by reading what the teacher actually
                    # wrote — `word_count` on a `<report>` answer looks like a
                    # terse reply because prose_only strips the report first.
                    "final_head": next(
                        (m.get("content", "")[:4000] for m in reversed(r.messages)
                         if m.get("role") == "assistant" and m.get("content")), ""
                    ),
                }, ensure_ascii=False) + "\n")
                rejected_f.flush()
                continue

            messages, dropped = (
                trim_blocked_retries(r.messages)
                if q.cat == "budget-exhausted" else (r.messages, 0)
            )
            e = r.enrichment or Enrichment()
            accepted_f.write(json.dumps({
                "id": q.id, "cat": q.cat, "lang": q.lang,
                "variant": v.index,
                # The envelope this row was produced under. Recorded rather than
                # recomputed: `variants_for` is seeded and reproducible today,
                # but a row has to stay readable after the pools change.
                "context": {
                    "blocks": v.blocks,
                    "personalization": v.personalization,
                    "user_memory": v.user_memory,
                    "skill": v.skill,
                    "source_url": v.source_url,
                    "attached": v.attach_doc,
                },
                "enrichment": {
                    "action": e.action,
                    "tool": (e.events[0]["tool"] if e.events else None),
                    "n_sources": len(e.sources),
                },
                "text": q.text,
                "run_limit": spec.run_limit_for(q),
                "n_tool_calls": r.n_tool_calls,
                "blocked_calls": r.blocked_calls,
                "trimmed_blocked": dropped,
                # Recorded, not enforced — see the note above.
                "gate": {"passed": not gate_failed, "failed": r.failed_checks},
                "system_prompt": r.system_prompt,
                "messages": messages,
            }, ensure_ascii=False) + "\n")
            accepted_f.flush()
            n_ok += 1
            n_gate_fail += int(gate_failed)
            trim = f" trim-{dropped}" if dropped else ""
            gate = " gate-FAIL" if gate_failed else ""
            scout = f" scout={e.action}" if e.action != "direct_response" else ""
            print(f"[{i:>3}/{total}] {q.id:<8} v{v.index} {q.cat:<17} ok  "
                  f"tools={r.n_tool_calls:<3}{scout}{trim}{gate}")

    accepted_f.close()
    rejected_f.close()
    print(f"\n{n_ok}/{total} rows written to {out_dir/'traces.jsonl'}"
          f"  ({n_gate_fail} of them fail the rubric)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
