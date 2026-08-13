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
import json
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

import core.stream as stream_mod  # noqa: E402
from core.agent import (  # noqa: E402
    SYSTEM_PROMPT,
    SKILL_FILES,
    RETRIEVAL_TOOLS,
    SKILLS_SOURCE,
    _register_harness_profiles,
)
from core.utils.citations import all_citations, reset_citation_registry  # noqa: E402
from core.utils.data_model import Personalization  # noqa: E402
from core.utils.utils import format_personalization  # noqa: E402
from evals import checks as checks_mod  # noqa: E402
from evals.config import CheckSpec  # noqa: E402
from evals.runner import _normalize_stream_item, _text_of  # noqa: E402
from evals.toolcache import DEFAULT_CACHE_DIR, ToolCache, wrap_tools  # noqa: E402
from evals.trace import RunTrace, ToolCallRecord, TurnTrace  # noqa: E402

from spec import Query, Spec, doc_path, load  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"

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


def build_message(spec: Spec, q: Query) -> tuple[Any, dict, list[dict]]:
    """Production's user message, with this query's optional block attached.

    The `<personalization>` body goes through production's own
    `format_personalization` rather than being assembled here. It used to be
    hand-rolled, and the labels had drifted from production's — training and
    the benchmark both emitted `Response language:` / `User location:` while
    production emits `Response Language:` / `User Location:`. The adapter was
    keyed on a string production never sends, and because the benchmark shared
    the *training* spelling, no eval could see it.

    Trajectories collected before 2026-08-12 carry the old spelling. They are
    not rewritten: the mix is harmless and arguably useful, since production
    and the benchmark still disagree until `evals/agent_factory.py` is fixed
    too, and a model invariant to the label is what serves both.
    """
    p = spec.personalization_for(q)
    personalization = format_personalization(
        Personalization(
            response_language=p.get("language") or "",
            user_location=p.get("location") or "Unknown",
            user_local_datetime=p["datetime"],
        )
    )

    kwargs: dict[str, Any] = {}
    attached: list[dict[str, str]] | None = None
    if q.block == "user_memory":
        kwargs["user_memory"] = USER_MEMORY_POOL[
            int(q.id.split("-")[1]) % len(USER_MEMORY_POOL)
        ]
    elif q.block == "requested_skill":
        kwargs["skill"] = REQUESTED_SKILL_BY_CAT.get(q.cat, "web-research")
    elif q.block == "follow_up_selection":
        kwargs["follow_up_content"] = q.follow_up or ""
    elif q.block == "priority_sources":
        kwargs["source_url"] = list(q.source_url or [])
    elif q.block == "attached_files":
        _patch_file_records(q)
        attached = [{f"synthetic-{q.id}": f"{q.id}.md"}]

    return stream_mod.build_message_content(
        q.text, personalization, attached, thread_id=f"collect-{q.id}", **kwargs
    )


async def rollout(spec: Spec, q: Query, sample: int, cache: ToolCache) -> Rollout:
    from core.llm import gpt_5_6_luna

    _register_harness_profiles()
    cap = _CaptureSystem()
    agent = create_deep_agent(
        name="omni-eval-pro",
        model=gpt_5_6_luna,
        tools=wrap_tools(RETRIEVAL_TOOLS, cache),
        system_prompt=SYSTEM_PROMPT,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        middleware=[
            cap,
            ToolRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
            ToolCallLimitMiddleware(run_limit=spec.run_limit_for(q)),
        ],
    )

    reset_citation_registry(f"collect-{q.id}", 1)
    content, doc_files, _doc_sources = build_message(spec, q)
    files = {**SKILL_FILES, **doc_files}
    state = {"messages": [{"role": "user", "content": content}], "files": files}
    cfg = {"configurable": {"thread_id": f"collect-{q.id}-{sample}-{int(time.time()*1000)}"}}

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
        sample=sample,
        ok=not failed and bool(turn.text),
        reason="" if turn.text else "no final answer",
        n_tool_calls=len(turn.tool_calls),
        latency_ms=turn.latency_ms,
        system_prompt=cap.system,
        messages=raw_messages,
        failed_checks=failed,
        blocked_calls=blocked,
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
    ap.add_argument("--samples", type=int, default=3)
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

    n_ok = 0
    for i, q in enumerate(queries, 1):
        rolls: list[Rollout] = []
        for s in range(args.samples):
            try:
                rolls.append(await rollout(spec, q, s, cache))
            except Exception as e:
                rolls.append(Rollout(sample=s, ok=False, reason=f"{type(e).__name__}: {e}"))

        good = [r for r in rolls if r.ok]
        if not good:
            worst = rolls[0]
            print(f"[{i:>3}/{len(queries)}] {q.id:<8} {q.cat:<17} REJECTED  "
                  f"{(worst.failed_checks or [worst.reason])[:1]}")
            rejected_f.write(json.dumps({
                "id": q.id, "cat": q.cat, "text": q.text,
                "attempts": [{
                    "sample": r.sample, "reason": r.reason,
                    "failed": r.failed_checks,
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
                } for r in rolls],
            }, ensure_ascii=False) + "\n")
            rejected_f.flush()
            continue

        pick = min(good, key=lambda r: r.n_tool_calls)
        messages, dropped = (
            trim_blocked_retries(pick.messages)
            if q.cat == "budget-exhausted" else (pick.messages, 0)
        )
        accepted_f.write(json.dumps({
            "id": q.id, "cat": q.cat, "lang": q.lang, "block": q.block,
            "text": q.text,
            "run_limit": spec.run_limit_for(q),
            "n_tool_calls": pick.n_tool_calls,
            "blocked_calls": pick.blocked_calls,
            "trimmed_blocked": dropped,
            "chose_sample": pick.sample,
            "n_candidates_passing": len(good),
            "system_prompt": pick.system_prompt,
            "messages": messages,
        }, ensure_ascii=False) + "\n")
        accepted_f.flush()
        n_ok += 1
        trim = f" trim-{dropped}" if dropped else ""
        print(f"[{i:>3}/{len(queries)}] {q.id:<8} {q.cat:<17} ok  "
              f"tools={pick.n_tool_calls:<3} best-of={len(good)}/{args.samples}{trim}")

    accepted_f.close()
    rejected_f.close()
    print(f"\naccepted {n_ok}/{len(queries)}  ->  {out_dir/'traces.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
