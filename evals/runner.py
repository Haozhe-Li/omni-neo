"""Execute one case against one model and produce a `RunTrace`.

The agent is driven through `astream(stream_mode=["messages", "updates"])` —
the same call shape `core/stream.py` uses in production. Streaming is not
optional here: `ttft_ms` only exists if chunks are timestamped as they arrive,
and switching to `ainvoke` would silently drop the entire latency half of layer
C.

The two stream modes carry different halves of the picture:

- `messages` — token-level chunks, timestamped. Source of the three TTFT
  measures and nothing else.
- `updates` — completed `AIMessage` / `ToolMessage` objects, already merged by
  LangGraph. Source of the tool trace, turn count and token usage.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from core.utils.citations import all_citations, reset_citation_registry
from evals.agent_factory import build_eval_agent, build_personalization, build_user_message
from evals.config import Case
from evals.models import ModelSpec
from evals.toolcache import ToolCache, load_fixture, wrap_tools
from evals.trace import RunTrace, ToolCallRecord, TurnTrace


async def run_case(
    case: Case,
    model: ModelSpec,
    *,
    cache: ToolCache,
) -> RunTrace:
    """Run every turn of `case` in one thread, returning the full trace."""
    trace = RunTrace(case_id=case.id, model_label=model.label)

    # thread_id=None keeps the citation registry run-scoped and in-memory: with
    # a thread id it would hydrate from (and persist to) the production Upstash
    # source store, so eval runs would both read real users' citation numbering
    # and write into it.
    reset_citation_registry(None, None)

    if case.fixture:
        load_fixture(cache, case.fixture)

    from core.agent import RETRIEVAL_TOOLS

    agent = build_eval_agent(model.llm, tools=wrap_tools(RETRIEVAL_TOOLS, cache))
    from evals.agent_factory import skill_files

    thread_id = f"eval-{uuid.uuid4()}"
    personalization = build_personalization(case.personalization)

    try:
        for i, turn in enumerate(case.turns):
            turn_trace = await asyncio.wait_for(
                _run_turn(
                    agent=agent,
                    index=i,
                    query=turn.text,
                    personalization=personalization,
                    thread_id=thread_id,
                    files=skill_files() if i == 0 else None,
                    requested_skill=None,
                ),
                timeout=case.timeout_s,
            )
            trace.turns.append(turn_trace)
    except asyncio.TimeoutError:
        trace.status = "timeout"
        trace.error = f"exceeded {case.timeout_s}s"
    except Exception as e:
        trace.status = "error"
        trace.error = f"{type(e).__name__}: {e}"

    trace.citations = [dict(c) for c in all_citations()]
    trace.cache_stats = cache.stats.as_dict()
    if trace.status == "ok" and not trace.turns:
        trace.status = "error"
        trace.error = trace.error or "no turns completed"
    return trace


async def _run_turn(
    *,
    agent,
    index: int,
    query: str,
    personalization: str,
    thread_id: str,
    files: dict | None,
    requested_skill: str | None,
) -> TurnTrace:
    turn = TurnTrace(index=index, query=query)
    content = build_user_message(query, personalization, skill=requested_skill)
    input_state: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
    if files:
        # Skill files ride in on the first turn only; the `files` channel merges
        # additively across turns, exactly as in production.
        input_state["files"] = files

    config = {"configurable": {"thread_id": thread_id}}
    started = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    tool_args_by_id: dict[str, tuple[str, dict]] = {}
    step = 0
    answer_started = False
    report_seen = False
    text_buffer = ""

    try:
        async for raw in agent.astream(
            input_state, config=config, stream_mode=["messages", "updates"], subgraphs=True
        ):
            mode, data = _normalize_stream_item(raw)

            if mode == "messages":
                chunk = data[0] if isinstance(data, tuple) else data
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if turn.ttft_ms is None:
                    # Anything at all — a reasoning token or the first tool-call
                    # fragment. This is the moment the UI stops looking frozen.
                    turn.ttft_ms = ms()
                text = _text_of(chunk.content)
                if text:
                    if not answer_started:
                        answer_started = True
                        turn.ttft_answer_ms = ms()
                    text_buffer += text
                    if not report_seen and "<report" in text_buffer:
                        report_seen = True
                        turn.ttft_report_ms = ms()

            elif mode == "updates":
                for node_state in (data or {}).values():
                    if not isinstance(node_state, dict):
                        continue
                    for message in node_state.get("messages") or []:
                        if isinstance(message, AIMessage):
                            turn.n_llm_turns += 1
                            turn.usage.add_usage(getattr(message, "usage_metadata", None))
                            body = _text_of(message.content).strip()
                            if message.tool_calls:
                                if body:
                                    turn.mixed_messages.append(body[:300])
                                for call in message.tool_calls:
                                    step += 1
                                    record = ToolCallRecord(
                                        index=step,
                                        name=call.get("name", "?"),
                                        args=call.get("args") or {},
                                    )
                                    turn.tool_calls.append(record)
                                    if call.get("id"):
                                        tool_args_by_id[call["id"]] = record
                            elif body:
                                turn.text = body
                        elif isinstance(message, ToolMessage):
                            record = tool_args_by_id.get(message.tool_call_id)
                            content_str = _text_of(message.content)
                            if record is not None:
                                record.result_full = content_str
                                record.result_head = content_str[:600]
                            if _is_run_limit(content_str):
                                turn.hit_run_limit = True
    except Exception as e:
        turn.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        turn.latency_ms = ms()

    if not turn.text and text_buffer.strip():
        # Fell through without a clean final AIMessage in `updates` (some
        # providers only surface the last message via the token stream) — use
        # what was streamed rather than scoring an empty answer.
        turn.text = text_buffer.strip()

    turn.text = _normalize_citations(turn.text)
    return turn


def _normalize_citations(text: str) -> str:
    """Apply production's citation repair before grading.

    `core/stream.py` fixes a class of citation formatting on the way to the
    frontend, so grading the raw model output would penalise the model for
    something the user never sees. What's graded is what shipped.
    """
    if not text:
        return text
    try:
        from core.stream import _normalize_citations as prod_normalize

        return prod_normalize(text)
    except Exception:
        return text


def _normalize_stream_item(item: Any) -> tuple[str | None, Any]:
    """Unwrap LangGraph's stream tuple.

    With `subgraphs=True` items arrive as `(namespace, mode, payload)`; without
    a subgraph they are `(mode, payload)`. Both shapes occur in one run because
    deepagents' middleware nests some steps and not others.
    """
    if isinstance(item, tuple):
        if len(item) == 3:
            return item[1], item[2]
        if len(item) == 2:
            return item[0], item[1]
    return None, item


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Skip reasoning blocks: they are not the answer, and counting
                # them as text would make ttft_answer_ms fire on the first
                # thinking token and report a ~0.5s answer latency for a run
                # that actually took two minutes to start writing.
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts)
    return ""


def _is_run_limit(text: str) -> bool:
    lowered = (text or "").lower()
    return "tool call limit" in lowered or "run limit" in lowered
