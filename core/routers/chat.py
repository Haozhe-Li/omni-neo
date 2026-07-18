import asyncio
import json
import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage as LCHumanMessage

from core.agent import get_agent
from core.stream import run_agent_stream, build_message_content
from core.redis_stream import (
    stream_write_batch,
    stream_set_status,
    stream_is_generating,
    stream_expire,
    stream_begin,
    stream_read,
    STREAM_TTL_DONE,
)
from core.utils.data_model import Personalization, QueryRequest, CheckSourceRequest
from core.check_source import check_source_matches
from core.utils.citations import reset_citation_registry_async
from core.utils.utils import format_personalization, append_memory_context
from core.auth import get_current_user
from core.database.db_user_threads import (
    get_thread_messages,
    upsert_thread_messages,
)
from core.database.db_user_usage import evaluate_charge_fast, commit_charge_fast, usage_snapshot_hit
from core.database.db_threads_control import touch_thread, get_thread_owner_async, owner_cache_hit
from core.utils.timing import Timing
from core.database.db_user_memories import get_user_memory, save_user_memory
from core.memories_update_llm import get_update_memories
from core.routers.state import (
    db_executor,
    cancellation_events,
    generation_tasks,
    assert_thread_access_async,
    PERSIST_GRACE_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Redis write batching — text/reasoning tokens are accumulated and flushed
# together to reduce round-trips from 2×N (xadd + expire per event) down to
# ~1×ceil(N/batch). Non-token events (tool calls, artifacts, done …) bypass
# the batch and flush immediately so the frontend doesn't wait for them.
_FLUSH_IMMEDIATELY: frozenset[str] = frozenset({
    "tool_call", "artifact", "sources", "done", "error", "stopped",
    "widget", "drafting",
})
_BATCH_SIZE = 15        # flush when this many events are pending
_BATCH_TIMEOUT_S = 0.03 # flush after 30 ms even if batch isn't full (slow models)
# Reasoning renders inside a collapsed "Thinking" step, so its latency matters
# less than answer text — buffer it bigger/longer to cut Redis round-trips
# (gpt-oss emits reasoning one token per chunk, easily hundreds per turn).
_BATCH_SIZE_REASONING = 30
_BATCH_TIMEOUT_REASONING_S = 0.03


async def _update_memory_background(user_id: str, user_query: str, assistant_text: str) -> None:
    """Fire-and-forget: extract durable facts from this turn and persist them.

    Independent of thread persistence below — runs even if the client never
    reconnects to sync the turn, since memory is keyed by user_id, not thread_id.
    """
    try:
        current = await asyncio.to_thread(get_user_memory, user_id)
        updated = await get_update_memories(current, user_query, assistant_text)
        if updated.strip() != current.strip():
            await asyncio.to_thread(save_user_memory, user_id, updated)
    except Exception as e:
        logger.error(f"[chat] memory update failed for user {user_id}: {e}")


async def _generate_background(
    thread_id: str,
    user_id: str,
    query: str,
    mode: str,
    personalization: str,
    attached_file_ids: list | None,
    user_location: str | None,
    user_local_datetime: str | None,
    turn: int | None,
    cancel_event: asyncio.Event | None,
    memory_enabled: bool = False,
) -> None:
    """Run the agent, buffer every SSE event to Redis, and save to Postgres on done."""
    # Structured timing for the generation task: how long until the first event
    # reaches the buffer (brackets the in-task pre-LLM work + first model call),
    # the first text token, and total run. Complements the chat_prelude span.
    tg = Timing("chat_generation", thread_id=thread_id, mode=mode)
    n_events = 0

    # Accumulate message fields for the final Postgres upsert. `steps` is the
    # same interleaved timeline the frontend builds: tool_call entries plus
    # {type: "reasoning"} entries, one per contiguous thinking run (a tool call
    # or answer text closes the open run).
    text = ""
    steps: list[dict] = []
    open_reasoning: dict | None = None
    sources: list[dict] = []
    artifacts: list[dict] = []
    widgets: list[dict] = []

    batch: list[str] = []
    last_flush = time.monotonic()

    async def _flush() -> None:
        nonlocal batch, last_flush
        if batch:
            await stream_write_batch(thread_id, batch)
            batch = []
        last_flush = time.monotonic()

    try:
        async for event_str in run_agent_stream(
            query=query,
            thread_id=thread_id,
            mode=mode,
            personalization=personalization,
            attached_file_ids=attached_file_ids,
            user_location=user_location,
            user_local_datetime=user_local_datetime,
            turn=turn,
            cancellation_event=cancel_event,
        ):
            if n_events == 0:
                tg.mark("first_event")
            n_events += 1

            try:
                ev = json.loads(event_str[6:])  # strip "data: "
            except Exception:
                ev = {}

            ev_type = ev.get("type")
            if ev_type == "text":
                if "first_text_ms" not in tg.stages:
                    tg.mark("first_text")
                open_reasoning = None
                text += ev.get("content", "")
            elif ev_type == "reasoning":
                if open_reasoning is not None:
                    open_reasoning["content"] += ev.get("content", "")
                else:
                    open_reasoning = {
                        "type": "reasoning",
                        "content": ev.get("content", ""),
                        "timestamp": int(time.time() * 1000),
                    }
                    steps.append(open_reasoning)
            elif ev_type == "tool_call":
                open_reasoning = None
                steps.append({
                    "tool": ev.get("tool"),
                    "args": ev.get("args", {}),
                    "timestamp": int(time.time() * 1000),
                })
            elif ev_type == "sources":
                sources.extend(ev.get("sources", []))
            elif ev_type == "artifact":
                artifacts.append({
                    "id": ev["id"],
                    "title": ev.get("title"),
                    "kind": "echarts",
                    "spec": ev.get("spec"),
                })
            elif ev_type == "widget":
                widgets.append({"widget": ev.get("widget"), "data": ev.get("data")})

            batch.append(event_str)
            batch_size, batch_timeout = (
                (_BATCH_SIZE_REASONING, _BATCH_TIMEOUT_REASONING_S)
                if ev_type == "reasoning"
                else (_BATCH_SIZE, _BATCH_TIMEOUT_S)
            )
            if (
                ev_type in _FLUSH_IMMEDIATELY
                or len(batch) >= batch_size
                or time.monotonic() - last_flush >= batch_timeout
            ):
                await _flush()

        await _flush()  # drain any remaining buffered events

        # Shrink TTL now that generation is complete.
        await stream_expire(thread_id)
        await stream_set_status(thread_id, "done", STREAM_TTL_DONE)
        tg.emit(outcome="done", events=n_events, chars=len(text))

        if memory_enabled and text:
            asyncio.create_task(_update_memory_background(user_id, query, text))

        # Persist to Postgres as a FALLBACK only. A connected client — even one
        # that navigated away within the SPA — syncs the turn itself on the `done`
        # event via POST /sync. Writing here unconditionally races that sync and
        # produces a duplicate assistant message. So we wait a short grace period
        # for the client to sync, then write only if it didn't (e.g. tab closed).
        if thread_id and text:
            await asyncio.sleep(PERSIST_GRACE_SECONDS)
            existing = await asyncio.to_thread(get_thread_messages, thread_id, user_id) or []
            # Skip if the client already synced this turn (its text is present),
            # which also guards against a stale fallback after a rapid next turn.
            already_synced = any(
                isinstance(m, dict)
                and m.get("role") == "assistant"
                and m.get("content") == text
                for m in existing
            )
            if not already_synced:
                msgs = list(existing)
                # The frontend persists the user's question at turn start; only add
                # one here if it isn't already the trailing message (tab closed
                # before that early sync landed).
                if not (msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "user"):
                    msgs.append({"role": "user", "content": query})
                msgs.append({
                    "role": "assistant",
                    "content": text,
                    "steps": steps,
                    "sources": sources,
                    "artifacts": artifacts,
                    "widgets": widgets,
                })
                await asyncio.to_thread(
                    upsert_thread_messages, thread_id, user_id, msgs,
                )

    except asyncio.CancelledError:
        await _flush()
        await stream_set_status(thread_id, "done", STREAM_TTL_DONE)
        await stream_expire(thread_id)
        tg.emit(outcome="cancelled", events=n_events, chars=len(text))
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        error_event = f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        batch.append(error_event)
        await _flush()
        await stream_set_status(thread_id, "error", STREAM_TTL_DONE)
        await stream_expire(thread_id)
        tg.emit(outcome="error", events=n_events, error=str(exc)[:200])
    finally:
        cancellation_events.pop(thread_id, None)
        generation_tasks.pop(thread_id, None)


# ---------------------------------------------------------------------------
# Chat – the single all-around agent endpoint
# ---------------------------------------------------------------------------


def _usage_limit_detail(user_id: str, usage: dict) -> dict:
    """The structured 429 body — scope, current totals, reset times — the
    frontend uses to render the usage-limit dialog."""
    return {
        "error": "usage_limit_exceeded",
        "scope": usage["exceeded_scope"],
        "is_guest": user_id.startswith("guest_"),
        "day_used": usage["day_used"], "day_limit": usage["day_limit"],
        "month_used": usage["month_used"], "month_limit": usage["month_limit"],
        "resets_day_at": usage["resets_day_at"],
        "resets_month_at": usage["resets_month_at"],
    }


@router.post("/chat")
async def chat(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Stream the Omni agent's response as Server-Sent Events.

    The actual generation runs as a background asyncio task that survives HTTP
    disconnects — events are buffered in a Redis Stream so the client can
    reconnect at any time and replay from the beginning.

    `request.mode` selects the profile: "fast" (lean, gpt-oss) or "pro" (deep
    agent, Gemini, with chart/report artifacts). Every turn spends credits
    against the caller's daily/monthly usage.
    """
    thread_id = request.thread_id
    p = request.personalization
    memory_enabled = bool(p and p.memory_enabled)

    # Structured timing for the non-LLM prelude (see core/utils/timing.py).
    t = Timing("chat_prelude", thread_id=thread_id, mode=request.mode,
               is_guest=user_id.startswith("guest_"), memory_enabled=memory_enabled)
    t.set(owner_cache=("hit" if (thread_id and owner_cache_hit(thread_id)) else "miss"),
          charge_cache=("hit" if usage_snapshot_hit(user_id) else "miss"))

    # ── one concurrent batch of the independent pre-LLM reads ──────────────
    # These four don't depend on each other, so fire them together (~1 round
    # trip) instead of serially (~5). The charge is only *evaluated* here — the
    # write is deferred until we've confirmed we're actually starting a new
    # generation, so a reconnect (below) never charges.
    async def _owner():
        return await get_thread_owner_async(thread_id) if thread_id else None

    async def _generating():
        return await stream_is_generating(thread_id) if thread_id else False

    async def _memory():
        return await asyncio.to_thread(get_user_memory, user_id) if memory_enabled else ""

    _batch_start = time.perf_counter()
    charge_result, owner, is_generating_now, stored_memory = await asyncio.gather(
        t.atimed("charge", evaluate_charge_fast(user_id, request.mode)),
        t.atimed("owner", _owner()),
        t.atimed("is_generating", _generating()),
        t.atimed("memory", _memory()),
    )
    t.record("batch", (time.perf_counter() - _batch_start) * 1000)

    # ── decide, in priority order (pure in-memory, no I/O) ─────────────────
    # 1. access gate — reject before any side effect (nothing charged yet)
    if owner is not None and owner != user_id:
        t.emit(outcome="denied")
        raise HTTPException(status_code=403, detail="Thread access denied.")
    # 2. reconnect gate — a generation is already in flight: just re-attach.
    #    No charge (already paid when it started), no new LLM run.
    if thread_id and is_generating_now:
        t.emit(outcome="reconnect")
        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(
            stream_read(thread_id),
            media_type="text/event-stream",
            headers=headers,
        )
    # 3. usage gate
    if not charge_result["charged"]:
        t.emit(outcome="usage_limit")
        raise HTTPException(status_code=429, detail=_usage_limit_detail(user_id, charge_result))

    # Committed to a new generation → reconcile the charge in the background
    # (off the critical path) and bump the thread's updated_at.
    commit_charge_fast(user_id, request.mode)
    if thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(db_executor, touch_thread, thread_id, user_id)

    query_text = request.query
    if request.follow_up_content:
        query_text += f"\n\nFollow up text selection: {request.follow_up_content}"
    if request.skill:
        query_text += f"\n\nUser explicitly asked for the '{request.skill}' skill. Please first activate this skill."

    personalization_str = format_personalization(request.personalization)
    if memory_enabled:
        personalization_str = append_memory_context(personalization_str, stored_memory)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

    if thread_id:
        # ── background task mode: survives HTTP disconnect ─────────────────
        cancel_event = asyncio.Event()
        cancellation_events[thread_id] = cancel_event
        # Clear the previous turn's buffer and mark this one generating, in a
        # single pipelined round trip (status set *before* stream_read starts so
        # it doesn't exit on an empty stream).
        with t.stage("stream_begin"):
            await stream_begin(thread_id)
        task = asyncio.create_task(
            _generate_background(
                thread_id=thread_id,
                user_id=user_id,
                query=query_text,
                mode=request.mode,
                personalization=personalization_str,
                attached_file_ids=request.attached_file_ids,
                user_location=p.user_location if p else None,
                user_local_datetime=p.user_local_datetime if p else None,
                turn=request.turn,
                cancel_event=cancel_event,
                memory_enabled=memory_enabled,
            )
        )
        generation_tasks[thread_id] = task
        t.emit(outcome="started")
        return StreamingResponse(
            stream_read(thread_id),
            media_type="text/event-stream",
            headers=headers,
        )

    # ── no thread_id: direct streaming, backward-compatible ───────────────
    t.emit(outcome="started_direct")
    cancel_event = asyncio.Event()

    async def _direct_stream():
        try:
            async for chunk in run_agent_stream(
                query=query_text,
                thread_id=None,
                mode=request.mode,
                personalization=personalization_str,
                attached_file_ids=request.attached_file_ids,
                user_location=p.user_location if p else None,
                user_local_datetime=p.user_local_datetime if p else None,
                turn=request.turn,
                cancellation_event=cancel_event,
            ):
                yield chunk
        finally:
            pass

    return StreamingResponse(
        _direct_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


class RewindRequest(BaseModel):
    mode: Literal["fast", "pro"] = "fast"
    new_query: str | None = None  # None = pure regenerate; set to edit the last user msg
    personalization: Personalization | None = None
    attached_file_ids: list[dict[str, str]] | None = None
    turn: int | None = None


@router.post("/api/threads/{thread_id}/rewind")
async def api_rewind_thread(
    thread_id: str,
    body: RewindRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Regenerate or edit-and-resend the last user message in a thread.

    - ``new_query=None``  → pure regenerate: re-run from the same user message.
    - ``new_query="…"``   → edit: replace the last user message then re-run.

    Uses LangGraph time travel: locates the checkpoint where the last message is a
    HumanMessage, optionally forks it via ``aupdate_state``, then streams from there.
    The LangGraph agent state is rewound; the frontend is responsible for trimming
    its own UI message list before calling this endpoint.

    This re-runs the agent exactly like a fresh /chat turn, so it's metered
    the same way — otherwise "regenerate" would be a free, unlimited bypass
    around the credit system.
    """
    t = Timing("rewind_prelude", thread_id=thread_id, mode=body.mode,
               is_guest=user_id.startswith("guest_"))
    # Charge-evaluate and ownership check are independent reads — run them
    # concurrently (~1 round trip instead of 2), then commit the charge
    # fire-and-forget once both gates pass. Rewind has no reconnect path.
    _b = time.perf_counter()
    charge_result, owner = await asyncio.gather(
        t.atimed("charge", evaluate_charge_fast(user_id, body.mode)),
        t.atimed("owner", get_thread_owner_async(thread_id)),
    )
    t.record("batch", (time.perf_counter() - _b) * 1000)
    if owner is not None and owner != user_id:
        t.emit(outcome="denied")
        raise HTTPException(status_code=403, detail="Thread access denied.")
    if not charge_result["charged"]:
        t.emit(outcome="usage_limit")
        raise HTTPException(status_code=429, detail=_usage_limit_detail(user_id, charge_result))
    commit_charge_fast(user_id, body.mode)
    t.emit(outcome="charged")

    profile = "pro" if body.mode == "pro" else "fast"
    agent = get_agent(profile)
    lg_config = {"configurable": {"thread_id": thread_id}}

    # Find the most recent checkpoint where the last message is a HumanMessage
    # (i.e. the point just after the user sent their message, before the agent replied).
    target = None
    async for state in agent.aget_state_history(lg_config):
        msgs = state.values.get("messages", [])
        if msgs and isinstance(msgs[-1], LCHumanMessage):
            target = state
            break

    if target is None:
        raise HTTPException(status_code=404, detail="No rewindable checkpoint found.")

    doc_sources: list[dict] = []
    if body.new_query is not None:
        # Edit mode: replace the last HumanMessage in-place (same id → add_messages
        # reducer treats it as an update, not an append).
        personalization_str = format_personalization(body.personalization)
        # build_message_content assigns a citation number to any newly-attached
        # document via register_document_citation, which needs the thread/turn
        # context this sets up — same ordering requirement as in _stream_agent.
        await reset_citation_registry_async(thread_id, body.turn)
        new_content, doc_files, doc_sources = await asyncio.to_thread(
            build_message_content,
            body.new_query, personalization_str, body.attached_file_ids, thread_id,
        )
        last_human = target.values["messages"][-1]
        updated_msg = LCHumanMessage(id=last_human.id, content=new_content)
        state_update = {"messages": [updated_msg]}
        if doc_files:
            state_update["files"] = doc_files
        rewind_config = await agent.aupdate_state(target.config, state_update)
    else:
        # Regenerate mode: replay from the existing checkpoint as-is.
        rewind_config = target.config

    p = body.personalization
    personalization_str = format_personalization(p)
    if p and p.memory_enabled:
        stored_memory = await asyncio.to_thread(get_user_memory, user_id)
        personalization_str = append_memory_context(personalization_str, stored_memory)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    cancel_event = asyncio.Event()
    cancellation_events[thread_id] = cancel_event

    async def _rewind_stream_with_cleanup():
        try:
            async for chunk in run_agent_stream(
                query="",  # unused in rewind mode
                thread_id=thread_id,
                mode=body.mode,
                personalization=personalization_str,
                attached_file_ids=body.attached_file_ids,
                user_location=p.user_location if p else None,
                user_local_datetime=p.user_local_datetime if p else None,
                turn=body.turn,
                rewind_config=rewind_config,
                extra_sources=doc_sources,
                cancellation_event=cancel_event,
            ):
                yield chunk
        finally:
            cancellation_events.pop(thread_id, None)

    return StreamingResponse(
        _rewind_stream_with_cleanup(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/api/threads/{thread_id}/stop")
async def stop_generation(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """Signal the active generation for this thread to stop."""
    await assert_thread_access_async(thread_id, user_id)
    event = cancellation_events.get(thread_id)
    if event:
        event.set()
        return {"status": "stopped"}
    return {"status": "not_running"}


@router.post("/check_source")
async def check_source(
    request: CheckSourceRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Find every source passage that supports a piece of highlighted answer text.

    1. Semantic search (Upstash Search) over every chunk this thread has ever
       produced, filtered to `turn <= request.turn` so a claim from an early
       turn can never resolve to a source that only appeared later — see
       `core/utils/citations.py` for how `turn` gets stamped onto a source.
    2. An LLM rerank pass drops chunks that are merely topically similar (not
       actually supporting), and extracts a verbatim excerpt from each
       survivor for the frontend to fuzzy-match and highlight precisely.
    """
    await assert_thread_access_async(request.thread_id, user_id)

    return await check_source_matches(
        request.thread_id,
        request.text_selection,
        request.turn,
    )
