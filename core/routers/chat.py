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
    stream_reset,
    stream_read,
    STREAM_TTL_ACTIVE,
    STREAM_TTL_DONE,
)
from core.utils.data_model import Personalization, QueryRequest, CheckSourceRequest
from core.check_source import check_source_matches
from core.utils.citations import reset_citation_registry
from core.utils.utils import format_personalization, append_memory_context
from core.auth import get_current_user, GUEST_DAILY_LIMIT
from core.database.db_user_threads import (
    get_thread_messages,
    upsert_thread_messages,
    check_and_increment_guest_usage,
)
from core.database.db_threads_control import touch_thread
from core.database.db_user_memories import get_user_memory, save_user_memory
from core.memories_update_llm import get_update_memories
from core.routers.state import (
    db_executor,
    cancellation_events,
    generation_tasks,
    assert_thread_access,
    PERSIST_GRACE_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Redis write batching — text tokens are accumulated and flushed together to
# reduce round-trips from 2×N (xadd + expire per event) down to ~1×ceil(N/15).
# Non-text events (tool calls, artifacts, done …) bypass the batch and flush
# immediately so the frontend doesn't wait for them.
_FLUSH_IMMEDIATELY: frozenset[str] = frozenset({
    "tool_call", "artifact", "sources", "done", "error", "stopped",
    "widget", "drafting", "reasoning",
})
_BATCH_SIZE = 15        # flush when this many events are pending
_BATCH_TIMEOUT_S = 0.03 # flush after 30 ms even if batch isn't full (slow models)


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
    # Accumulate message fields for the final Postgres upsert.
    text = ""
    steps: list[dict] = []
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
            try:
                ev = json.loads(event_str[6:])  # strip "data: "
            except Exception:
                ev = {}

            ev_type = ev.get("type")
            if ev_type == "text":
                text += ev.get("content", "")
            elif ev_type == "tool_call":
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
            if (
                ev_type in _FLUSH_IMMEDIATELY
                or len(batch) >= _BATCH_SIZE
                or time.monotonic() - last_flush >= _BATCH_TIMEOUT_S
            ):
                await _flush()

        await _flush()  # drain any remaining buffered events

        # Shrink TTL now that generation is complete.
        await stream_expire(thread_id)
        await stream_set_status(thread_id, "done", STREAM_TTL_DONE)

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
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        error_event = f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        batch.append(error_event)
        await _flush()
        await stream_set_status(thread_id, "error", STREAM_TTL_DONE)
        await stream_expire(thread_id)
    finally:
        cancellation_events.pop(thread_id, None)
        generation_tasks.pop(thread_id, None)


# ---------------------------------------------------------------------------
# Chat – the single all-around agent endpoint
# ---------------------------------------------------------------------------


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

    `request.mode` selects the profile: "fast" (lean, gpt-oss — unmetered) or
    "pro" (deep agent, Gemini, with chart/report artifacts). Only "pro" counts
    against the guest daily quota; "fast" is free and unlimited.
    """
    # Pro mode is the metered profile: enforce the guest daily cap (fast is free).
    if request.mode == "pro" and user_id.startswith("guest_"):
        count = check_and_increment_guest_usage(user_id)
        if count > GUEST_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Daily Pro limit reached for guest users. Please sign in to continue.",
            )

    assert_thread_access(request.thread_id, user_id)
    thread_id = request.thread_id

    if thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(db_executor, touch_thread, thread_id, user_id)

    # If this thread already has an active background generation, just reconnect.
    if thread_id and await stream_is_generating(thread_id):
        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(
            stream_read(thread_id),
            media_type="text/event-stream",
            headers=headers,
        )

    query_text = request.query
    if request.follow_up_content:
        query_text += f"\n\nFollow up text selection: {request.follow_up_content}"
    if request.skill:
        query_text += f"\n\nUser explicitly asked for the '{request.skill}' skill. Please first activate this skill."

    personalization_str = format_personalization(request.personalization)
    p = request.personalization
    memory_enabled = bool(p and p.memory_enabled)
    if memory_enabled:
        stored_memory = await asyncio.to_thread(get_user_memory, user_id)
        personalization_str = append_memory_context(personalization_str, stored_memory)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

    if thread_id:
        # ── background task mode: survives HTTP disconnect ─────────────────
        cancel_event = asyncio.Event()
        cancellation_events[thread_id] = cancel_event
        # Clear the previous turn's buffer so stream_read doesn't replay it.
        await stream_reset(thread_id)
        # Set status *before* starting stream_read so it doesn't exit on an empty stream.
        await stream_set_status(thread_id, "generating", STREAM_TTL_ACTIVE)
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
        return StreamingResponse(
            stream_read(thread_id),
            media_type="text/event-stream",
            headers=headers,
        )

    # ── no thread_id: direct streaming, backward-compatible ───────────────
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
    """
    assert_thread_access(thread_id, user_id)

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
        reset_citation_registry(thread_id, body.turn)
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
    assert_thread_access(thread_id, user_id)
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
    assert_thread_access(request.thread_id, user_id)

    return await check_source_matches(
        request.thread_id,
        request.text_selection,
        request.turn,
    )
