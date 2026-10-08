"""Training-data collector API: generate an answer under inputs a human chose,
edit it, and file the conversation as a training example.

Operated through the password-protected /collect page of the frontend, whose
server holds COLLECTOR_API_KEY (see core/auth.py `get_collector`). Nothing here
is reachable with a Clerk token or a guest id.

## The rule this module exists to keep: the model sees what production shows it

A training example is worthless if the model was fed something production never
would. So the collector adds no code path of its own into the agent:

- The request is `CollectorGenerateRequest` (core/collector_schema.py), which
  accepts only values production's client can produce, in production's formats.
- It is turned into the `QueryRequest` the frontend would have POSTed to /chat,
  and the system reminder / memory block are built from that by the same
  functions /chat calls (`build_turn_context` in core/utils/utils.py).
- Generation is `_generate_background` from core/routers/chat.py, unmodified:
  same agent, same pre-flight scout, same message layout (`build_message_content`),
  same Redis stream, same checkpointer.
- The `turn` number is derived here from the checkpoint rather than trusted from
  the client, with production's convention (the K-th user message is turn 2K-1).
  It decides whether memory and the scout run, so a wrong one would silently
  change the input.
- What is stored is the real LangGraph checkpoint through `build_capture` — the
  same function a thumbs-up uses — not anything the UI held.

The one deliberate difference is the memory text: production reads it from
Postgres, the collector takes whatever the annotator wrote (a human-written
persona, never a real person's data), and so passes `allow_memory=True`.

## Editing

An edit replaces the text of the last turn's final assistant message in the
checkpoint itself (same id, same response metadata), so the next turn's model
reads the edited answer exactly as production would have read a real one, and
the stored `messages` are the checkpoint, edited. The original is kept in
collector_turns so the example records what was changed. Only the latest turn
can be edited: earlier ones are already history that a later answer was written
against.
"""

import asyncio
import logging
import uuid
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from core.agent import get_agent, resolve_skill_name
from core.auth import get_collector
from core.chat_models import resolve_model
from core.collector_schema import CollectorGenerateRequest, to_query_request
from core.database import db_collector
from core.database.db_sft_examples import asave_collector_example
from core.database.db_threads_control import delete_thread as delete_thread_state, upsert_thread
from core.database.db_user_memories import MAX_MEMORY_CHARS
from core.database.db_user_threads import delete_user_thread, get_thread_row, register_thread
from core.harness_snapshot import live_harness
from core.redis_stream import stream_get_status, stream_is_generating, stream_begin, stream_read
from core.routers.chat import _generate_background
from core.routers.state import cancellation_events, generation_tasks
from core.sft_capture import CaptureError, _text_of, build_capture, slice_through_turn
from core.utils.utils import build_turn_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/collector", tags=["collector"])

COLLECTOR_ORIGIN = "collector"
MAX_EDITED_CHARS = 200_000

_SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


def _deepagents_version() -> str | None:
    try:
        return version("deepagents")
    except PackageNotFoundError:
        return None


# ── helpers ─────────────────────────────────────────────────────────────────


async def _owned_thread(thread_id: str, user_id: str) -> dict:
    """The thread's row, 404 unless it is a collector thread of this annotator.

    `get_thread_row` already scopes by user; the origin check keeps this API from
    touching an ordinary chat thread even if ids were ever shared."""
    row = await asyncio.to_thread(get_thread_row, thread_id, user_id)
    if row is None or row.get("origin") != COLLECTOR_ORIGIN:
        raise HTTPException(status_code=404, detail="Collector thread not found.")
    return row


async def _checkpoint_messages(thread_id: str) -> list:
    state = await get_agent(None).aget_state({"configurable": {"thread_id": thread_id}})
    return list((state.values or {}).get("messages", []))


def _n_user_turns(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _last_turn(messages: list) -> int:
    """Production's turn number for the newest exchange: the K-th user message is 2K-1."""
    return 2 * _n_user_turns(messages) - 1


def _finished_prefix(messages: list) -> list | None:
    """The checkpoint through the newest exchange if that exchange ended on a
    finished answer, else None (empty thread, in flight, failed, mid-tool)."""
    if not messages:
        return None
    try:
        return slice_through_turn(messages, _last_turn(messages))
    except CaptureError:
        return None


async def _idle_or_409(thread_id: str) -> None:
    if await stream_is_generating(thread_id):
        raise HTTPException(status_code=409, detail="This thread is still generating.")


# ── threads ─────────────────────────────────────────────────────────────────


@router.post("/threads")
async def create_thread(user_id: str = Depends(get_collector)):
    """A fresh collector thread — the counterpart of GET /get_thread_id. Origin
    'collector' keeps it out of every chat list and away from the thumbs-up path."""
    thread_id = str(uuid.uuid4())
    await asyncio.to_thread(upsert_thread, thread_id, user_id)
    ok = await asyncio.to_thread(register_thread, thread_id, user_id, COLLECTOR_ORIGIN)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not create the thread.")
    return {"thread_id": thread_id}


@router.delete("/threads/{thread_id}")
async def discard_thread(thread_id: str, user_id: str = Depends(get_collector)):
    """Throw a collector thread away (its checkpoint, UI row and recorded turns).
    A thread already submitted keeps its sft_examples row — that is the dataset."""
    await _owned_thread(thread_id, user_id)
    event = cancellation_events.get(thread_id)
    if event:
        event.set()
    await asyncio.to_thread(delete_user_thread, thread_id, user_id)
    await asyncio.to_thread(delete_thread_state, thread_id)
    await db_collector.adelete_turns(thread_id, user_id)
    return {"status": "deleted"}


# ── generate ────────────────────────────────────────────────────────────────


@router.post("/generate")
async def generate(body: CollectorGenerateRequest, user_id: str = Depends(get_collector)):
    """Run one turn of the real agent and stream it back as /chat does (same SSE)."""
    row = await _owned_thread(body.thread_id, user_id)
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="This thread was locked for safety.")
    thread_id = body.thread_id

    if body.memory and len(body.memory) > MAX_MEMORY_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"memory is {len(body.memory)} chars; production stores at most {MAX_MEMORY_CHARS}.",
        )

    # Pre-flight reads, then the checks that depend on them.
    await _idle_or_409(thread_id)
    messages = await _checkpoint_messages(thread_id)
    if messages and _finished_prefix(messages) is None:
        raise HTTPException(
            status_code=409,
            detail="The previous turn did not finish. Discard this thread and start a new one.",
        )
    turn = 2 * _n_user_turns(messages) + 1
    # Every turn already in the checkpoint must have been recorded by this API. A
    # row at `turn` or later is an attempt that never reached the checkpoint (it
    # was refused before the model ran); it is simply overwritten below.
    recorded = [t["turn"] for t in await db_collector.alist_turns(thread_id, user_id) if t["turn"] < turn]
    if recorded != list(range(1, turn, 2)):
        raise HTTPException(
            status_code=409,
            detail="This thread has turns that did not go through the collector; discard it.",
        )
    if turn > 1 and body.memory:
        # Production injects `<user_memory>` on turn 1 only; accepting it here would
        # record a memory the model never saw.
        raise HTTPException(status_code=400, detail="memory can only be set on the first turn.")

    model = resolve_model(body.model)  # validated against COLLECTOR_MODELS by the schema
    request = to_query_request(body, turn=turn)
    system_reminder, user_memory = build_turn_context(request, body.memory)
    p = request.personalization

    await db_collector.arecord_turn(
        thread_id=thread_id,
        turn=turn,
        user_id=user_id,
        model=model.id,
        personalization=body.personalization.model_dump(exclude_none=True),
        memory=body.memory,
        skill=body.skill,
    )

    cancel_event = asyncio.Event()
    cancellation_events[thread_id] = cancel_event
    await stream_begin(thread_id)
    task = asyncio.create_task(
        _generate_background(
            thread_id=thread_id,
            user_id=user_id,
            query=request.query,
            model_id=model.id,
            system_reminder=system_reminder,
            attached_file_ids=None,
            user_memory=user_memory,
            follow_up_content=None,
            # Resolved exactly as /chat resolves it (`deep-research` -> `web-research`).
            skill=resolve_skill_name(request.skill),
            user_location=p.user_location,
            user_local_datetime=p.user_local_datetime,
            turn=turn,
            cancel_event=cancel_event,
            # Never: the memory text here is invented, and must not be written into
            # a user_memories row by the post-turn extraction.
            memory_enabled=False,
            source_url=None,
        )
    )
    generation_tasks[thread_id] = task
    return StreamingResponse(
        stream_read(thread_id), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/threads/{thread_id}/stream")
async def reconnect_stream(thread_id: str, user_id: str = Depends(get_collector)):
    """Re-attach to a generation whose HTTP stream dropped (a serverless proxy
    timing out, a closed tab). Replays the buffered events, then goes live."""
    await _owned_thread(thread_id, user_id)
    if await stream_get_status(thread_id) is None:
        raise HTTPException(status_code=404, detail="No active stream for this thread.")
    return StreamingResponse(
        stream_read(thread_id), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/threads/{thread_id}/stop")
async def stop(thread_id: str, user_id: str = Depends(get_collector)):
    await _owned_thread(thread_id, user_id)
    event = cancellation_events.get(thread_id)
    if event:
        event.set()
        return {"status": "stopped"}
    return {"status": "not_running"}


# ── state, edit ─────────────────────────────────────────────────────────────


@router.get("/threads/{thread_id}/state")
async def thread_state(thread_id: str, user_id: str = Depends(get_collector)):
    """Where the thread stands, and the newest answer as the checkpoint holds it.

    `final_text` is the text of the last assistant message in the checkpoint — the
    string a training row would end on — not the text that streamed to the page,
    which the stream layer may have rewritten. The editor starts from this, so
    submitting without touching it records exactly what a thumbs-up would have.
    """
    row = await _owned_thread(thread_id, user_id)
    generating = await stream_is_generating(thread_id)
    messages = await _checkpoint_messages(thread_id)
    prefix = None if generating else _finished_prefix(messages)
    last_turn = _last_turn(messages) if messages else 0
    turns = [t for t in await db_collector.alist_turns(thread_id, user_id) if t["turn"] <= last_turn]
    return {
        "is_generating": generating,
        "is_locked": bool(row.get("is_locked")),
        "n_turns": _n_user_turns(messages),
        "turn": last_turn,
        "complete": prefix is not None,
        "final_text": _text_of(prefix[-1].content) if prefix else None,
        "turns": [
            {
                "turn": t["turn"],
                "model": t["model"],
                "personalization": t["personalization"],
                "memory": t["memory"],
                "skill": t["skill"],
                "edited": t["edited_final_text"] is not None
                and t["edited_final_text"] != t["original_final_text"],
            }
            for t in turns
        ],
    }


class EditFinalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=MAX_EDITED_CHARS)


@router.put("/threads/{thread_id}/final")
async def edit_final(
    thread_id: str, body: EditFinalRequest, user_id: str = Depends(get_collector)
):
    """Replace the newest turn's final answer with the annotator's version.

    Writes the checkpoint (see the module docstring), so a following turn builds on
    the edited answer. Idempotent: PUT the same text twice and nothing changes.
    """
    await _owned_thread(thread_id, user_id)
    await _idle_or_409(thread_id)
    messages = await _checkpoint_messages(thread_id)
    prefix = _finished_prefix(messages)
    if prefix is None:
        raise HTTPException(status_code=409, detail="There is no finished answer to edit.")
    last = prefix[-1]
    if not isinstance(last, AIMessage) or not last.id:
        raise HTTPException(status_code=409, detail="The newest message cannot be edited.")
    turn = _last_turn(messages)

    current = _text_of(last.content)
    if body.text != current:
        # Same id -> the add_messages reducer replaces rather than appends, and
        # model_copy keeps response_metadata, which the teacher check reads.
        await get_agent(None).aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": [last.model_copy(update={"content": body.text})]},
        )
    if not await db_collector.arecord_edit(thread_id, turn, user_id, current, body.text):
        raise HTTPException(status_code=409, detail="That turn was not generated through the collector.")
    return {"status": "saved", "turn": turn, "changed": body.text != current}


# ── submit ──────────────────────────────────────────────────────────────────


class SubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(default=None, max_length=2000)


@router.post("/threads/{thread_id}/submit")
async def submit(
    thread_id: str, body: SubmitRequest, user_id: str = Depends(get_collector)
):
    """File the conversation through its newest turn as a training example.

    Status: 'accepted' when the annotator edited any answer in it (a person has
    read and corrected it), 'pending' when they filed the model's words untouched
    — the same standing a thumbs-up gives.
    """
    row = await _owned_thread(thread_id, user_id)
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="This thread was locked for safety.")
    await _idle_or_409(thread_id)

    state = await get_agent(None).aget_state({"configurable": {"thread_id": thread_id}})
    values = state.values or {}
    messages = list(values.get("messages", []))
    prefix = _finished_prefix(messages)
    if prefix is None:
        raise HTTPException(status_code=409, detail="The newest turn has no finished answer.")
    turn = _last_turn(messages)

    turns = [t for t in await db_collector.alist_turns(thread_id, user_id) if t["turn"] <= turn]
    if [t["turn"] for t in turns] != list(range(1, turn + 1, 2)):
        raise HTTPException(
            status_code=409,
            detail="This thread has turns that did not go through the collector; discard it.",
        )

    try:
        cap = build_capture(
            messages,
            turn,
            compacted=bool(values.get("_summarization_event")),
            allow_memory=True,
        )
    except CaptureError as e:
        # Unlike a thumbs-up, the annotator is watching: say why instead of "skipped".
        raise HTTPException(status_code=422, detail={"reason": e.code, "message": str(e)})

    edits = [t for t in turns if t["edited_final_text"] is not None and t["edited_final_text"] != t["original_final_text"]]
    edited = bool(edits)
    # The row's headline "before" is the newest turn's original if it was edited;
    # per-turn originals for earlier turns live in collect_meta.
    last_turn_row = turns[-1]
    original_final_text = (
        last_turn_row["original_final_text"]
        if last_turn_row["edited_final_text"] is not None
        and last_turn_row["edited_final_text"] != last_turn_row["original_final_text"]
        else None
    )
    collect_meta = {
        "note": body.note,
        "turns": [
            {
                "turn": t["turn"],
                "model": t["model"],
                "personalization": t["personalization"],
                "memory": t["memory"],
                "skill": t["skill"],
                "original_final_text": t["original_final_text"] if t in edits else None,
            }
            for t in turns
        ],
    }

    harness_hash, system_prompt, tools = await live_harness()
    example_id = await asave_collector_example(
        thread_id=thread_id,
        turn=turn,
        user_id=user_id,
        cap=cap,
        harness_hash=harness_hash,
        system_prompt=system_prompt,
        tools=tools,
        deepagents_version=_deepagents_version(),
        status="accepted" if edited else "pending",
        edited=edited,
        original_final_text=original_final_text,
        collect_meta=collect_meta,
    )
    logger.info(
        f"[collector] recorded example={example_id} thread={thread_id} turn={turn} "
        f"edited={edited} by={user_id}"
    )
    return {
        "status": "accepted" if edited else "pending",
        "id": example_id,
        "edited": edited,
        "turn": turn,
    }
