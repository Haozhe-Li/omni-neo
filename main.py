import dotenv

dotenv.load_dotenv()

import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_core.messages import HumanMessage as LCHumanMessage

from core.agent import SYSTEM_PROMPTS, get_agent, initialize_agents
from core.stream import run_agent_stream, build_message_content
from core.redis_stream import (
    stream_write,
    stream_set_status,
    stream_get_status,
    stream_is_generating,
    stream_expire,
    stream_reset,
    stream_read,
    STREAM_TTL_ACTIVE,
    STREAM_TTL_DONE,
)
from core.database.postgresql_saver import setup_checkpointer, teardown_checkpointer
from core.utils.data_model import Personalization
from core.prompt_guard import register_sensitive_prompts
from core.utils.data_model import (
    QueryRequest,
    UpdateMemoriesRequest,
    AutoCompleteRequest,
)
from core.utils.utils import format_personalization
from core.auto_complete import auto_complete
from core.auth import (
    get_current_user,
    get_optional_user,
    GUEST_DAILY_LIMIT,
)
from core.get_title import get_title
from core.memories_update_llm import get_update_memories
from core.audio_sst import get_text_from_audio
from core.database.db_user_threads import (
    get_guest_usage_today as _get_guest_usage_today,
    check_and_increment_guest_usage,
)
from core.database.db_user_threads import (
    get_threads_for_user,
    get_thread_messages,
    upsert_thread_messages,
    register_thread,
    update_thread_title,
    delete_user_thread,
    pin_user_thread,
    merge_guest_to_user,
    count_user_threads,
    GUEST_MAX_THREADS,
)
from core.database.db_threads_control import (
    upsert_thread,
    touch_thread,
    cleanup_old_threads,
    delete_thread as delete_thread_state,
    reassign_threads_user,
    pin_thread as pin_thread_state,
    get_thread_owner,
)
from core.database.db_user_files import (
    create_pending_file,
    setup_user_files_table,
)
from core.RAG.file_parser import (
    get_put_presigned_url,
    process_uploaded_file,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_checkpointer()
    setup_user_files_table()
    initialize_agents()
    yield
    await teardown_checkpointer()


app = FastAPI(title="Omni Agent API", lifespan=lifespan)

register_sensitive_prompts(SYSTEM_PROMPTS)

# Thread pool for fire-and-forget blocking DB calls
_db_executor = ThreadPoolExecutor(max_workers=4)

# Active generation cancellation events keyed by thread_id.
_cancellation_events: dict[str, asyncio.Event] = {}

# Background generation tasks — held here so GC doesn't kill them when the HTTP
# connection drops.  Tasks remove themselves on completion.
_generation_tasks: dict[str, asyncio.Task] = {}

# Grace period to let a connected client sync the finished turn to Postgres
# before the backend writes its own fallback copy (avoids a duplicate write).
PERSIST_GRACE_SECONDS = 3.0


async def _generate_background(
    thread_id: str,
    user_id: str,
    query: str,
    mode: str,
    personalization: str,
    attached_file_ids: list | None,
    user_location: str | None,
    user_local_datetime: str | None,
    cancel_event: asyncio.Event | None,
) -> None:
    """Run the agent, buffer every SSE event to Redis, and save to Postgres on done."""
    # Accumulate message fields for the final Postgres upsert.
    text = ""
    steps: list[dict] = []
    sources: list[dict] = []
    artifacts: list[dict] = []
    widgets: list[dict] = []

    try:
        async for event_str in run_agent_stream(
            query=query,
            thread_id=thread_id,
            mode=mode,
            personalization=personalization,
            attached_file_ids=attached_file_ids,
            user_location=user_location,
            user_local_datetime=user_local_datetime,
            cancellation_event=cancel_event,
        ):
            await stream_write(thread_id, event_str)

            try:
                ev = json.loads(event_str[6:])  # strip "data: "
            except Exception:
                continue

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

        # Shrink TTL now that generation is complete.
        await stream_expire(thread_id)
        await stream_set_status(thread_id, "done", STREAM_TTL_DONE)

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
        await stream_set_status(thread_id, "done", STREAM_TTL_DONE)
        await stream_expire(thread_id)
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        error_event = f'data: {json.dumps({"type": "error", "content": str(exc)})}\n\n'
        await stream_write(thread_id, error_event)
        await stream_set_status(thread_id, "error", STREAM_TTL_DONE)
        await stream_expire(thread_id)
    finally:
        _cancellation_events.pop(thread_id, None)
        _generation_tasks.pop(thread_id, None)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _assert_thread_access(thread_id: str | None, user_id: str) -> None:
    """
    Verify the requesting user is allowed to access the given thread.
    Raises HTTP 403 if the thread is claimed by a *different* user.
    Unclaimed threads (owner is None) are accessible by anyone.
    """
    if not thread_id:
        return
    owner = get_thread_owner(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="Thread access denied.")


# ---------------------------------------------------------------------------
# Chat – the single all-around agent endpoint
# ---------------------------------------------------------------------------


@app.post("/chat")
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

    _assert_thread_access(request.thread_id, user_id)
    thread_id = request.thread_id

    if thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, thread_id, user_id)

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

    personalization_str = format_personalization(request.personalization)
    p = request.personalization
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

    if thread_id:
        # ── background task mode: survives HTTP disconnect ─────────────────
        cancel_event = asyncio.Event()
        _cancellation_events[thread_id] = cancel_event
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
                cancel_event=cancel_event,
            )
        )
        _generation_tasks[thread_id] = task
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


@app.post("/api/threads/{thread_id}/rewind")
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
    _assert_thread_access(thread_id, user_id)

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

    if body.new_query is not None:
        # Edit mode: replace the last HumanMessage in-place (same id → add_messages
        # reducer treats it as an update, not an append).
        personalization_str = format_personalization(body.personalization)
        new_content = await asyncio.to_thread(
            build_message_content,
            body.new_query, personalization_str, body.attached_file_ids,
        )
        last_human = target.values["messages"][-1]
        updated_msg = LCHumanMessage(id=last_human.id, content=new_content)
        rewind_config = await agent.aupdate_state(target.config, {"messages": [updated_msg]})
    else:
        # Regenerate mode: replay from the existing checkpoint as-is.
        rewind_config = target.config

    p = body.personalization
    personalization_str = format_personalization(p)
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    cancel_event = asyncio.Event()
    _cancellation_events[thread_id] = cancel_event

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
                rewind_config=rewind_config,
                cancellation_event=cancel_event,
            ):
                yield chunk
        finally:
            _cancellation_events.pop(thread_id, None)

    return StreamingResponse(
        _rewind_stream_with_cleanup(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/api/threads/{thread_id}/stop")
async def stop_generation(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """Signal the active generation for this thread to stop."""
    _assert_thread_access(thread_id, user_id)
    event = _cancellation_events.get(thread_id)
    if event:
        event.set()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.post("/auto_complete")
async def api_auto_complete(request: AutoCompleteRequest):
    """Endpoint for text autocomplete."""
    try:
        results = auto_complete(request.text.strip())
        return {"texts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# File uploads (S3 presigned + async parsing)
# ---------------------------------------------------------------------------


class UploadUrlRequest(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    thread_id: str | None = None


@app.post("/api/upload/url")
def api_upload_url(
    request: UploadUrlRequest,
    user_id: str = Depends(get_current_user),
):
    # Use thread_id from request body; generate one only if frontend didn't provide it.
    thread_id = request.thread_id or str(uuid.uuid4())
    raw_file_id = str(uuid.uuid4())
    file_id = f"user_uploads/{user_id}/{raw_file_id}"
    s3_bucket = os.getenv("S3_BUCKET_NAME", "omni")

    if request.file_type.startswith("image/"):
        category = "image"
    elif (
        request.file_type == "application/pdf"
        or request.file_type.startswith("text/")
        or request.file_type
        in [
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-javascript",
            "application/x-python",
            "application/x-sh",
            "application/x-httpd-php",
            "application/yaml",
            "application/x-yaml",
        ]
    ):
        category = "document"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    create_pending_file(
        file_id=file_id,
        user_id=user_id,
        thread_id=thread_id,
        original_filename=request.filename,
        file_type=request.file_type,
        file_size_bytes=request.file_size_bytes,
        s3_bucket=s3_bucket,
        category=category,
    )

    url = get_put_presigned_url(s3_bucket, file_id, request.file_type)
    return {"upload_url": url, "file_id": file_id, "thread_id": thread_id}


@app.post("/api/upload/confirm")
def api_upload_confirm(file_id: str, user_id: str = Depends(get_current_user)):
    # Run the file parsing asynchronously without event loop collision in worker thread
    _db_executor.submit(process_uploaded_file, file_id)
    return {"status": "processing", "file_id": file_id}


# ---------------------------------------------------------------------------
# Misc agent utilities
# ---------------------------------------------------------------------------


@app.get("/get_thread_id")
def get_thread_id(user_id: str | None = Depends(get_optional_user)):
    """
    Generate a new thread ID and register it in both threads_control and user_threads.
    If auth headers are present the thread is immediately bound to the user.
    Guests are capped at GUEST_MAX_THREADS active threads.
    """
    if user_id and user_id.startswith("guest_"):
        if count_user_threads(user_id) >= GUEST_MAX_THREADS:
            raise HTTPException(
                status_code=429,
                detail=f"Guest accounts are limited to {GUEST_MAX_THREADS} threads. Please sign in for unlimited threads.",
            )
    thread_id = str(uuid.uuid4())
    upsert_thread(thread_id, user_id)
    if user_id:
        register_thread(thread_id, user_id)
    return thread_id


@app.post("/get_title")
def generate_title(request: QueryRequest):
    return get_title(request.query)


@app.post("/update_memories")
async def update_memories_api(request: UpdateMemoriesRequest):
    res = await get_update_memories(request.past_queries, request.past_memories)
    return res


@app.post("/api/sst")
async def speech_to_text_api(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are supported.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = await get_text_from_audio(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SST failed: {str(e)}")


@app.get("/health")
async def health():
    # Fire-and-forget: clean up stale threads asynchronously
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_db_executor, cleanup_old_threads)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# User threads – auth-gated endpoints
# ---------------------------------------------------------------------------


class SyncThreadRequest(BaseModel):
    messages: list
    title: str | None = None


class MergeRequest(BaseModel):
    guest_id: str


@app.get("/api/threads")
def api_get_threads(user_id: str = Depends(get_current_user)):
    """Return the list of threads owned by the current user."""
    threads = get_threads_for_user(user_id)
    # Serialise datetime objects so they become JSON-safe strings
    for t in threads:
        if hasattr(t.get("updated_at"), "isoformat"):
            t["updated_at"] = t["updated_at"].isoformat()
    return {"threads": threads}


@app.get("/api/threads/{thread_id}")
async def api_get_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    """Return the stored ui_messages for a single thread, with generation status."""
    messages = get_thread_messages(thread_id, user_id)
    if messages is None:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    generating = await stream_is_generating(thread_id)
    return {"messages": messages, "is_generating": generating}


@app.get("/api/threads/{thread_id}/stream")
async def api_reconnect_stream(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """Reconnect SSE endpoint: replays all buffered events then continues live.

    Call this when returning to a thread where is_generating is true.
    The stream ends with a `done` (or `error`) event, identical to /chat.
    """
    _assert_thread_access(thread_id, user_id)
    status = await stream_get_status(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No active stream for this thread.")
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(
        stream_read(thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/api/threads/{thread_id}/sync")
def api_sync_thread(
    thread_id: str,
    body: SyncThreadRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Upsert the ui_messages (and optionally the title) for a thread.
    The row is only written if it belongs to the requesting user.
    """
    ok = upsert_thread_messages(thread_id, user_id, body.messages)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    if body.title:
        title_ok = update_thread_title(thread_id, user_id, body.title)
        if not title_ok:
            raise HTTPException(
                status_code=404, detail="Thread not found or access denied."
            )
    return {"status": "success"}


@app.get("/api/guests/daily-quota")
def api_guest_daily_quota(user_id: str = Depends(get_current_user)):
    """
    Return the remaining daily quota for a guest user today.
    Signed-in users always get unlimited (-1).
    """
    if not user_id.startswith("guest_"):
        return {"daily_limit": -1, "used": 0, "remaining": -1}
    used = _get_guest_usage_today(user_id)
    remaining = max(GUEST_DAILY_LIMIT - used, 0)
    return {"daily_limit": GUEST_DAILY_LIMIT, "used": used, "remaining": remaining}


@app.post("/api/users/merge")
def api_merge_guest(
    body: MergeRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Migrate all threads from a guest account to the authenticated user.
    Updates both user_threads (UI state) and threads_control (LangGraph state).
    Must be called with a valid Bearer token (i.e. after sign-in).
    """
    if not body.guest_id.startswith("guest_"):
        raise HTTPException(status_code=400, detail="Invalid guest_id format.")
    if user_id.startswith("guest_"):
        raise HTTPException(status_code=403, detail="Must be signed in to merge.")
    count = merge_guest_to_user(user_id, body.guest_id)
    # Mirror the reassignment in threads_control so retention rules apply correctly
    reassign_threads_user(body.guest_id, user_id)
    return {"status": "merged", "threads_migrated": count}


@app.delete("/api/threads/{thread_id}")
def api_delete_thread(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Hard-delete a thread.
    Removes the UI record (user_threads) and the full LangGraph checkpoint state
    (checkpoints, checkpoint_blobs, checkpoint_writes, threads_control).
    Returns 404 if the thread doesn't belong to this user.
    """
    owned = delete_user_thread(thread_id, user_id)
    if not owned:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    # Clean up LangGraph state + threads_control row
    delete_thread_state(thread_id)
    return {"status": "deleted"}


class PatchTitleRequest(BaseModel):
    title: str


class PatchPinRequest(BaseModel):
    is_pinned: bool


@app.patch("/api/threads/{thread_id}/title")
def api_rename_thread(
    thread_id: str,
    body: PatchTitleRequest,
    user_id: str = Depends(get_current_user),
):
    """Rename a thread. Only the owning user can rename."""
    ok = update_thread_title(thread_id, user_id, body.title)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    return {"status": "updated"}


@app.patch("/api/threads/{thread_id}/pin")
def api_pin_thread(
    thread_id: str,
    body: PatchPinRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Pin or unpin a thread.
    Pinned threads are sorted to the top of the list and exempted from auto-cleanup.
    """
    ok = pin_user_thread(thread_id, user_id, body.is_pinned)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    # Mirror pin state in threads_control so cleanup respects it
    pin_thread_state(thread_id, body.is_pinned)
    return {"status": "updated", "is_pinned": body.is_pinned}
