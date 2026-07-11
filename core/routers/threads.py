import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.auth import get_current_user, get_optional_user
from core.redis_stream import stream_is_generating, stream_get_status, stream_read
from core.database.db_user_threads import (
    get_threads_for_user,
    get_thread_messages,
    upsert_thread_messages,
    register_thread,
    update_thread_title,
    delete_user_thread,
    delete_user_threads_bulk,
    pin_user_thread,
    count_user_threads,
    search_user_threads,
    GUEST_MAX_THREADS,
)
from core.database.db_threads_control import (
    upsert_thread,
    delete_thread as delete_thread_state,
    delete_threads_bulk as delete_threads_state_bulk,
    pin_thread as pin_thread_state,
)
from core.routers.state import assert_thread_access

router = APIRouter(tags=["threads"])


@router.get("/get_thread_id")
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


@router.get("/api/threads")
def api_get_threads(user_id: str = Depends(get_current_user)):
    """Return the list of threads owned by the current user."""
    threads = get_threads_for_user(user_id)
    # Serialise datetime objects so they become JSON-safe strings
    for t in threads:
        if hasattr(t.get("updated_at"), "isoformat"):
            t["updated_at"] = t["updated_at"].isoformat()
    return {"threads": threads}


@router.get("/api/threads/search")
def api_search_threads(
    q: str,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
):
    """Fuzzy-search the current user's own threads by title and message content."""
    q = q.strip()
    if not q:
        return {"results": []}
    if len(q) > 200:
        raise HTTPException(status_code=400, detail="Query too long.")
    results = search_user_threads(user_id, q, limit)
    for r in results:
        if hasattr(r.get("updated_at"), "isoformat"):
            r["updated_at"] = r["updated_at"].isoformat()
    return {"results": results}


class BatchDeleteRequest(BaseModel):
    thread_ids: list[str]


@router.post("/api/threads/batch-delete")
def api_batch_delete_threads(
    body: BatchDeleteRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Hard-delete multiple threads owned by the current user in one call.
    Ids that don't exist or aren't owned by the caller are skipped, not errored.
    """
    thread_ids = list(dict.fromkeys(body.thread_ids))[:100]
    if not thread_ids:
        return {"deleted": [], "not_found": []}
    deleted = delete_user_threads_bulk(thread_ids, user_id)
    delete_threads_state_bulk(deleted)
    not_found = [t for t in thread_ids if t not in deleted]
    return {"deleted": deleted, "not_found": not_found}


@router.get("/api/threads/{thread_id}")
async def api_get_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    """Return the stored ui_messages for a single thread, with generation status."""
    messages = get_thread_messages(thread_id, user_id)
    if messages is None:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    generating = await stream_is_generating(thread_id)
    return {"messages": messages, "is_generating": generating}


@router.get("/api/threads/{thread_id}/stream")
async def api_reconnect_stream(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """Reconnect SSE endpoint: replays all buffered events then continues live.

    Call this when returning to a thread where is_generating is true.
    The stream ends with a `done` (or `error`) event, identical to /chat.
    """
    assert_thread_access(thread_id, user_id)
    status = await stream_get_status(thread_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No active stream for this thread.")
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(
        stream_read(thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


class SyncThreadRequest(BaseModel):
    messages: list
    title: str | None = None


@router.post("/api/threads/{thread_id}/sync")
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


@router.delete("/api/threads/{thread_id}")
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


@router.patch("/api/threads/{thread_id}/title")
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


@router.patch("/api/threads/{thread_id}/pin")
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
