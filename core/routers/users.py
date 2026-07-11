from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user, GUEST_DAILY_LIMIT
from core.database.db_user_threads import (
    get_guest_usage_today as _get_guest_usage_today,
    merge_guest_to_user,
    delete_all_threads_for_user,
    delete_guest_usage,
)
from core.database.db_threads_control import (
    reassign_threads_user,
    get_thread_ids_owned_by_user,
    delete_threads_bulk as delete_threads_state_bulk,
)
from core.database.db_user_memories import migrate_guest_memory, delete_user_memory
from core.database.db_user_files import get_user_file_buckets, delete_user_files
from core.RAG.file_parser import delete_user_uploads_from_s3

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/guests/daily-quota")
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


class MergeRequest(BaseModel):
    guest_id: str


@router.post("/users/merge")
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
    migrate_guest_memory(user_id, body.guest_id)
    return {"status": "merged", "threads_migrated": count}


@router.delete("/user-data")
def api_delete_all_user_data(user_id: str = Depends(get_current_user)):
    """
    Permanently erase every piece of data associated with this user_id:
    all threads (LangGraph checkpoints, cached citations in Redis, and the
    Upstash vector index), every uploaded file (DB rows + S3 objects), the
    long-term memory document, and — for guests — the daily quota counter.

    Irreversible. Published "pages" live in the frontend's own Redis (Upstash)
    and are purged separately by the Next.js /api/unpublish-all route.
    """
    # Union of both tables' id sets: a thread can in principle exist in
    # threads_control without ever having synced a user_threads row.
    thread_ids = list(set(delete_all_threads_for_user(user_id)) | set(get_thread_ids_owned_by_user(user_id)))
    delete_threads_state_bulk(thread_ids)

    buckets = get_user_file_buckets(user_id)
    files_deleted = delete_user_files(user_id)
    objects_deleted = delete_user_uploads_from_s3(user_id, buckets) if buckets else 0

    memory_deleted = delete_user_memory(user_id)

    if user_id.startswith("guest_"):
        delete_guest_usage(user_id)

    return {
        "status": "deleted",
        "threads_deleted": len(thread_ids),
        "files_deleted": files_deleted,
        "s3_objects_deleted": objects_deleted,
        "memory_deleted": memory_deleted,
    }
