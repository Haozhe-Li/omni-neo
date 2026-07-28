from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user
from core.database.db_user_threads import (
    merge_guest_to_user,
    delete_all_threads_for_user,
)
from core.database.db_threads_control import (
    reassign_threads_user,
    get_thread_ids_owned_by_user,
    delete_threads_bulk as delete_threads_state_bulk,
)
from core.database.db_user_memories import migrate_guest_memory, delete_user_memory
from core.database.db_user_files import get_user_file_buckets, delete_user_files
from core.database.db_user_usage import get_usage, delete_user_usage
from core.database.db_redeem_codes import redeem_code
from core.RAG.file_parser import delete_user_uploads_from_s3

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/usage")
def api_get_usage(user_id: str = Depends(get_current_user)):
    """
    Return the caller's current credit usage: today's and this calendar
    month's totals, limits, remaining balance, per-mode credit cost, and the
    next reset times. Works for both guests and signed-in users — limits
    differ by tier, everything else about the shape is the same.
    """
    return get_usage(user_id)


class RedeemRequest(BaseModel):
    code: str


# Failure reason -> HTTP status. The body always carries the reason verbatim as
# `detail.error` so the frontend can pick its own copy per case rather than
# parsing a sentence.
_REDEEM_STATUS_CODES = {
    "invalid_code": 404,
    "code_expired": 410,
    "code_exhausted": 409,
    "already_redeemed": 409,
    "error": 500,
}


@router.post("/redeem")
def api_redeem_code(
    body: RedeemRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Redeem a prepaid credit code, granting extra credits that are spent before
    the daily/monthly allowances and don't count against either.

    Signed-in only. A guest id is a client-generated uuid, so a guest could
    redeem the same code from an unlimited supply of fresh identities and the
    UNIQUE (code, user_id) guard would never see a duplicate. There's a second
    reason too: /users/merge deletes the guest's usage row on sign-in, which
    would silently destroy the credits they just redeemed.
    """
    if user_id.startswith("guest_"):
        raise HTTPException(status_code=403, detail={"error": "sign_in_required"})
    result = redeem_code(user_id, body.code)
    if result["status"] != "ok":
        raise HTTPException(
            status_code=_REDEEM_STATUS_CODES.get(result["status"], 400),
            detail={"error": result["status"]},
        )
    return result


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
    # Signing in is an upgrade, not a punishment: drop the guest's usage
    # rather than carrying its (much lower) tier's counters over.
    delete_user_usage(body.guest_id)
    return {"status": "merged", "threads_migrated": count}


@router.delete("/user-data")
def api_delete_all_user_data(user_id: str = Depends(get_current_user)):
    """
    Permanently erase every piece of data associated with this user_id:
    all threads (LangGraph checkpoints, cached citations in Redis, and the
    Upstash vector index), every uploaded file (DB rows + S3 objects), the
    long-term memory document.

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

    # important: do not delete usage tracking, otherwise the user will be able to create a new account and get a fresh usage allowance.
    # delete_user_usage(user_id)

    return {
        "status": "deleted",
        "threads_deleted": len(thread_ids),
        "files_deleted": files_deleted,
        "s3_objects_deleted": objects_deleted,
        "memory_deleted": memory_deleted,
    }
