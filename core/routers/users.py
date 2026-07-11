from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user, GUEST_DAILY_LIMIT
from core.database.db_user_threads import (
    get_guest_usage_today as _get_guest_usage_today,
    merge_guest_to_user,
)
from core.database.db_threads_control import reassign_threads_user

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
    return {"status": "merged", "threads_migrated": count}
