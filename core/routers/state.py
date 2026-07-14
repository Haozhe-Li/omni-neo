"""
Module-level state shared across routers.

These live outside any single router module because both `chat.py` (writes
cancellation events / touches threads on a worker) and `misc.py` (`/health`
cleanup) or `threads.py` (`stop`) need the *same* instances — not a copy each.
"""

from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import HTTPException

from core.database.db_threads_control import get_thread_owner, get_thread_owner_async

# Thread pool for fire-and-forget blocking DB calls
db_executor = ThreadPoolExecutor(max_workers=4)

# Active generation cancellation events keyed by thread_id.
cancellation_events: dict[str, asyncio.Event] = {}

# Background generation tasks — held here so GC doesn't kill them when the HTTP
# connection drops.  Tasks remove themselves on completion.
generation_tasks: dict[str, asyncio.Task] = {}

# Grace period to let a connected client sync the finished turn to Postgres
# before the backend writes its own fallback copy (avoids a duplicate write).
PERSIST_GRACE_SECONDS = 3.0


def assert_thread_access(thread_id: str | None, user_id: str) -> None:
    """
    Verify the requesting user is allowed to access the given thread.
    Raises HTTP 403 if the thread is claimed by a *different* user.
    Unclaimed threads (owner is None) are accessible by anyone.
    Sync variant — used by sync endpoints (threads.py); async endpoints should
    use assert_thread_access_async so the owner lookup doesn't block the loop.
    """
    if not thread_id:
        return
    owner = get_thread_owner(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="Thread access denied.")


async def assert_thread_access_async(thread_id: str | None, user_id: str) -> None:
    """Async variant of assert_thread_access for the chat request handlers."""
    if not thread_id:
        return
    owner = await get_thread_owner_async(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="Thread access denied.")
