"""
Database operations for threads_control table (over Supabase PostgREST, HTTP).

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE threads_control (
        thread_id TEXT PRIMARY KEY,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        is_pinned BOOLEAN DEFAULT FALSE,
        user_id VARCHAR(255)   -- NULL=unclaimed, 'guest_xxx'=guest, Clerk id=user
    );

Retention policy:
    - guest (user_id IS NULL or LIKE 'guest_%'): 3 days
    - logged-in user:                            90 days
    - pinned threads:                            never auto-deleted

LangGraph checkpoint state now lives in Upstash Redis (see checkpointer.py),
not Postgres tables, so thread deletion clears it via the sync Upstash saver's
`delete_thread` instead of `DELETE FROM checkpoints`.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

from core.database.supabase_client import supabase, get_async_supabase, utcnow_iso
from core.database.checkpointer import sync_checkpointer
from core.utils import redis_sources, vector_sources

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process owner cache
# ---------------------------------------------------------------------------
# Thread ownership is read on the hot path of every chat/rewind/stop/check_source
# request but changes almost never (set once at claim, only altered by
# guest-merge or delete). Cache it per-process with a short TTL so those reads
# usually skip the Supabase round trip. Writes below invalidate eagerly, so the
# only staleness window is a claim/merge racing an in-flight cached read — and
# ownership only ever becomes *more* restrictive, bounded by the TTL.
_OWNER_TTL = 30.0
_owner_cache: dict[str, tuple[str | None, float]] = {}
_owner_lock = threading.Lock()


def _owner_cache_get(thread_id: str) -> tuple[str | None] | None:
    """Return a 1-tuple (owner,) on a live hit, or None on miss/expiry.
    The tuple wrapper lets a cached owner of None be distinguished from a miss."""
    ent = _owner_cache.get(thread_id)
    if ent is not None and ent[1] > time.monotonic():
        return (ent[0],)
    return None


def _owner_cache_put(thread_id: str, owner: str | None) -> None:
    with _owner_lock:
        _owner_cache[thread_id] = (owner, time.monotonic() + _OWNER_TTL)


def _owner_cache_invalidate(thread_id: str) -> None:
    with _owner_lock:
        _owner_cache.pop(thread_id, None)


def _owner_cache_clear() -> None:
    with _owner_lock:
        _owner_cache.clear()


def get_thread_owner(thread_id: str) -> str | None:
    """
    Return the user_id that owns this thread.
    Returns None if the thread doesn't exist yet (new thread) or is unclaimed (NULL).
    """
    cached = _owner_cache_get(thread_id)
    if cached is not None:
        return cached[0]
    try:
        res = (
            supabase.table("threads_control")
            .select("user_id")
            .eq("thread_id", thread_id)
            .limit(1)
            .execute()
        )
        owner = res.data[0]["user_id"] if res.data else None  # None = unclaimed/new
        _owner_cache_put(thread_id, owner)
        return owner
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_owner error: {e}")
        return None


async def get_thread_owner_async(thread_id: str) -> str | None:
    """True-async owner lookup for the hot chat path (access checks run on the
    event loop). Same semantics as get_thread_owner, same in-process cache."""
    cached = _owner_cache_get(thread_id)
    if cached is not None:
        return cached[0]
    try:
        sb = await get_async_supabase()
        res = (
            await sb.table("threads_control")
            .select("user_id")
            .eq("thread_id", thread_id)
            .limit(1)
            .execute()
        )
        owner = res.data[0]["user_id"] if res.data else None
        _owner_cache_put(thread_id, owner)
        return owner
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_owner_async error: {e}")
        return None


def upsert_thread(thread_id: str, user_id: str | None = None) -> None:
    """
    Insert a new thread_id into threads_control.
    If it already exists, claim it with user_id only if currently unclaimed.
    Called synchronously when GET /get_thread_id is requested.
    """
    try:
        existing = (
            supabase.table("threads_control")
            .select("user_id")
            .eq("thread_id", thread_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            supabase.table("threads_control").upsert(
                {
                    "thread_id": thread_id, "user_id": user_id,
                    "is_pinned": False, "updated_at": utcnow_iso(),
                },
                on_conflict="thread_id", ignore_duplicates=True,
            ).execute()
        elif user_id is not None and existing.data[0]["user_id"] is None:
            # Claim an unclaimed row without disturbing updated_at.
            supabase.table("threads_control").update({"user_id": user_id}).eq(
                "thread_id", thread_id
            ).is_("user_id", "null").execute()
        _owner_cache_invalidate(thread_id)  # ownership may have just changed
    except Exception as e:
        logger.error(f"[db_threads_control] upsert_thread error for {thread_id}: {e}")


def touch_thread(thread_id: str, user_id: str | None = None) -> None:
    """
    Update the updated_at timestamp for an existing thread_id.
    If user_id is provided and the row is currently unclaimed, claim it.
    Called asynchronously (fire-and-forget) from /chat and /light_chat.
    """
    try:
        existing = (
            supabase.table("threads_control")
            .select("user_id")
            .eq("thread_id", thread_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            supabase.table("threads_control").upsert(
                {
                    "thread_id": thread_id, "user_id": user_id,
                    "is_pinned": False, "updated_at": utcnow_iso(),
                },
                on_conflict="thread_id", ignore_duplicates=True,
            ).execute()
            _owner_cache_invalidate(thread_id)
            return
        payload = {"updated_at": utcnow_iso()}
        if user_id is not None and existing.data[0]["user_id"] is None:
            payload["user_id"] = user_id
            _owner_cache_invalidate(thread_id)  # claimed → drop stale unclaimed entry
        supabase.table("threads_control").update(payload).eq(
            "thread_id", thread_id
        ).execute()
    except Exception as e:
        logger.error(f"[db_threads_control] touch_thread error for {thread_id}: {e}")


def pin_thread(thread_id: str, is_pinned: bool) -> bool:
    """Toggle the pinned state of a thread. Returns True if the row was found."""
    try:
        res = supabase.table("threads_control").update({"is_pinned": is_pinned}).eq(
            "thread_id", thread_id
        ).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_threads_control] pin_thread error for {thread_id}: {e}")
        return False


def get_thread_ids_owned_by_user(user_id: str) -> list[str]:
    """
    Return every thread_id in threads_control claimed by this user.
    Used alongside user_threads' own row set when purging an entire account,
    since a thread can in principle exist here without a user_threads row
    (e.g. created but never synced with a title).
    """
    try:
        res = (
            supabase.table("threads_control")
            .select("thread_id")
            .eq("user_id", user_id)
            .execute()
        )
        return [r["thread_id"] for r in res.data]
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_ids_owned_by_user error: {e}")
        return []


def _delete_checkpoint_state(thread_id: str) -> None:
    """Clear a thread's LangGraph checkpoint state from Upstash Redis."""
    try:
        sync_checkpointer.delete_thread(thread_id)
    except Exception as e:
        logger.error(f"[db_threads_control] checkpoint cleanup error for {thread_id}: {e}")


def delete_thread(thread_id: str) -> bool:
    """
    Hard-delete a thread from threads_control AND its LangGraph checkpoint state.
    Ownership should be verified by the caller before invoking this.
    """
    try:
        supabase.table("threads_control").delete().eq("thread_id", thread_id).execute()
        _owner_cache_invalidate(thread_id)
        _delete_checkpoint_state(thread_id)
        try:
            redis_sources.delete_thread_sources(thread_id)
        except Exception as e:
            logger.error(f"[db_threads_control] redis source cleanup error for {thread_id}: {e}")
        try:
            vector_sources.delete_thread_vectors(thread_id)
        except Exception as e:
            logger.error(f"[db_threads_control] vector source cleanup error for {thread_id}: {e}")
        return True
    except Exception as e:
        logger.error(f"[db_threads_control] delete_thread error for {thread_id}: {e}")
        return False


def delete_threads_bulk(thread_ids: list[str]) -> None:
    """
    Hard-delete multiple threads from threads_control AND their LangGraph
    checkpoint state. Ownership must already be verified by the caller
    (pass only ids confirmed deleted from user_threads).
    """
    if not thread_ids:
        return
    try:
        supabase.table("threads_control").delete().in_("thread_id", thread_ids).execute()
        for thread_id in thread_ids:
            _owner_cache_invalidate(thread_id)
            _delete_checkpoint_state(thread_id)
        try:
            redis_sources.delete_threads_sources_bulk(thread_ids)
        except Exception as e:
            logger.error(f"[db_threads_control] redis source cleanup error for {thread_ids}: {e}")
        for thread_id in thread_ids:
            try:
                vector_sources.delete_thread_vectors(thread_id)
            except Exception as e:
                logger.error(f"[db_threads_control] vector source cleanup error for {thread_id}: {e}")
    except Exception as e:
        logger.error(f"[db_threads_control] delete_threads_bulk error for {thread_ids}: {e}")


def reassign_threads_user(old_user_id: str, new_user_id: str) -> int:
    """
    Update user_id in threads_control when guest threads are merged into a real account.
    Returns the number of rows updated.
    """
    try:
        res = supabase.table("threads_control").update({"user_id": new_user_id}).eq(
            "user_id", old_user_id
        ).execute()
        # Bulk owner change across an unknown set of thread_ids — clear the whole
        # cache (guest-merge is rare, so this is cheap enough).
        _owner_cache_clear()
        return len(res.data or [])
    except Exception as e:
        logger.error(f"[db_threads_control] reassign_threads_user error: {e}")
        return 0


def cleanup_old_threads() -> None:
    """
    Differential retention cleanup:
      - guests (user_id IS NULL or starts with 'guest_'): deleted after 3 days
      - logged-in users: deleted after 90 days
      - pinned threads: never deleted
    Called asynchronously (fire-and-forget) on GET /health.
    """
    now = datetime.now(timezone.utc)
    cutoff_guest = (now - timedelta(days=3)).isoformat()
    cutoff_user = (now - timedelta(days=90)).isoformat()
    try:
        guest_res = (
            supabase.table("threads_control")
            .delete()
            .eq("is_pinned", False)
            .lt("updated_at", cutoff_guest)
            .or_("user_id.is.null,user_id.like.guest_*")
            .execute()
        )
        user_res = (
            supabase.table("threads_control")
            .delete()
            .eq("is_pinned", False)
            .lt("updated_at", cutoff_user)
            .not_.is_("user_id", "null")
            .not_.like("user_id", "guest_*")
            .execute()
        )
        logger.info(
            f"[db_threads_control] cleanup: {len(guest_res.data or [])} guest threads "
            f"(3d), {len(user_res.data or [])} user threads (90d) deleted"
        )
    except Exception as e:
        logger.error(f"[db_threads_control] cleanup_old_threads error: {e}")
