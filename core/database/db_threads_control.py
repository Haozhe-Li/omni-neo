"""
Database operations for the threads_control table (direct Postgres, see pg.py).

Table schema: see schema.sql.

Retention policy:
    - guest (user_id IS NULL or starts with 'guest_'): 3 days
    - logged-in user:                                  90 days
    - pinned threads:                                  never auto-deleted

Deleting a threads_control row cascades to its user_threads row (ON DELETE
CASCADE), so retention removes a thread's history together with its ownership
record. LangGraph checkpoint state lives in the application's Redis (see
checkpointer.py), so thread deletion clears it via the sync Redis saver's
`delete_thread`.
"""

import logging
import threading
import time

from core.database import pg
from core.database.checkpointer import get_sync_checkpointer, delete_rewind_points
from core.utils import redis_sources, vector_sources
from core.utils.redis_client import get_redis

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


def owner_cache_hit(thread_id: str) -> bool:
    """Whether a live owner-cache entry exists (for timing/observability logs)."""
    return _owner_cache_get(thread_id) is not None


# ---------------------------------------------------------------------------
# In-process lock-state cache
# ---------------------------------------------------------------------------
# Separate from the owner cache above: it's read on the same hot path (chat/
# rewind, batched into the same asyncio.gather as the owner/charge/is_generating
# reads) but locking is a distinct, much rarer write, so keeping its own small
# cache avoids coupling the two reads' invalidation together.
_LOCK_TTL = 30.0
_lock_cache: dict[str, tuple[bool, str | None, float]] = {}
_lock_lock = threading.Lock()


def _lock_cache_get(thread_id: str) -> tuple[bool, str | None] | None:
    ent = _lock_cache.get(thread_id)
    if ent is not None and ent[2] > time.monotonic():
        return (ent[0], ent[1])
    return None


def _lock_cache_put(thread_id: str, is_locked: bool, locked_reason: str | None) -> None:
    with _lock_lock:
        _lock_cache[thread_id] = (is_locked, locked_reason, time.monotonic() + _LOCK_TTL)


def _lock_cache_invalidate(thread_id: str) -> None:
    with _lock_lock:
        _lock_cache.pop(thread_id, None)


async def get_thread_lock_state_async(thread_id: str) -> tuple[bool, str | None]:
    """Return (is_locked, locked_reason) for the hot chat/rewind path."""
    cached = _lock_cache_get(thread_id)
    if cached is not None:
        return cached
    try:
        row = await pg.afetch_one(
            "SELECT is_locked, locked_reason FROM threads_control WHERE thread_id = %s",
            (thread_id,),
        )
        is_locked = bool(row["is_locked"]) if row else False
        locked_reason = row["locked_reason"] if row else None
        _lock_cache_put(thread_id, is_locked, locked_reason)
        return is_locked, locked_reason
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_lock_state_async error: {e}")
        return False, None


def lock_thread_state(thread_id: str, reason: str) -> bool:
    """Set threads_control.is_locked (idempotent — a repeat call just refreshes
    locked_at/locked_reason). Returns True if the row was found and updated."""
    try:
        n = pg.execute(
            "UPDATE threads_control SET is_locked = TRUE, locked_reason = %s, locked_at = now() "
            "WHERE thread_id = %s",
            (reason, thread_id),
        )
        _lock_cache_invalidate(thread_id)
        return n > 0
    except Exception as e:
        logger.error(f"[db_threads_control] lock_thread_state error for {thread_id}: {e}")
        return False


def get_thread_owner(thread_id: str) -> str | None:
    """
    Return the user_id that owns this thread.
    Returns None if the thread doesn't exist yet (new thread) or is unclaimed (NULL).
    """
    cached = _owner_cache_get(thread_id)
    if cached is not None:
        return cached[0]
    try:
        row = pg.fetch_one("SELECT user_id FROM threads_control WHERE thread_id = %s", (thread_id,))
        owner = row["user_id"] if row else None  # None = unclaimed/new
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
        row = await pg.afetch_one(
            "SELECT user_id FROM threads_control WHERE thread_id = %s", (thread_id,)
        )
        owner = row["user_id"] if row else None
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
        pg.execute(
            "INSERT INTO threads_control (thread_id, user_id) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO UPDATE SET user_id = EXCLUDED.user_id "
            "WHERE threads_control.user_id IS NULL AND EXCLUDED.user_id IS NOT NULL",
            (thread_id, user_id),
        )
        _owner_cache_invalidate(thread_id)  # ownership may have just changed
    except Exception as e:
        logger.error(f"[db_threads_control] upsert_thread error for {thread_id}: {e}")


def touch_thread(thread_id: str, user_id: str | None = None) -> None:
    """
    Update the updated_at timestamp for an existing thread_id (creating the row if
    it is missing). If user_id is provided and the row is currently unclaimed,
    claim it. Called asynchronously (fire-and-forget) from /chat and /light_chat.
    """
    try:
        pg.execute(
            "INSERT INTO threads_control (thread_id, user_id) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO UPDATE SET updated_at = now(), "
            "user_id = COALESCE(threads_control.user_id, EXCLUDED.user_id)",
            (thread_id, user_id),
        )
        if user_id is not None:
            _owner_cache_invalidate(thread_id)  # may have claimed → drop stale unclaimed entry
    except Exception as e:
        logger.error(f"[db_threads_control] touch_thread error for {thread_id}: {e}")


def pin_thread(thread_id: str, is_pinned: bool) -> bool:
    """Toggle the pinned state of a thread. Returns True if the row was found."""
    try:
        return pg.execute(
            "UPDATE threads_control SET is_pinned = %s WHERE thread_id = %s", (is_pinned, thread_id)
        ) > 0
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
        rows = pg.fetch_all("SELECT thread_id FROM threads_control WHERE user_id = %s", (user_id,))
        return [r["thread_id"] for r in rows]
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_ids_owned_by_user error: {e}")
        return []


def _delete_checkpoint_state(thread_id: str) -> None:
    """Clear a thread's LangGraph checkpoint state from Redis."""
    try:
        get_sync_checkpointer().delete_thread(thread_id)
    except Exception as e:
        logger.error(f"[db_threads_control] checkpoint cleanup error for {thread_id}: {e}")
    try:
        delete_rewind_points(thread_id)
    except Exception as e:
        logger.error(f"[db_threads_control] rewind_points cleanup error for {thread_id}: {e}")


def delete_thread(thread_id: str) -> bool:
    """
    Hard-delete a thread from threads_control AND its LangGraph checkpoint state.
    Ownership should be verified by the caller before invoking this.
    """
    try:
        pg.execute("DELETE FROM threads_control WHERE thread_id = %s", (thread_id,))
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
        pg.execute("DELETE FROM threads_control WHERE thread_id = ANY(%s)", (list(thread_ids),))
        for thread_id in thread_ids:
            _owner_cache_invalidate(thread_id)
            _delete_checkpoint_state(thread_id)
        try:
            redis_sources.delete_threads_sources_bulk(thread_ids)
        except Exception as e:
            logger.error(f"[db_threads_control] redis source cleanup error for {thread_ids}: {e}")
        try:
            vector_sources.delete_threads_vectors(thread_ids)
        except Exception as e:
            logger.error(f"[db_threads_control] vector source cleanup error for {thread_ids}: {e}")
    except Exception as e:
        logger.error(f"[db_threads_control] delete_threads_bulk error for {thread_ids}: {e}")


def reassign_threads_user(old_user_id: str, new_user_id: str) -> int:
    """
    Update user_id in threads_control when guest threads are merged into a real account.
    Returns the number of rows updated.
    """
    try:
        n = pg.execute(
            "UPDATE threads_control SET user_id = %s WHERE user_id = %s", (new_user_id, old_user_id)
        )
        # Bulk owner change across an unknown set of thread_ids — clear the whole
        # cache (guest-merge is rare, so this is cheap enough).
        _owner_cache_clear()
        return n
    except Exception as e:
        logger.error(f"[db_threads_control] reassign_threads_user error: {e}")
        return 0


def cleanup_old_threads() -> None:
    """
    Differential retention cleanup, one DELETE:
      - guests (user_id IS NULL or starts with 'guest_'): deleted after 3 days
      - logged-in users: deleted after 90 days
      - pinned threads: never deleted
    Deleting the threads_control row cascades to user_threads.
    Called asynchronously (fire-and-forget) on GET /health.
    """
    try:
        rows = pg.fetch_all(
            """
            DELETE FROM threads_control
            WHERE NOT is_pinned AND (
                  ((user_id IS NULL OR starts_with(user_id, 'guest_'))
                       AND updated_at < now() - interval '3 days')
               OR (user_id IS NOT NULL AND NOT starts_with(user_id, 'guest_')
                       AND updated_at < now() - interval '90 days')
            )
            RETURNING thread_id, (user_id IS NULL OR starts_with(user_id, 'guest_')) AS guest
            """
        )
        n_guest = sum(1 for r in rows if r["guest"])
        logger.info(
            f"[db_threads_control] cleanup: {n_guest} guest threads (3d), "
            f"{len(rows) - n_guest} user threads (90d) deleted"
        )
    except Exception as e:
        logger.error(f"[db_threads_control] cleanup_old_threads error: {e}")
        return

    # Expiry deletes rows directly, so the vector index has to be told separately:
    # drop the chunks of exactly the threads that were just removed...
    try:
        vector_sources.delete_threads_vectors([r["thread_id"] for r in rows])
    except Exception as e:
        logger.error(f"[db_threads_control] vector cleanup after expiry failed: {e}")
    # ...and reconcile, which catches anything that step (or any other path) missed.
    reconcile_vector_index()


# Chunks younger than this are never treated as orphans: a thread's first sources can
# be indexed before its threads_control row is visible.
_VECTOR_ORPHAN_GRACE_SECONDS = 3600
_VECTOR_RECONCILE_INTERVAL_SECONDS = 3600


def reconcile_vector_index() -> int:
    """Delete the vector chunks of every thread that no longer exists in
    threads_control. Returns the number of orphaned threads removed.

    This is what guarantees the index cannot leak: a thread's chunks live exactly
    as long as its row, whichever way the row went away (expiry, explicit delete,
    a failed cleanup call). It runs from cleanup_old_threads, which fires on every
    GET /health, so it is throttled with a Redis lock to once per interval across
    all instances.
    """
    try:
        if not get_redis().set(
            "omni:vector_reconcile", "1", nx=True, ex=_VECTOR_RECONCILE_INTERVAL_SECONDS
        ):
            return 0
        indexed = vector_sources.list_indexed_thread_ids(_VECTOR_ORPHAN_GRACE_SECONDS)
        alive: set[str] = set()
        for i in range(0, len(indexed), 1000):
            rows = pg.fetch_all(
                "SELECT thread_id FROM threads_control WHERE thread_id = ANY(%s)",
                (indexed[i : i + 1000],),
            )
            alive.update(r["thread_id"] for r in rows)
        orphans = [t for t in indexed if t not in alive]
        if orphans:
            vector_sources.delete_threads_vectors(orphans)
        logger.info(
            f"[db_threads_control] vector reconcile: {len(indexed)} indexed threads, "
            f"{len(orphans)} orphaned and removed"
        )
        return len(orphans)
    except Exception as e:
        logger.error(f"[db_threads_control] vector reconcile failed: {e}")
        return 0
