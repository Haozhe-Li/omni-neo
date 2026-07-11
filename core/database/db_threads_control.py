"""
Database operations for threads_control table.

Table schema (post-migration):
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
"""

import logging
from core.database.postgresql_saver import sync_pool as pool
from core.utils import redis_sources, vector_sources

logger = logging.getLogger(__name__)


def get_thread_owner(thread_id: str) -> str | None:
    """
    Return the user_id that owns this thread.
    Returns None if the thread doesn't exist yet (new thread) or is unclaimed (NULL).
    """
    sql = "SELECT user_id FROM threads_control WHERE thread_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id,))
                row = cur.fetchone()
                if row is None:
                    return None  # Thread not registered yet – treat as unclaimed
                return row["user_id"]  # May itself be None (unclaimed legacy thread)
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_owner error: {e}")
        return None


def upsert_thread(thread_id: str, user_id: str | None = None) -> None:
    """
    Insert a new thread_id into threads_control.
    If it already exists, claim it with user_id only if currently unclaimed.
    Called synchronously when GET /get_thread_id is requested.
    """
    sql = """
        INSERT INTO threads_control (thread_id, updated_at, is_pinned, user_id)
        VALUES (%s, NOW(), FALSE, %s)
        ON CONFLICT (thread_id) DO UPDATE
            SET user_id = COALESCE(threads_control.user_id, EXCLUDED.user_id)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
    except Exception as e:
        logger.error(f"[db_threads_control] upsert_thread error for {thread_id}: {e}")


def touch_thread(thread_id: str, user_id: str | None = None) -> None:
    """
    Update the updated_at timestamp for an existing thread_id.
    If user_id is provided and the row is currently unclaimed, claim it.
    Called asynchronously (fire-and-forget) from /chat and /light_chat.
    """
    sql = """
        INSERT INTO threads_control (thread_id, updated_at, is_pinned, user_id)
        VALUES (%s, NOW(), FALSE, %s)
        ON CONFLICT (thread_id) DO UPDATE
            SET updated_at = NOW(),
                user_id = COALESCE(threads_control.user_id, EXCLUDED.user_id)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
    except Exception as e:
        logger.error(f"[db_threads_control] touch_thread error for {thread_id}: {e}")


def pin_thread(thread_id: str, is_pinned: bool) -> bool:
    """Toggle the pinned state of a thread. Returns True if the row was found."""
    sql = "UPDATE threads_control SET is_pinned = %s WHERE thread_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (is_pinned, thread_id))
                return cur.rowcount > 0
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
    sql = "SELECT thread_id FROM threads_control WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return [r["thread_id"] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"[db_threads_control] get_thread_ids_owned_by_user error: {e}")
        return []


def delete_thread(thread_id: str) -> bool:
    """
    Hard-delete a thread from threads_control AND all LangGraph checkpoint tables
    (checkpoints, checkpoint_blobs, checkpoint_writes).
    Ownership should be verified by the caller before invoking this.
    """
    # LangGraph stores state across three tables that all use thread_id.
    checkpoint_tables = ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for table in checkpoint_tables:
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
                cur.execute(
                    "DELETE FROM threads_control WHERE thread_id = %s", (thread_id,)
                )
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
    Hard-delete multiple threads from threads_control AND all LangGraph checkpoint
    tables in one round trip. Ownership must already be verified by the caller
    (pass only ids confirmed deleted from user_threads).
    """
    if not thread_ids:
        return
    checkpoint_tables = ["checkpoint_writes", "checkpoint_blobs", "checkpoints"]
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for table in checkpoint_tables:
                    cur.execute(f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (thread_ids,))
                cur.execute(
                    "DELETE FROM threads_control WHERE thread_id = ANY(%s)", (thread_ids,)
                )
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
    sql = "UPDATE threads_control SET user_id = %s WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (new_user_id, old_user_id))
                return cur.rowcount
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
    guest_sql = """
        DELETE FROM threads_control
        WHERE is_pinned = FALSE
          AND updated_at < NOW() - INTERVAL '3 days'
          AND (user_id IS NULL OR user_id LIKE 'guest_%')
    """
    user_sql = """
        DELETE FROM threads_control
        WHERE is_pinned = FALSE
          AND updated_at < NOW() - INTERVAL '90 days'
          AND user_id IS NOT NULL
          AND user_id NOT LIKE 'guest_%'
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(guest_sql)
                guest_deleted = cur.rowcount
                cur.execute(user_sql)
                user_deleted = cur.rowcount
                logger.info(
                    f"[db_threads_control] cleanup: {guest_deleted} guest threads "
                    f"(3d), {user_deleted} user threads (90d) deleted"
                )
    except Exception as e:
        logger.error(f"[db_threads_control] cleanup_old_threads error: {e}")
