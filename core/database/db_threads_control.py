"""
Database operations for threads_control table.

Table schema:
    CREATE TABLE threads_control (
        thread_id TEXT PRIMARY KEY,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        is_pinned BOOLEAN DEFAULT FALSE
    );
"""

import logging
from core.database.postgresql_saver import pool

logger = logging.getLogger(__name__)


def upsert_thread(thread_id: str) -> None:
    """
    Insert a new thread_id into threads_control.
    If it already exists, do nothing (keep existing updated_at and is_pinned).
    Called synchronously when GET /get_thread_id is requested.
    """
    sql = """
        INSERT INTO threads_control (thread_id, updated_at, is_pinned)
        VALUES (%s, NOW(), FALSE)
        ON CONFLICT (thread_id) DO NOTHING
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id,))
    except Exception as e:
        logger.error(f"[db_threads_control] upsert_thread error for {thread_id}: {e}")


def touch_thread(thread_id: str) -> None:
    """
    Update the updated_at timestamp for an existing thread_id.
    If the thread does not exist yet, insert it.
    This is called asynchronously (fire-and-forget) from /chat and /light_chat.
    """
    sql = """
        INSERT INTO threads_control (thread_id, updated_at, is_pinned)
        VALUES (%s, NOW(), FALSE)
        ON CONFLICT (thread_id) DO UPDATE
            SET updated_at = NOW()
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id,))
    except Exception as e:
        logger.error(f"[db_threads_control] touch_thread error for {thread_id}: {e}")


def cleanup_old_threads() -> None:
    """
    Delete threads that have not been updated in the last 3 days and are not pinned.
    Called asynchronously (fire-and-forget) on GET /health.
    """
    sql = """
        DELETE FROM threads_control
        WHERE updated_at < NOW() - INTERVAL '3 days'
          AND is_pinned = FALSE
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                logger.info(
                    f"[db_threads_control] cleanup_old_threads: deleted {cur.rowcount} rows"
                )
    except Exception as e:
        logger.error(f"[db_threads_control] cleanup_old_threads error: {e}")
