"""
Database operations for user_threads and guest_usage tables.

Table schemas (post-migration):
    CREATE TABLE IF NOT EXISTS user_threads (
        thread_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        ui_messages JSONB DEFAULT '[]',
        is_pinned BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_user_threads_user_id ON user_threads(user_id);

    CREATE TABLE IF NOT EXISTS guest_usage (
        guest_id VARCHAR(255) PRIMARY KEY,
        usage_date DATE NOT NULL,
        request_count INT DEFAULT 1
    );

Limits:
    - Guests: max GUEST_MAX_THREADS active threads
    - Logged-in users: no hard thread count limit
"""

import json
import logging
import os
from datetime import date

from core.database.postgresql_saver import sync_pool as pool

logger = logging.getLogger(__name__)

GUEST_MAX_THREADS: int = int(os.getenv("GUEST_MAX_THREADS", "5"))


# ---------------------------------------------------------------------------
# Thread listing / reading
# ---------------------------------------------------------------------------

def get_threads_for_user(user_id: str) -> list[dict]:
    """Return all threads belonging to a user, newest first."""
    sql = """
        SELECT thread_id, title, is_pinned, updated_at
        FROM user_threads
        WHERE user_id = %s
        ORDER BY is_pinned DESC, updated_at DESC
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[db_user_threads] get_threads_for_user error: {e}")
        return []


def get_thread_messages(thread_id: str, user_id: str) -> list | None:
    """Return ui_messages for a specific owned thread, or None if not found."""
    sql = """
        SELECT ui_messages
        FROM user_threads
        WHERE thread_id = %s AND user_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
                row = cur.fetchone()
                if row:
                    msgs = row["ui_messages"]
                    if isinstance(msgs, str):
                        return json.loads(msgs)
                    return msgs or []
        return None
    except Exception as e:
        logger.error(f"[db_user_threads] get_thread_messages error: {e}")
        return None


def count_user_threads(user_id: str) -> int:
    """Return the number of active threads for a user (used for guest cap)."""
    sql = "SELECT COUNT(*) AS cnt FROM user_threads WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
                return row["cnt"] if row else 0
    except Exception as e:
        logger.error(f"[db_user_threads] count_user_threads error: {e}")
        return 0


# ---------------------------------------------------------------------------
# Thread creation / update
# ---------------------------------------------------------------------------

def upsert_thread_messages(thread_id: str, user_id: str, messages: list) -> bool:
    """
    Insert or update a thread's ui_messages.
    Only updates if the row's user_id matches (prevents overwriting another user's data).
    """
    sql = """
        INSERT INTO user_threads (thread_id, user_id, ui_messages, updated_at)
        VALUES (%s, %s, %s::jsonb, NOW())
        ON CONFLICT (thread_id) DO UPDATE
            SET ui_messages = EXCLUDED.ui_messages,
                updated_at = NOW()
            WHERE user_threads.user_id = EXCLUDED.user_id
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id, json.dumps(messages)))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_threads] upsert_thread_messages error: {e}")
        return False


def register_thread(thread_id: str, user_id: str) -> bool:
    """
    Create a user_threads row at thread-creation time (called from /get_thread_id).
    No-op if the thread is already registered.
    """
    sql = """
        INSERT INTO user_threads (thread_id, user_id, ui_messages, updated_at)
        VALUES (%s, %s, '[]'::jsonb, NOW())
        ON CONFLICT (thread_id) DO NOTHING
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
        return True
    except Exception as e:
        logger.error(f"[db_user_threads] register_thread error: {e}")
        return False


def update_thread_title(thread_id: str, user_id: str, title: str) -> bool:
    """Update the title of a thread owned by the user."""
    sql = """
        UPDATE user_threads SET title = %s, updated_at = NOW()
        WHERE thread_id = %s AND user_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (title, thread_id, user_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_threads] update_thread_title error: {e}")
        return False


# ---------------------------------------------------------------------------
# Thread deletion
# ---------------------------------------------------------------------------

def delete_user_thread(thread_id: str, user_id: str) -> bool:
    """
    Delete a thread from user_threads if it belongs to the given user.
    Returns True if a row was deleted (ownership confirmed), False otherwise.
    The caller is responsible for also calling delete_thread() in db_threads_control
    to clean up the LangGraph state.
    """
    sql = "DELETE FROM user_threads WHERE thread_id = %s AND user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_threads] delete_user_thread error: {e}")
        return False


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------

def pin_user_thread(thread_id: str, user_id: str, is_pinned: bool) -> bool:
    """
    Set the is_pinned flag on a thread owned by the user.
    Returns True if the row was found and updated.
    """
    sql = """
        UPDATE user_threads SET is_pinned = %s
        WHERE thread_id = %s AND user_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (is_pinned, thread_id, user_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_threads] pin_user_thread error: {e}")
        return False


# ---------------------------------------------------------------------------
# Account merge (guest → real user)
# ---------------------------------------------------------------------------

def merge_guest_to_user(user_id: str, guest_id: str) -> int:
    """
    Re-assign all threads belonging to guest_id to user_id in user_threads.
    Returns the number of threads migrated.
    Call reassign_threads_user() in db_threads_control separately to
    also update the LangGraph-side table.
    """
    sql = "UPDATE user_threads SET user_id = %s WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id, guest_id))
                return cur.rowcount
    except Exception as e:
        logger.error(f"[db_user_threads] merge_guest_to_user error: {e}")
        return 0


# ---------------------------------------------------------------------------
# Guest rate limiting
# ---------------------------------------------------------------------------

def get_guest_usage_today(guest_id: str) -> int:
    """
    Read-only: return today's request count for a guest without incrementing.
    Returns 0 if no record exists for today.
    """
    today = date.today()
    sql = """
        SELECT request_count FROM guest_usage
        WHERE guest_id = %s AND usage_date = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (guest_id, today))
                row = cur.fetchone()
                return row["request_count"] if row else 0
    except Exception as e:
        logger.error(f"[db_user_threads] get_guest_usage_today error: {e}")
        return 0


def check_and_increment_guest_usage(guest_id: str) -> int:
    """
    Atomically increment (or reset-and-set) today's request count for a guest.
    Returns the updated request_count.
    """
    today = date.today()
    sql = """
        INSERT INTO guest_usage (guest_id, usage_date, request_count)
        VALUES (%s, %s, 1)
        ON CONFLICT (guest_id) DO UPDATE
            SET request_count = CASE
                    WHEN guest_usage.usage_date = EXCLUDED.usage_date
                    THEN guest_usage.request_count + 1
                    ELSE 1
                END,
                usage_date = EXCLUDED.usage_date
        RETURNING request_count
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (guest_id, today))
                row = cur.fetchone()
                return row["request_count"] if row else 1
    except Exception as e:
        logger.error(f"[db_user_threads] check_and_increment_guest_usage error: {e}")
        return 0


