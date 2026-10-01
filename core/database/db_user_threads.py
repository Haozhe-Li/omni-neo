"""
Database operations for the user_threads table (direct Postgres, see pg.py).

Table schema: see schema.sql.

Search is done in the database: substring matches on title and body are served by
pg_trgm GIN indexes, ranking is "title hit > body hit" plus trigram similarity of
the title, and the excerpt shown with each hit is cut in SQL (make_snippet) so
whole message bodies never travel to the app.

Limits:
    - Guests: max GUEST_MAX_THREADS active threads
    - Logged-in users: no hard thread count limit
"""

import json
import logging
import os

from psycopg.types.json import Jsonb

from core.database import pg

logger = logging.getLogger(__name__)

GUEST_MAX_THREADS: int = int(os.getenv("GUEST_MAX_THREADS", "500"))

# Caps how much text per thread gets indexed/stored for search, guarding
# against pathologically long threads bloating the stored search_text.
SEARCH_TEXT_MAX_CHARS = 50_000

# Scheduled-research threads (origin='scheduled_task') are surfaced only through
# /schedule_task's own run list, never in the regular chat sidebar or search.
# Voice threads (origin='voice') ARE included: once a voice call has any content
# synced (see core/voice/session.py) it reads like any other thread, tagged with
# its origin so the frontend can render its restricted composer.
_VISIBLE = "(origin IS NULL OR origin = 'voice')"


# ---------------------------------------------------------------------------
# Thread listing / reading
# ---------------------------------------------------------------------------

def get_threads_for_user(user_id: str) -> list[dict]:
    """Return all threads belonging to a user, pinned first, then newest first."""
    try:
        return pg.fetch_all(
            "SELECT thread_id, title, is_pinned, is_locked, origin, updated_at "
            f"FROM user_threads WHERE user_id = %s AND {_VISIBLE} "
            "ORDER BY is_pinned DESC, updated_at DESC",
            (user_id,),
        )
    except Exception as e:
        logger.error(f"[db_user_threads] get_threads_for_user error: {e}")
        return []


def get_thread_messages(thread_id: str, user_id: str) -> list | None:
    """Return ui_messages for a specific owned thread, or None if not found."""
    try:
        row = pg.fetch_one(
            "SELECT ui_messages FROM user_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        if not row:
            return None
        msgs = row["ui_messages"]
        if isinstance(msgs, str):
            return json.loads(msgs)
        return msgs or []
    except Exception as e:
        logger.error(f"[db_user_threads] get_thread_messages error: {e}")
        return None


def count_user_threads(user_id: str) -> int:
    """Return the number of active threads for a user (used for guest cap)."""
    try:
        row = pg.fetch_one("SELECT count(*) AS n FROM user_threads WHERE user_id = %s", (user_id,))
        return int(row["n"])
    except Exception as e:
        logger.error(f"[db_user_threads] count_user_threads error: {e}")
        return 0


# ---------------------------------------------------------------------------
# Search indexing helpers
# ---------------------------------------------------------------------------

def _extract_search_text(messages: list) -> str:
    """Flatten a ui_messages list into plain text for search indexing."""
    parts = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    text = "\n".join(parts)
    return text[:SEARCH_TEXT_MAX_CHARS]


# ---------------------------------------------------------------------------
# Thread creation / update
# ---------------------------------------------------------------------------

def upsert_thread_messages(thread_id: str, user_id: str, messages: list) -> bool:
    """
    Insert or update a thread's ui_messages (and derived search_text) in one
    statement. The update is guarded by ownership: a thread owned by a different
    user is never overwritten (the statement then touches no row and this returns
    False). The thread must already have its threads_control row (foreign key).
    """
    try:
        row = pg.fetch_one(
            """
            INSERT INTO user_threads (thread_id, user_id, ui_messages, search_text, updated_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (thread_id) DO UPDATE
               SET ui_messages = EXCLUDED.ui_messages,
                   search_text = EXCLUDED.search_text,
                   updated_at  = now()
             WHERE user_threads.user_id = EXCLUDED.user_id
            RETURNING thread_id
            """,
            (thread_id, user_id, Jsonb(messages), _extract_search_text(messages)),
        )
        return row is not None
    except Exception as e:
        logger.error(f"[db_user_threads] upsert_thread_messages error: {e}")
        return False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _like_pattern(term: str) -> str:
    """`%term%` with LIKE metacharacters in the term escaped."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def search_user_threads(user_id: str, query: str, limit: int = 20) -> list[dict]:
    """
    Search a user's own threads by title and message content.

    A thread matches when the query is a case-insensitive substring of its title
    or body. Matches are ordered by relevance — a title hit outranks a body-only
    hit, ties broken by trigram similarity of the title, then recency — and each
    carries a short excerpt of the body around the first occurrence.
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        return pg.fetch_all(
            f"""
            SELECT thread_id, title, is_pinned, is_locked, updated_at,
                   make_snippet(search_text, %(term)s) AS snippet
              FROM user_threads
             WHERE user_id = %(uid)s AND {_VISIBLE}
               AND (title ILIKE %(pat)s OR search_text ILIKE %(pat)s)
             ORDER BY (CASE WHEN title ILIKE %(pat)s THEN 1.0
                            WHEN search_text ILIKE %(pat)s THEN 0.5 ELSE 0 END)
                      + similarity(coalesce(title, ''), %(term)s) DESC,
                      updated_at DESC
             LIMIT %(limit)s
            """,
            {"term": query, "pat": _like_pattern(query), "uid": user_id, "limit": limit},
        )
    except Exception as e:
        logger.error(f"[db_user_threads] search_user_threads error: {e}")
        return []


def register_thread(thread_id: str, user_id: str, origin: str | None = None) -> bool:
    """
    Create a user_threads row at thread-creation time (called from /get_thread_id,
    and from the scheduled-task webhook with origin='scheduled_task'), together
    with its threads_control parent if that does not exist yet.
    No-op if the thread is already registered.
    """
    try:
        with pg.pool().connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO threads_control (thread_id, user_id) VALUES (%s, %s) "
                "ON CONFLICT (thread_id) DO NOTHING",
                (thread_id, user_id),
            )
            conn.execute(
                "INSERT INTO user_threads (thread_id, user_id, ui_messages, origin) "
                "VALUES (%s, %s, '[]', %s) ON CONFLICT (thread_id) DO NOTHING",
                (thread_id, user_id, origin),
            )
        return True
    except Exception as e:
        logger.error(f"[db_user_threads] register_thread error: {e}")
        return False


def update_thread_title(thread_id: str, user_id: str, title: str) -> bool:
    """Update the title of a thread owned by the user."""
    try:
        return pg.execute(
            "UPDATE user_threads SET title = %s, updated_at = now() "
            "WHERE thread_id = %s AND user_id = %s",
            (title, thread_id, user_id),
        ) > 0
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
    try:
        return pg.execute(
            "DELETE FROM user_threads WHERE thread_id = %s AND user_id = %s", (thread_id, user_id)
        ) > 0
    except Exception as e:
        logger.error(f"[db_user_threads] delete_user_thread error: {e}")
        return False


def delete_all_threads_for_user(user_id: str) -> list[str]:
    """
    Delete every user_threads row owned by user_id, regardless of count
    (unlike delete_user_threads_bulk, not capped to a caller-supplied id list).
    Returns the deleted thread_ids so the caller can cascade the LangGraph
    state cleanup via delete_threads_bulk() in db_threads_control.
    """
    try:
        rows = pg.fetch_all("DELETE FROM user_threads WHERE user_id = %s RETURNING thread_id", (user_id,))
        return [r["thread_id"] for r in rows]
    except Exception as e:
        logger.error(f"[db_user_threads] delete_all_threads_for_user error: {e}")
        return []


def delete_user_threads_bulk(thread_ids: list[str], user_id: str) -> list[str]:
    """
    Delete multiple threads from user_threads in one statement, scoped to user_id.
    Returns the subset of thread_ids that were actually owned and deleted; ids that
    don't exist or belong to another user are silently skipped.
    The caller is responsible for also calling delete_threads_bulk() in
    db_threads_control to clean up the LangGraph state for the returned ids.
    """
    if not thread_ids:
        return []
    try:
        rows = pg.fetch_all(
            "DELETE FROM user_threads WHERE user_id = %s AND thread_id = ANY(%s) RETURNING thread_id",
            (user_id, list(thread_ids)),
        )
        return [r["thread_id"] for r in rows]
    except Exception as e:
        logger.error(f"[db_user_threads] delete_user_threads_bulk error: {e}")
        return []


# ---------------------------------------------------------------------------
# Pin / unpin
# ---------------------------------------------------------------------------

def pin_user_thread(thread_id: str, user_id: str, is_pinned: bool) -> bool:
    """
    Set the is_pinned flag on a thread owned by the user.
    Returns True if the row was found and updated.
    """
    try:
        return pg.execute(
            "UPDATE user_threads SET is_pinned = %s WHERE thread_id = %s AND user_id = %s",
            (is_pinned, thread_id, user_id),
        ) > 0
    except Exception as e:
        logger.error(f"[db_user_threads] pin_user_thread error: {e}")
        return False


# ---------------------------------------------------------------------------
# Safety lock
# ---------------------------------------------------------------------------

def lock_user_thread_row(thread_id: str, reason: str) -> bool:
    """
    Mirror of lock_thread_state (db_threads_control) onto user_threads, so
    GET /api/threads/{id} and the thread list can read lock state off the same
    row as ui_messages. Not scoped to a specific user_id — this fires from a
    background safety-cutoff path that only has the thread_id.
    """
    try:
        return pg.execute(
            "UPDATE user_threads SET is_locked = TRUE, locked_reason = %s, locked_at = now() "
            "WHERE thread_id = %s",
            (reason, thread_id),
        ) > 0
    except Exception as e:
        logger.error(f"[db_user_threads] lock_user_thread_row error: {e}")
        return False


def get_thread_row(thread_id: str, user_id: str) -> dict | None:
    """Return ui_messages + title + lock state for an owned thread, or None if not found."""
    try:
        row = pg.fetch_one(
            "SELECT ui_messages, title, is_locked, locked_reason, locked_at, origin "
            "FROM user_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        )
        if not row:
            return None
        msgs = row["ui_messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        return {
            "messages": msgs or [],
            "title": row.get("title"),
            "is_locked": bool(row.get("is_locked")),
            "locked_reason": row.get("locked_reason"),
            "locked_at": row.get("locked_at"),
            "origin": row.get("origin"),
        }
    except Exception as e:
        logger.error(f"[db_user_threads] get_thread_row error: {e}")
        return None


# ---------------------------------------------------------------------------
# Account merge (guest → real user)
# ---------------------------------------------------------------------------

def merge_guest_to_user(user_id: str, guest_id: str) -> int:
    """
    Re-assign all threads belonging to guest_id to user_id in user_threads.
    Returns the number of threads migrated.
    Call reassign_threads_user() in db_threads_control separately to
    also update the ownership table.
    """
    try:
        return pg.execute("UPDATE user_threads SET user_id = %s WHERE user_id = %s", (user_id, guest_id))
    except Exception as e:
        logger.error(f"[db_user_threads] merge_guest_to_user error: {e}")
        return 0
