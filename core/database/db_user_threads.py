"""
Database operations for the user_threads table (over Supabase PostgREST, HTTP).

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE IF NOT EXISTS user_threads (
        thread_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        title VARCHAR(255),
        ui_messages JSONB DEFAULT '[]',
        search_text TEXT NOT NULL DEFAULT '',
        is_pinned BOOLEAN DEFAULT FALSE,
        origin VARCHAR(20),   -- NULL=chat, 'scheduled_task'=scheduled research run
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

Thread search used to lean on Postgres pg_trgm (GIN indexes + similarity()).
Over PostgREST that isn't available, so search now fetches the user's own
threads and does substring matching + fuzzy ranking here (see
search_user_threads). Thread counts are small per user, so this stays cheap.

Limits:
    - Guests: max GUEST_MAX_THREADS active threads
    - Logged-in users: no hard thread count limit
"""

import difflib
import json
import logging
import os

from core.database.supabase_client import supabase, utcnow_iso

logger = logging.getLogger(__name__)

GUEST_MAX_THREADS: int = int(os.getenv("GUEST_MAX_THREADS", "5"))

# Caps how much text per thread gets indexed/stored for search, guarding
# against pathologically long threads bloating the stored search_text.
SEARCH_TEXT_MAX_CHARS = 50_000


# ---------------------------------------------------------------------------
# Thread listing / reading
# ---------------------------------------------------------------------------

def get_threads_for_user(user_id: str) -> list[dict]:
    """Return all threads belonging to a user, newest first.

    Excludes scheduled-research threads (origin='scheduled_task') — those are
    surfaced only via /schedule_task's own run list (see
    core/routers/scheduled_tasks.py), never in the regular chat sidebar."""
    try:
        res = (
            supabase.table("user_threads")
            .select("thread_id, title, is_pinned, updated_at")
            .eq("user_id", user_id)
            .is_("origin", "null")
            .order("is_pinned", desc=True)
            .order("updated_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"[db_user_threads] get_threads_for_user error: {e}")
        return []


def get_thread_messages(thread_id: str, user_id: str) -> list | None:
    """Return ui_messages for a specific owned thread, or None if not found."""
    try:
        res = (
            supabase.table("user_threads")
            .select("ui_messages")
            .eq("thread_id", thread_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        msgs = res.data[0]["ui_messages"]
        if isinstance(msgs, str):
            return json.loads(msgs)
        return msgs or []
    except Exception as e:
        logger.error(f"[db_user_threads] get_thread_messages error: {e}")
        return None


def count_user_threads(user_id: str) -> int:
    """Return the number of active threads for a user (used for guest cap)."""
    try:
        res = (
            supabase.table("user_threads")
            .select("thread_id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        return res.count or 0
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


def _make_snippet(text: str, query: str, radius: int = 40) -> str:
    """Return a short excerpt of `text` centered on the first case-insensitive match of `query`."""
    if not text:
        return ""
    idx = text.lower().find(query.lower())
    if idx == -1:
        return text[:80]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(query) + radius)
    snippet = text[start:end]
    return ("…" if start > 0 else "") + snippet + ("…" if end < len(text) else "")


def _rank(title: str | None, search_text: str, query: str) -> float:
    """Relevance score approximating the old pg_trgm ranking: a title hit
    outranks a body-only hit, ties broken by fuzzy title similarity."""
    ql = query.lower()
    tl = (title or "").lower()
    base = 1.0 if ql in tl else (0.5 if ql in (search_text or "").lower() else 0.0)
    return base + difflib.SequenceMatcher(None, ql, tl).ratio()


# ---------------------------------------------------------------------------
# Thread creation / update
# ---------------------------------------------------------------------------

def upsert_thread_messages(thread_id: str, user_id: str, messages: list) -> bool:
    """
    Insert or update a thread's ui_messages (and derived search_text).
    Only updates if the row's user_id matches (prevents overwriting another user's data).
    """
    try:
        existing = (
            supabase.table("user_threads")
            .select("user_id")
            .eq("thread_id", thread_id)
            .limit(1)
            .execute()
        )
        # Guard: never clobber a thread owned by a different user.
        if existing.data and existing.data[0]["user_id"] != user_id:
            return False
        supabase.table("user_threads").upsert(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "ui_messages": messages,
                "search_text": _extract_search_text(messages),
                "updated_at": utcnow_iso(),
            },
            on_conflict="thread_id",
        ).execute()
        return True
    except Exception as e:
        logger.error(f"[db_user_threads] upsert_thread_messages error: {e}")
        return False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_user_threads(user_id: str, query: str, limit: int = 20) -> list[dict]:
    """
    Fuzzy-search a user's own threads by title and message content.
    Fetches the user's threads and ranks matches in-process (substring match +
    fuzzy title similarity), returning them ordered by relevance with a short
    match snippet.
    """
    query = (query or "").strip()
    if not query:
        return []
    try:
        res = (
            supabase.table("user_threads")
            .select("thread_id, title, is_pinned, updated_at, search_text")
            .eq("user_id", user_id)
            .is_("origin", "null")
            .execute()
        )
        ql = query.lower()
        matched = []
        for r in res.data:
            title = r.get("title") or ""
            search_text = r.get("search_text") or ""
            if ql in title.lower() or ql in search_text.lower():
                matched.append((_rank(title, search_text, query), r))
        matched.sort(key=lambda pair: (pair[0], pair[1].get("updated_at") or ""), reverse=True)
        return [
            {
                "thread_id": r["thread_id"],
                "title": r["title"],
                "is_pinned": r["is_pinned"],
                "updated_at": r["updated_at"],
                "snippet": _make_snippet(r.get("search_text") or "", query),
            }
            for _, r in matched[:limit]
        ]
    except Exception as e:
        logger.error(f"[db_user_threads] search_user_threads error: {e}")
        return []


def register_thread(thread_id: str, user_id: str, origin: str | None = None) -> bool:
    """
    Create a user_threads row at thread-creation time (called from /get_thread_id,
    and from the scheduled-task webhook with origin='scheduled_task').
    No-op if the thread is already registered.
    """
    try:
        supabase.table("user_threads").upsert(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "ui_messages": [],
                "origin": origin,
                "updated_at": utcnow_iso(),
            },
            on_conflict="thread_id",
            ignore_duplicates=True,
        ).execute()
        return True
    except Exception as e:
        logger.error(f"[db_user_threads] register_thread error: {e}")
        return False


def update_thread_title(thread_id: str, user_id: str, title: str) -> bool:
    """Update the title of a thread owned by the user."""
    try:
        res = (
            supabase.table("user_threads")
            .update({"title": title, "updated_at": utcnow_iso()})
            .eq("thread_id", thread_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)
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
        res = (
            supabase.table("user_threads")
            .delete()
            .eq("thread_id", thread_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)
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
        res = (
            supabase.table("user_threads")
            .delete()
            .eq("user_id", user_id)
            .execute()
        )
        return [r["thread_id"] for r in (res.data or [])]
    except Exception as e:
        logger.error(f"[db_user_threads] delete_all_threads_for_user error: {e}")
        return []


def delete_user_threads_bulk(thread_ids: list[str], user_id: str) -> list[str]:
    """
    Delete multiple threads from user_threads in one round trip, scoped to user_id.
    Returns the subset of thread_ids that were actually owned and deleted; ids that
    don't exist or belong to another user are silently skipped.
    The caller is responsible for also calling delete_threads_bulk() in
    db_threads_control to clean up the LangGraph state for the returned ids.
    """
    if not thread_ids:
        return []
    try:
        res = (
            supabase.table("user_threads")
            .delete()
            .eq("user_id", user_id)
            .in_("thread_id", thread_ids)
            .execute()
        )
        return [r["thread_id"] for r in (res.data or [])]
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
        res = (
            supabase.table("user_threads")
            .update({"is_pinned": is_pinned})
            .eq("thread_id", thread_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)
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
    try:
        res = (
            supabase.table("user_threads")
            .update({"user_id": user_id})
            .eq("user_id", guest_id)
            .execute()
        )
        return len(res.data or [])
    except Exception as e:
        logger.error(f"[db_user_threads] merge_guest_to_user error: {e}")
        return 0
