"""
Database operations for the user_memories table — durable, cross-thread,
per-user long-term memory (a single freeform markdown document per user).

All access is over Supabase's PostgREST HTTP API (see supabase_client.py).

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE IF NOT EXISTS user_memories (
        user_id VARCHAR(255) PRIMARY KEY,
        content TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
"""

import logging

from core.database.supabase_client import supabase, utcnow_iso
from core.utils.redis_cache import l1cache

logger = logging.getLogger(__name__)

# Hard cap on stored memory length. Backstops the extraction LLM's own
# "stay concise" instruction (core/memories_update_llm.py) so a run of bad
# rewrites can't grow the document — and therefore the per-turn prompt
# injection cost — unbounded.
MAX_MEMORY_CHARS = 3000


@l1cache(ttl=60)
def get_user_memory(user_id: str) -> str:
    """Return the user's long-term memory document, or '' if none stored yet."""
    try:
        res = (
            supabase.table("user_memories")
            .select("content")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        return res.data[0]["content"] if res.data else ""
    except Exception as e:
        logger.error(f"[db_user_memories] get_user_memory error: {e}")
        return ""


def save_user_memory(user_id: str, content: str) -> bool:
    """Insert or overwrite the user's memory document."""
    content = (content or "").strip()[:MAX_MEMORY_CHARS]
    try:
        supabase.table("user_memories").upsert(
            {"user_id": user_id, "content": content, "updated_at": utcnow_iso()},
            on_conflict="user_id",
        ).execute()
        l1cache.invalidate(get_user_memory, user_id)
        return True
    except Exception as e:
        logger.error(f"[db_user_memories] save_user_memory error: {e}")
        return False


def delete_user_memory(user_id: str) -> bool:
    """Clear a user's memory document. Returns True if a row was deleted."""
    try:
        res = supabase.table("user_memories").delete().eq("user_id", user_id).execute()
        l1cache.invalidate(get_user_memory, user_id)
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_user_memories] delete_user_memory error: {e}")
        return False


def migrate_guest_memory(user_id: str, guest_id: str) -> bool:
    """
    On guest -> signed-in merge: adopt the guest's memory only if the signed-in
    user doesn't already have one (never clobbers existing memory), then drop
    the guest's row either way. Call alongside merge_guest_to_user().
    """
    try:
        # Only adopt if the target user has no memory yet.
        existing = (
            supabase.table("user_memories")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not existing.data:
            guest = (
                supabase.table("user_memories")
                .select("content")
                .eq("user_id", guest_id)
                .limit(1)
                .execute()
            )
            if guest.data:
                supabase.table("user_memories").insert(
                    {"user_id": user_id, "content": guest.data[0]["content"]}
                ).execute()
        supabase.table("user_memories").delete().eq("user_id", guest_id).execute()
        l1cache.invalidate(get_user_memory, user_id)
        l1cache.invalidate(get_user_memory, guest_id)
        return True
    except Exception as e:
        logger.error(f"[db_user_memories] migrate_guest_memory error: {e}")
        return False
