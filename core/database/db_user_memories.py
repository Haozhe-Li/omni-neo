"""
Database operations for the user_memories table — durable, cross-thread,
per-user long-term memory (a single freeform markdown document per user).

Direct Postgres access (see pg.py); table schema in schema.sql.
"""

import logging

from core.database import pg
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
        row = pg.fetch_one("SELECT content FROM user_memories WHERE user_id = %s", (user_id,))
        return row["content"] if row else ""
    except Exception as e:
        logger.error(f"[db_user_memories] get_user_memory error: {e}")
        return ""


def save_user_memory(user_id: str, content: str) -> bool:
    """Insert or overwrite the user's memory document."""
    content = (content or "").strip()[:MAX_MEMORY_CHARS]
    try:
        pg.execute(
            "INSERT INTO user_memories (user_id, content) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET content = EXCLUDED.content, updated_at = now()",
            (user_id, content),
        )
        l1cache.invalidate(get_user_memory, user_id)
        return True
    except Exception as e:
        logger.error(f"[db_user_memories] save_user_memory error: {e}")
        return False


def delete_user_memory(user_id: str) -> bool:
    """Clear a user's memory document. Returns True if a row was deleted."""
    try:
        deleted = pg.execute("DELETE FROM user_memories WHERE user_id = %s", (user_id,)) > 0
        l1cache.invalidate(get_user_memory, user_id)
        return deleted
    except Exception as e:
        logger.error(f"[db_user_memories] delete_user_memory error: {e}")
        return False


def migrate_guest_memory(user_id: str, guest_id: str) -> bool:
    """
    On guest -> signed-in merge: adopt the guest's memory only if the signed-in
    user doesn't already have one (never clobbers existing memory), then drop
    the guest's row either way. One transaction. Call alongside merge_guest_to_user().
    """
    try:
        with pg.pool().connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO user_memories (user_id, content) "
                "SELECT %s, content FROM user_memories WHERE user_id = %s "
                "ON CONFLICT (user_id) DO NOTHING",
                (user_id, guest_id),
            )
            conn.execute("DELETE FROM user_memories WHERE user_id = %s", (guest_id,))
        l1cache.invalidate(get_user_memory, user_id)
        l1cache.invalidate(get_user_memory, guest_id)
        return True
    except Exception as e:
        logger.error(f"[db_user_memories] migrate_guest_memory error: {e}")
        return False
