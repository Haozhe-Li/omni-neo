"""
Database operations for shared_threads / thread_forks — share-a-conversation-by-link
(see core/sharing.py for what a share holds, schema.sql for the columns).

Direct Postgres access (see pg.py). Async because every caller is an async
endpoint that also talks to the checkpointer; `delete_user_shares` is sync for
the account-erase handler, which is.
"""

import logging

from core.database import pg
from core.database.db_user_threads import _extract_search_text
from core.sharing import Snapshot

logger = logging.getLogger(__name__)


async def acount_shares(owner_id: str) -> int:
    row = await pg.afetch_one("SELECT count(*) AS n FROM shared_threads WHERE owner_id = %s", (owner_id,))
    return int(row["n"])


async def asave_share(
    *, share_id: str, owner_id: str, source_thread_id: str, title: str | None, snap: Snapshot
) -> None:
    await pg.aexecute(
        "INSERT INTO shared_threads (share_id, owner_id, source_thread_id, title, ui_messages, "
        "agent_state, citations, files_meta, n_messages, size_bytes) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            share_id, owner_id, source_thread_id, title, pg.adapt(snap.ui_messages),
            pg.adapt(snap.agent_state), pg.adapt(snap.citations), pg.adapt(snap.files_meta),
            snap.n_messages, snap.size_bytes,
        ),
    )


async def aget_share_public(share_id: str) -> dict | None:
    """What the public page needs and nothing more — never the agent state."""
    return await pg.afetch_one(
        "SELECT share_id, title, ui_messages, n_messages, created_at FROM shared_threads WHERE share_id = %s",
        (share_id,),
    )


async def aget_share_full(share_id: str) -> dict | None:
    """Everything, for forking."""
    return await pg.afetch_one("SELECT * FROM shared_threads WHERE share_id = %s", (share_id,))


async def alist_shares(owner_id: str) -> list[dict]:
    return await pg.afetch_all(
        "SELECT share_id, source_thread_id, title, n_messages, created_at FROM shared_threads "
        "WHERE owner_id = %s ORDER BY created_at DESC",
        (owner_id,),
    )


async def adelete_share(share_id: str, owner_id: str) -> bool:
    """Revoke. Scoped to the owner so a guessed id cannot remove someone else's link."""
    return await pg.aexecute(
        "DELETE FROM shared_threads WHERE share_id = %s AND owner_id = %s", (share_id, owner_id)
    ) > 0


def delete_user_shares(owner_id: str) -> int:
    """Erase every link a user published (DELETE /user-data)."""
    try:
        return pg.execute("DELETE FROM shared_threads WHERE owner_id = %s", (owner_id,))
    except Exception as e:
        logger.error(f"[db_shared_threads] delete_user_shares error: {e}")
        return 0


async def aget_user_files_meta(user_id: str, file_ids: list[str]) -> list[dict]:
    """Metadata (no content) of the user's own attachments, for the snapshot."""
    if not file_ids:
        return []
    return await pg.afetch_all(
        "SELECT file_id, original_filename, file_type, file_size_bytes, status, category, created_at "
        "FROM user_files WHERE user_id = %s AND file_id = ANY(%s) ORDER BY created_at",
        (user_id, file_ids),
    )


async def aregister_fork(
    *,
    thread_id: str,
    user_id: str,
    share_id: str,
    title: str | None,
    ui_messages: list[dict],
    files: list[tuple[str, dict]],
) -> None:
    """Create the viewer's private copy in Postgres, in one transaction: the
    ownership row, the visible thread, the fork marker, and a metadata-only copy
    of each attachment under its new id (`files` is `[(new_file_id, meta)]`).

    The attachment copies carry no S3 object and no extracted text — the content
    is already in the forked agent state. They exist so that a same-named upload
    in the fork is numbered after the inherited one instead of overwriting its
    mounted path (`_dedupe_document_name` counts rows by thread).
    """
    async with (await pg.apool()).connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO threads_control (thread_id, user_id) VALUES (%s, %s)", (thread_id, user_id)
        )
        await conn.execute(
            "INSERT INTO user_threads (thread_id, user_id, title, ui_messages, search_text) "
            "VALUES (%s, %s, %s, %s, %s)",
            (thread_id, user_id, title, pg.adapt(ui_messages), _extract_search_text(ui_messages)),
        )
        await conn.execute(
            "INSERT INTO thread_forks (thread_id, share_id, inherited_messages) VALUES (%s, %s, %s)",
            (thread_id, share_id, len(ui_messages)),
        )
        for new_id, meta in files:
            await conn.execute(
                "INSERT INTO user_files (file_id, user_id, thread_id, original_filename, file_type, "
                "file_size_bytes, status, s3_bucket, category, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, %s, %s)",
                (
                    new_id, user_id, thread_id, meta["original_filename"], meta["file_type"],
                    meta["file_size_bytes"], meta["status"], meta["category"], meta["created_at"],
                ),
            )


async def aget_fork(thread_id: str) -> dict | None:
    """`{share_id, inherited_messages}` if this thread is a copy of a shared one."""
    return await pg.afetch_one(
        "SELECT share_id, inherited_messages FROM thread_forks WHERE thread_id = %s", (thread_id,)
    )
