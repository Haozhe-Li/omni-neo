"""Database operations for collector_turns — what an annotator asked for on each
turn of a collector thread, and the edit they made to its answer.

Written by core/routers/collector.py; schema in schema.sql. Direct Postgres
access (see pg.py); async because every caller is an async endpoint.
"""

from core.database import pg


async def arecord_turn(
    *,
    thread_id: str,
    turn: int,
    user_id: str,
    model: str | None,
    personalization: dict,
    memory: str | None,
    skill: str | None = None,
    attachments: list[dict] | None = None,
    source_urls: list[str] | None = None,
) -> None:
    """Remember the inputs of one turn. Re-running a turn (it failed, or the
    annotator regenerated it) overwrites the row and drops any edit made to the
    answer it replaces."""
    await pg.aexecute(
        """
        INSERT INTO collector_turns
            (thread_id, turn, user_id, model, personalization, memory, skill, attachments, source_urls)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (thread_id, turn) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            model = EXCLUDED.model,
            personalization = EXCLUDED.personalization,
            memory = EXCLUDED.memory,
            skill = EXCLUDED.skill,
            attachments = EXCLUDED.attachments,
            source_urls = EXCLUDED.source_urls,
            original_final_text = NULL,
            edited_final_text = NULL,
            updated_at = now()
        """,
        (thread_id, turn, user_id, model, pg.adapt(personalization), memory, skill,
         pg.adapt(attachments or []), pg.adapt(source_urls or [])),
    )


async def arecord_edit(thread_id: str, turn: int, user_id: str, original: str, edited: str) -> bool:
    """Store an edit. `original_final_text` is written once, the first time, so
    editing twice still compares against what the model actually said."""
    return await pg.aexecute(
        """
        UPDATE collector_turns SET
            original_final_text = COALESCE(original_final_text, %s),
            edited_final_text = %s,
            updated_at = now()
        WHERE thread_id = %s AND turn = %s AND user_id = %s
        """,
        (original, edited, thread_id, turn, user_id),
    ) > 0


async def alist_turns(thread_id: str, user_id: str) -> list[dict]:
    return await pg.afetch_all(
        "SELECT turn, model, personalization, memory, skill, attachments, source_urls, "
        "       original_final_text, edited_final_text "
        "FROM collector_turns WHERE thread_id = %s AND user_id = %s ORDER BY turn",
        (thread_id, user_id),
    )


async def adelete_turns(thread_id: str, user_id: str) -> int:
    return await pg.aexecute(
        "DELETE FROM collector_turns WHERE thread_id = %s AND user_id = %s", (thread_id, user_id)
    )


async def arebind_files(old_thread_id: str, new_thread_id: str, user_id: str) -> int:
    """Move a thread's uploaded files to another thread, keeping their ids.

    Used when a conversation is restarted: the files the annotator staged for the
    first turn belong to the new thread, so the mount names `build_message_content`
    derives from the thread's files come out as they would have in the first run.
    """
    return await pg.aexecute(
        "UPDATE user_files SET thread_id = %s, updated_at = now() WHERE thread_id = %s AND user_id = %s",
        (new_thread_id, old_thread_id, user_id),
    )


async def apurge_files(thread_id: str, user_id: str) -> list[tuple[str, str]]:
    """Delete a thread's `user_files` rows; returns `(bucket, key)` of the objects to remove."""
    rows = await pg.afetch_all(
        "DELETE FROM user_files WHERE thread_id = %s AND user_id = %s RETURNING file_id, s3_bucket",
        (thread_id, user_id),
    )
    return [(r["s3_bucket"], r["file_id"]) for r in rows if r.get("s3_bucket")]
