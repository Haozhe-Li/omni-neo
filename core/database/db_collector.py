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
) -> None:
    """Remember the inputs of one turn. Re-running a turn (it failed, or the
    annotator regenerated it) overwrites the row and drops any edit made to the
    answer it replaces."""
    await pg.aexecute(
        """
        INSERT INTO collector_turns (thread_id, turn, user_id, model, personalization, memory)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (thread_id, turn) DO UPDATE SET
            user_id = EXCLUDED.user_id,
            model = EXCLUDED.model,
            personalization = EXCLUDED.personalization,
            memory = EXCLUDED.memory,
            original_final_text = NULL,
            edited_final_text = NULL,
            updated_at = now()
        """,
        (thread_id, turn, user_id, model, pg.adapt(personalization), memory),
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
        "SELECT turn, model, personalization, memory, original_final_text, edited_final_text "
        "FROM collector_turns WHERE thread_id = %s AND user_id = %s ORDER BY turn",
        (thread_id, user_id),
    )


async def adelete_turns(thread_id: str, user_id: str) -> int:
    return await pg.aexecute(
        "DELETE FROM collector_turns WHERE thread_id = %s AND user_id = %s", (thread_id, user_id)
    )
