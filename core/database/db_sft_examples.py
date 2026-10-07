"""
Database operations for sft_examples / harness_snapshots / sft_images — the
thumbs-up -> fine-tuning pipeline (see core/sft_capture.py for what is stored,
schema.sql for the columns).

Direct Postgres access (see pg.py). The write path is async because it runs in
the feedback endpoint, next to the checkpoint read; the erase/reassign helpers
are sync because their callers (core/routers/users.py) are.
"""

import logging

from core.database import pg
from core.sft_capture import CapturedTurn

logger = logging.getLogger(__name__)


async def _insert_snapshot_and_images(
    conn, cap: CapturedTurn, harness_hash: str, system_prompt: str, tools: list[dict],
    deepagents_version: str | None,
) -> None:
    """The rows an example points at, written in the example's own transaction."""
    await conn.execute(
        "INSERT INTO harness_snapshots (harness_hash, system_prompt, tools, deepagents_version) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (harness_hash) DO NOTHING",
        (harness_hash, system_prompt, pg.adapt(tools), deepagents_version),
    )
    for sha, (mime, data) in cap.images.items():
        await conn.execute(
            "INSERT INTO sft_images (sha256, mime, data) VALUES (%s, %s, %s) "
            "ON CONFLICT (sha256) DO NOTHING",
            (sha, mime, data),
        )


async def asave_example(
    *,
    thread_id: str,
    turn: int,
    user_id: str,
    cap: CapturedTurn,
    harness_hash: str,
    system_prompt: str,
    tools: list[dict],
    deepagents_version: str | None,
) -> int:
    """Insert or refresh the example for (thread, turn); returns its id.

    One transaction: the snapshot and images an example points at are written
    with it, so a half-saved example can never reference something missing.

    Re-saving the same turn (a second thumbs-up after un-liking) keeps a
    reviewer's `status` when the conversation is unchanged, and resets it to
    'pending' when it is not — a regenerated answer is a different example and
    must not inherit the approval of the one it replaced.
    """
    async with (await pg.apool()).connection() as conn, conn.transaction():
        await _insert_snapshot_and_images(
            conn, cap, harness_hash, system_prompt, tools, deepagents_version
        )
        cur = await conn.execute(
            """
            INSERT INTO sft_examples (
                thread_id, turn, user_id, harness_hash, models_seen, messages,
                n_assistant_turns, n_tool_calls, tools_used, approx_chars,
                has_image, has_memory, has_attachments, compacted
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, turn) DO UPDATE SET
                user_id           = EXCLUDED.user_id,
                harness_hash      = EXCLUDED.harness_hash,
                models_seen       = EXCLUDED.models_seen,
                n_assistant_turns = EXCLUDED.n_assistant_turns,
                n_tool_calls      = EXCLUDED.n_tool_calls,
                tools_used        = EXCLUDED.tools_used,
                approx_chars      = EXCLUDED.approx_chars,
                has_image         = EXCLUDED.has_image,
                has_memory        = EXCLUDED.has_memory,
                has_attachments   = EXCLUDED.has_attachments,
                compacted         = EXCLUDED.compacted,
                status            = CASE WHEN sft_examples.messages = EXCLUDED.messages
                                         THEN sft_examples.status ELSE 'pending' END,
                status_note       = CASE WHEN sft_examples.messages = EXCLUDED.messages
                                         THEN sft_examples.status_note ELSE NULL END,
                messages          = EXCLUDED.messages,
                updated_at        = now()
            RETURNING id
            """,
            (
                thread_id, turn, user_id, harness_hash, cap.models_seen, pg.adapt(cap.messages),
                cap.n_assistant_turns, cap.n_tool_calls, cap.tools_used, cap.approx_chars,
                cap.has_image, cap.has_memory, cap.has_attachments, cap.compacted,
            ),
        )
        row = await cur.fetchone()
    return row["id"]


async def asave_collector_example(
    *,
    thread_id: str,
    turn: int,
    user_id: str,
    cap: CapturedTurn,
    harness_hash: str,
    system_prompt: str,
    tools: list[dict],
    deepagents_version: str | None,
    status: str,
    edited: bool,
    original_final_text: str | None,
    collect_meta: dict,
) -> int:
    """Insert or replace the collector's example for (thread, turn); returns its id.

    Unlike `asave_example` there is no "keep the reviewer's status when the
    conversation is unchanged" rule: the annotator is the reviewer, and
    submitting again is them saying what this row should now be, so the status,
    the edit flag and the metadata are all overwritten with what they just
    submitted.
    """
    async with (await pg.apool()).connection() as conn, conn.transaction():
        await _insert_snapshot_and_images(
            conn, cap, harness_hash, system_prompt, tools, deepagents_version
        )
        cur = await conn.execute(
            """
            INSERT INTO sft_examples (
                thread_id, turn, user_id, harness_hash, models_seen, messages,
                n_assistant_turns, n_tool_calls, tools_used, approx_chars,
                has_image, has_memory, has_attachments, compacted,
                source, status, edited, original_final_text, collect_meta
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      'collector', %s, %s, %s, %s)
            ON CONFLICT (thread_id, turn) DO UPDATE SET
                user_id             = EXCLUDED.user_id,
                harness_hash        = EXCLUDED.harness_hash,
                models_seen         = EXCLUDED.models_seen,
                messages            = EXCLUDED.messages,
                n_assistant_turns   = EXCLUDED.n_assistant_turns,
                n_tool_calls        = EXCLUDED.n_tool_calls,
                tools_used          = EXCLUDED.tools_used,
                approx_chars        = EXCLUDED.approx_chars,
                has_image           = EXCLUDED.has_image,
                has_memory          = EXCLUDED.has_memory,
                has_attachments     = EXCLUDED.has_attachments,
                compacted           = EXCLUDED.compacted,
                source              = 'collector',
                status              = EXCLUDED.status,
                status_note         = NULL,
                edited              = EXCLUDED.edited,
                original_final_text = EXCLUDED.original_final_text,
                collect_meta        = EXCLUDED.collect_meta,
                updated_at          = now()
            RETURNING id
            """,
            (
                thread_id, turn, user_id, harness_hash, cap.models_seen, pg.adapt(cap.messages),
                cap.n_assistant_turns, cap.n_tool_calls, cap.tools_used, cap.approx_chars,
                cap.has_image, cap.has_memory, cap.has_attachments, cap.compacted,
                status, edited, original_final_text, pg.adapt(collect_meta),
            ),
        )
        row = await cur.fetchone()
    return row["id"]


async def adelete_example(thread_id: str, turn: int, user_id: str) -> bool:
    """Drop one example (the user took their thumbs-up back). Scoped to the
    user so a guessed thread id cannot remove someone else's row."""
    return await pg.aexecute(
        "DELETE FROM sft_examples WHERE thread_id = %s AND turn = %s AND user_id = %s",
        (thread_id, turn, user_id),
    ) > 0


async def adelete_examples_from_turn(thread_id: str, user_id: str, turn: int) -> int:
    """Drop the examples for `turn` and every later turn of a thread.

    Called when the thread is rewound to `turn`: the answer is regenerated (or
    the question edited) and everything after it is discarded, so an example
    captured from the old timeline is stale — it would train on an answer the
    user no longer has, under the approval they gave the old one.
    """
    return await pg.aexecute(
        "DELETE FROM sft_examples WHERE thread_id = %s AND user_id = %s AND turn >= %s",
        (thread_id, user_id, turn),
    )


def delete_user_examples(user_id: str) -> int:
    """Erase every example a user produced (DELETE /user-data), then any image
    no remaining example references."""
    try:
        deleted = pg.execute("DELETE FROM sft_examples WHERE user_id = %s", (user_id,))
        if deleted:
            _purge_orphan_images()
        return deleted
    except Exception as e:
        logger.error(f"[db_sft_examples] delete_user_examples error: {e}")
        return 0


def reassign_examples_user(old_user_id: str, new_user_id: str) -> int:
    """Guest -> signed-in merge: the guest's examples follow their threads, so
    that erasing the account later erases them too."""
    try:
        return pg.execute(
            "UPDATE sft_examples SET user_id = %s WHERE user_id = %s",
            (new_user_id, old_user_id),
        )
    except Exception as e:
        logger.error(f"[db_sft_examples] reassign_examples_user error: {e}")
        return 0


def _purge_orphan_images() -> int:
    """Delete images that no example's messages reference by hash.

    A text scan of the messages column, which is fine because it only runs on
    account erasure — nothing on a request path calls it.
    """
    return pg.execute(
        "DELETE FROM sft_images i WHERE NOT EXISTS ("
        "  SELECT 1 FROM sft_examples e WHERE e.messages::text LIKE '%%' || i.sha256 || '%%')"
    )
