"""Writes a voice-agent turn into the same `user_threads.ui_messages` store
the regular chat frontend syncs to (see core/database/db_user_threads.py) —
shared by the live voice call (core/voice/session.py) and the typed-
continuation endpoint (core/routers/voice.py), so a voice-origin thread
reads the same way in the sidebar/text view regardless of which one
produced a given turn.

Deliberately NOT the fallback-then-race pattern core/routers/chat.py uses for
regular chat (client syncs, backend only persists if the client didn't) —
voice has no equivalent client-side sync call, so this is the only writer,
and it's simpler for it. That includes the title: regular chat's frontend
calls /get_title and syncs the result, voice has nothing to do that, and the
sidebar hides any thread without a title.
"""
from __future__ import annotations

import asyncio
import logging

from core.database.db_user_threads import get_thread_row, update_thread_title, upsert_thread_messages
from core.get_title import get_title

logger = logging.getLogger(__name__)

_PLACEHOLDER_TITLE_CHARS = 40


def _generated_title(first_user_text: str) -> str | None:
    """The same short LLM topic label regular chat gets from /get_title."""
    try:
        return (get_title(first_user_text) or "").strip() or None
    except Exception:
        logger.exception("voice title generation failed")
        return None


async def _upgrade_title(thread_id: str, user_id: str, first_user_text: str) -> None:
    title = await asyncio.to_thread(_generated_title, first_user_text)
    if title:
        await asyncio.to_thread(update_thread_title, thread_id, user_id, title)


# Strong references for fire-and-forget tasks — the event loop only keeps a
# weak one, so an unreferenced task can be garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


async def persist_voice_turn(
    thread_id: str,
    user_id: str,
    user_text: str,
    agent_text: str,
    steps: list[dict],
    sources: list[dict],
    *,
    call_start: bool = False,
) -> int | None:
    """Append one user+assistant exchange to the thread's ui_messages.

    `call_start` flags the user message as where a live call began; the
    thread view draws its "Voice call started" banner there (the matching
    "ended" one comes from mark_voice_call_end). Returns the saved reply's
    index in ui_messages, or None if nothing was saved.

    Best-effort: a failed write means this turn doesn't show up in the
    sidebar/text transcript, not that the call or reply itself failed, so
    errors are logged and swallowed rather than raised into the caller's
    own turn-handling flow.
    """
    if not agent_text.strip():
        return None
    try:
        row = await asyncio.to_thread(get_thread_row, thread_id, user_id)
        # Every voice thread's row exists from the moment its id is minted
        # (/get_thread_id), so None here means the read failed — writing on
        # anyway would replace the whole saved history with just this turn.
        if row is None:
            logger.error("voice turn not saved: couldn't read thread %s", thread_id)
            return None
        msgs = list(row.get("messages") or [])
        user_msg: dict = {"role": "user", "content": user_text}
        if call_start:
            user_msg["voice_call"] = "start"
        msgs.append(user_msg)
        assistant_msg: dict = {"role": "assistant", "content": agent_text.strip()}
        if steps:
            assistant_msg["steps"] = steps
        if sources:
            assistant_msg["sources"] = sources
        msgs.append(assistant_msg)
        if not await asyncio.to_thread(upsert_thread_messages, thread_id, user_id, msgs):
            return None
        # Checked on every turn, not just the first, so a voice thread saved
        # before titles existed picks one up the next time it's resumed.
        if not (row.get("title") or "").strip():
            first_user = next((m["content"] for m in msgs if m.get("role") == "user" and m.get("content")), user_text)
            # The raw words go in first, the LLM label replaces them after —
            # the same order regular chat uses. Hanging up right after the
            # first reply lands on the thread (and its sidebar fetch) within
            # a second, before a label-only write would have finished. The
            # label runs on its own: the caller serializes these saves, and
            # nothing after this one should wait on an LLM call.
            placeholder = first_user.strip()[:_PLACEHOLDER_TITLE_CHARS]
            await asyncio.to_thread(update_thread_title, thread_id, user_id, placeholder)
            task = asyncio.create_task(_upgrade_title(thread_id, user_id, first_user))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        return len(msgs) - 1
    except Exception:
        logger.exception("failed to persist voice turn for thread %s", thread_id)
        return None


async def mark_voice_call_end(thread_id: str, user_id: str, reply_index: int) -> None:
    """Flag a call's last saved reply, so the thread view draws its "Voice
    call ended" banner after it. Best-effort, like persist_voice_turn."""
    try:
        row = await asyncio.to_thread(get_thread_row, thread_id, user_id)
        msgs = list((row or {}).get("messages") or [])
        if not (0 <= reply_index < len(msgs)) or msgs[reply_index].get("role") != "assistant":
            return
        msgs[reply_index] = {**msgs[reply_index], "voice_call": "end"}
        await asyncio.to_thread(upsert_thread_messages, thread_id, user_id, msgs)
    except Exception:
        logger.exception("failed to mark the voice call's end for thread %s", thread_id)
