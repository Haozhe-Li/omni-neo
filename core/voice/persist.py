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


async def persist_voice_turn(
    thread_id: str,
    user_id: str,
    user_text: str,
    agent_text: str,
    steps: list[dict],
    sources: list[dict],
) -> None:
    """Append one user+assistant exchange to the thread's ui_messages.

    Best-effort: a failed write means this turn doesn't show up in the
    sidebar/text transcript, not that the call or reply itself failed, so
    errors are logged and swallowed rather than raised into the caller's
    own turn-handling flow.
    """
    if not agent_text.strip():
        return
    try:
        row = await asyncio.to_thread(get_thread_row, thread_id, user_id) or {}
        msgs = list(row.get("messages") or [])
        msgs.append({"role": "user", "content": user_text})
        assistant_msg: dict = {"role": "assistant", "content": agent_text}
        if steps:
            assistant_msg["steps"] = steps
        if sources:
            assistant_msg["sources"] = sources
        msgs.append(assistant_msg)
        if not await asyncio.to_thread(upsert_thread_messages, thread_id, user_id, msgs):
            return
        # Checked on every turn, not just the first, so a voice thread saved
        # before titles existed picks one up the next time it's resumed.
        if not (row.get("title") or "").strip():
            first_user = next((m["content"] for m in msgs if m.get("role") == "user" and m.get("content")), user_text)
            # The raw words go in first, the LLM label replaces them after —
            # the same order regular chat uses. Hanging up right after the
            # first reply lands on the thread (and its sidebar fetch) within
            # a second, before a label-only write would have finished.
            placeholder = first_user.strip()[:_PLACEHOLDER_TITLE_CHARS]
            await asyncio.to_thread(update_thread_title, thread_id, user_id, placeholder)
            title = await asyncio.to_thread(_generated_title, first_user)
            if title:
                await asyncio.to_thread(update_thread_title, thread_id, user_id, title)
    except Exception:
        logger.exception("failed to persist voice turn for thread %s", thread_id)
