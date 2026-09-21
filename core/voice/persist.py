"""Writes a voice-agent turn into the same `user_threads.ui_messages` store
the regular chat frontend syncs to (see core/database/db_user_threads.py) —
shared by the live voice call (core/voice/session.py) and the typed-
continuation endpoint (core/routers/voice.py), so a voice-origin thread
reads the same way in the sidebar/text view regardless of which one
produced a given turn.

Deliberately NOT the fallback-then-race pattern core/routers/chat.py uses for
regular chat (client syncs, backend only persists if the client didn't) —
voice has no equivalent client-side sync call, so this is the only writer,
and it's simpler for it.
"""
from __future__ import annotations

import asyncio
import logging

from core.database.db_user_threads import get_thread_messages, upsert_thread_messages

logger = logging.getLogger(__name__)


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
        existing = await asyncio.to_thread(get_thread_messages, thread_id, user_id) or []
        msgs = list(existing)
        msgs.append({"role": "user", "content": user_text})
        assistant_msg: dict = {"role": "assistant", "content": agent_text}
        if steps:
            assistant_msg["steps"] = steps
        if sources:
            assistant_msg["sources"] = sources
        msgs.append(assistant_msg)
        await asyncio.to_thread(upsert_thread_messages, thread_id, user_id, msgs)
    except Exception:
        logger.exception("failed to persist voice turn for thread %s", thread_id)
