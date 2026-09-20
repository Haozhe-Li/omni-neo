"""WebSocket entry point for the live voice agent.

Auth + thread_id + checkpointer follow /chat exactly (core/routers/chat.py,
core/auth.py) — no more "no auth, no thread association" MVP shortcut. The
one real difference is *how* auth/thread_id arrive: a browser WebSocket
handshake can't carry a custom Authorization header the way fetch() can, so
the frontend puts the same values (Clerk JWT or guest id, thread_id from
/get_thread_id?origin=voice) into the connection URL's query string instead,
and this endpoint runs them through the same core.auth.resolve_user check and
the same assert_thread_access_async ownership guard /chat uses. From there
on, conversation state lives in the same Redis-backed LangGraph checkpointer
every other agent uses (see core/voice/agent.py) keyed by that thread_id, not
in anything this router or VoiceSession holds itself.

See core/voice/session.py for the per-connection orchestration and
core/static/voice_test.html for the manual test client.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from core.auth import resolve_user
from core.routers.state import assert_thread_access_async
from core.voice.session import VoiceSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice"])

_TEST_CLIENT_HTML = Path(__file__).resolve().parent.parent / "static" / "voice_test.html"


@router.get("/voice_test")
async def voice_test_page() -> HTMLResponse:
    return HTMLResponse(_TEST_CLIENT_HTML.read_text(encoding="utf-8"))


async def _reject(websocket: WebSocket, detail: str) -> None:
    await websocket.send_json({"type": "error", "detail": detail})
    await websocket.close(code=1008)


@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    missing = [k for k in ("DEEPGRAM_API_KEY", "FISH_API_KEY") if not os.environ.get(k)]
    if missing:
        await websocket.send_json({"type": "error", "detail": f"Missing env vars: {', '.join(missing)}"})
        await websocket.close()
        return

    params = websocket.query_params
    try:
        user_id = resolve_user(bearer_token=params.get("token"), guest_id=params.get("guest_id"))
    except HTTPException as e:
        await _reject(websocket, str(e.detail))
        return

    thread_id = params.get("thread_id")
    if not thread_id:
        await _reject(websocket, "thread_id is required")
        return

    try:
        await assert_thread_access_async(thread_id, user_id)
    except HTTPException as e:
        await _reject(websocket, str(e.detail))
        return

    try:
        session = VoiceSession(
            websocket,
            thread_id,
            user_location=params.get("user_location"),
            user_local_datetime=params.get("user_local_datetime"),
        )
    except Exception as e:
        await websocket.send_json({"type": "error", "detail": str(e)})
        await websocket.close()
        return

    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("voice session crashed")
