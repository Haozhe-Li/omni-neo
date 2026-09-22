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

import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.auth import get_current_user, resolve_user
from core.database.db_user_threads import get_thread_row
from core.database.db_user_usage import commit_charge_fast, evaluate_charge_fast
from core.routers.state import assert_thread_access_async
from core.utils.citations import all_citations, reset_citation_registry_async
from core.voice.agent import run_voice_turn
from core.voice.persist import persist_voice_turn
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

    missing = [k for k in ("OPENAI_API_KEY", "FISH_API_KEY") if not os.environ.get(k)]
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
            user_id,
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


# ---------------------------------------------------------------------------
# Typed continuation — a voice-origin thread stays on this same simple ReAct
# agent for its whole life, whether a given turn arrived spoken (above) or
# typed (below). No model choice, memory injection, rewind, file uploads, or
# skills — those are all deep-agent (core/agent.py) features this
# deliberately doesn't inherit, same reasoning as core/voice/agent.py's own
# docstring for why voice never used that harness to begin with.
# ---------------------------------------------------------------------------


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


_TEXT_TURN_CHARGE_KEY = "voice-text"  # 1 credit — see MODE_CREDIT_COST


async def _charge_voice_text_turn(user_id: str) -> bool:
    """Usage/credit gate for a typed voice-thread turn — same evaluate-then-
    commit split /chat uses (core/routers/chat.py): evaluate_charge_fast
    gates before anything runs, commit_charge_fast (fire-and-forget) only
    fires once we're actually proceeding, so a request that never starts
    never charges."""
    result = await evaluate_charge_fast(user_id, _TEXT_TURN_CHARGE_KEY)
    if not result["charged"]:
        return False
    commit_charge_fast(user_id, _TEXT_TURN_CHARGE_KEY)
    return True


class VoiceMessageRequest(BaseModel):
    query: str
    user_location: str | None = None
    user_local_datetime: str | None = None


@router.post("/api/voice_threads/{thread_id}/message")
async def voice_thread_message(
    thread_id: str,
    body: VoiceMessageRequest,
    user_id: str = Depends(get_current_user),
):
    """Stream a typed reply from the voice agent, in a chat-thread SSE shape
    (text/tool_call/sources/done — same events core/stream.py emits for
    regular chat) so the frontend's existing event handling covers this too.

    get_thread_row both confirms ownership and hands back `origin` in one
    query — reused here instead of assert_thread_access_async (which only
    checks threads_control, not what kind of thread this is).
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    row = get_thread_row(thread_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found or access denied.")
    if row.get("origin") != "voice":
        raise HTTPException(status_code=400, detail="Not a voice thread.")
    if row.get("is_locked"):
        raise HTTPException(status_code=403, detail="This conversation has been locked and can no longer be continued.")

    if not await _charge_voice_text_turn(user_id):
        raise HTTPException(status_code=429, detail="Usage limit exceeded.")

    async def _stream():
        # No rewind for voice threads, so no per-turn cap needed — see
        # core/utils/citations.py's reset_citation_registry_async.
        await reset_citation_registry_async(thread_id, None)
        full_text_parts: list[str] = []
        steps: list[dict] = []
        seen_citation_ns: set[int] = set()

        async for event in run_voice_turn(
            thread_id,
            query,
            user_location=body.user_location,
            user_local_datetime=body.user_local_datetime,
        ):
            etype = event["type"]
            if etype == "text":
                full_text_parts.append(event["delta"])
                yield _sse({"type": "text", "content": event["delta"]})
            elif etype == "tool_start":
                steps.append({
                    "tool": event["name"], "args": event.get("args", {}), "timestamp": int(time.time() * 1000),
                })
                yield _sse({"type": "tool_call", "tool": event["name"], "args": event["args"]})
            elif etype == "tool_end":
                # Whatever this tool call registered is ready to surface now —
                # same incremental "new since last check" pattern core/stream.py
                # uses, just read from the same registry instead of an SSE
                # loop over LangGraph's own message stream.
                new_sources = [c for c in all_citations() if c.get("n") not in seen_citation_ns]
                for c in new_sources:
                    seen_citation_ns.add(c["n"])
                if new_sources:
                    yield _sse({"type": "sources", "sources": new_sources})

        agent_text = "".join(full_text_parts)
        sources = all_citations()
        await persist_voice_turn(thread_id, user_id, query, agent_text, steps, sources)
        yield _sse({"type": "done", "sources": sources, "artifacts": []})

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(_stream(), media_type="text/event-stream", headers=headers)
