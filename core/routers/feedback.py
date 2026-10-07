"""Thumbs-up / thumbs-down on an answer.

A thumbs-up on a teacher-model (luna) answer also records the exchange as a
fine-tuning example (core/sft_capture.py, table sft_examples). That is invisible
to the client by design: the buttons behave identically whether or not the turn
qualified, and a turn that cannot be recorded is `status: "skipped"`, not an
error. The client treats the whole call as fire-and-forget.

Only an up-vote captures anything. A down-vote (or taking the up-vote back)
removes the example if one exists; nothing is ever stored for a down-vote, since
a rejected answer has no use as a training target and the conversation is the
user's.
"""

import asyncio
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.agent import get_agent
from core.auth import get_current_user
from core.database.db_sft_examples import adelete_example, asave_example
from core.database.db_user_threads import get_thread_row
from core.harness_snapshot import live_harness
from core.routers.state import assert_thread_access_async
from core.sft_capture import CaptureError, build_capture

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])

# Capture failures that mean "this turn is just not a candidate" (expected,
# quiet) versus "the two histories disagree" (a bug worth a warning).
_EXPECTED_SKIPS = {
    "not_teacher_model", "turn_incomplete", "unpaired_tool_calls", "too_large",
    "has_memory", "has_attachments",
}


class FeedbackRequest(BaseModel):
    # The frontend's turn number for this exchange — the index of the assistant
    # message in its list, which equals the QueryRequest.turn of the question
    # (see core/sft_capture.py).
    turn: int = Field(ge=1)
    rating: Literal["up", "down", "none"]


def _deepagents_version() -> str | None:
    try:
        return version("deepagents")
    except PackageNotFoundError:
        return None


@router.post("/api/threads/{thread_id}/feedback")
async def post_feedback(
    thread_id: str,
    body: FeedbackRequest,
    user_id: str = Depends(get_current_user),
):
    await assert_thread_access_async(thread_id, user_id)

    if body.rating != "up":
        removed = await adelete_example(thread_id, body.turn, user_id)
        return {"status": "cleared" if removed else "noop"}

    row = await asyncio.to_thread(get_thread_row, thread_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found or access denied.")
    # Voice and scheduled-research threads run different agents with different
    # prompts; their traces are not examples of this harness.
    if row.get("origin"):
        return {"status": "skipped", "reason": "not_a_chat_thread"}
    # A thread that tripped the safety guard is not training material, including
    # the turns that came before the one that tripped it.
    if row.get("is_locked"):
        return {"status": "skipped", "reason": "thread_locked"}

    # Any model's agent will do: they all share this checkpointer and thread
    # state is not per-model (see get_agent).
    state = await get_agent(None).aget_state({"configurable": {"thread_id": thread_id}})
    values = state.values or {}
    try:
        cap = build_capture(
            values.get("messages", []),
            body.turn,
            ui_messages=row["messages"],
            compacted=bool(values.get("_summarization_event")),
        )
    except CaptureError as e:
        log = logger.info if e.code in _EXPECTED_SKIPS else logger.warning
        log(f"[feedback] skipped thread={thread_id} turn={body.turn}: {e.code} ({e})")
        return {"status": "skipped", "reason": e.code}

    harness_hash, system_prompt, tools = await live_harness()
    example_id = await asave_example(
        thread_id=thread_id,
        turn=body.turn,
        user_id=user_id,
        cap=cap,
        harness_hash=harness_hash,
        system_prompt=system_prompt,
        tools=tools,
        deepagents_version=_deepagents_version(),
    )
    logger.info(
        f"[feedback] recorded example={example_id} thread={thread_id} turn={body.turn} "
        f"msgs={len(cap.messages)} tools={cap.n_tool_calls} chars={cap.approx_chars}"
    )
    return {"status": "recorded", "id": example_id}
