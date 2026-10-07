"""Share a conversation by link, and continue someone else's.

    POST   /api/threads/{id}/share        owner, signed in   -> freeze a snapshot, new link every time
    GET    /api/shares                    signed in          -> my links
    DELETE /api/shares/{share_id}         owner              -> revoke
    GET    /api/shared/{share_id}         public, no auth    -> the read-only page's data
    POST   /api/shared/{share_id}/fork    signed in          -> a private copy of it for me

A share is a Postgres snapshot (core/sharing.py, schema.sql); a fork is where it
becomes a real thread — checkpoint, citations, vector index — owned by the
viewer. The public endpoint never returns the agent state, only what the page
renders. Guests can look at a shared link but cannot create one or fork one: a
guest's threads are wiped after three days, which is the wrong home for a
conversation someone chose to keep.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from core.agent import get_agent
from core.auth import get_current_user
from core.database.db_shared_threads import (
    acount_shares,
    adelete_share,
    aget_share_full,
    aget_share_public,
    aget_user_files_meta,
    alist_shares,
    aregister_fork,
    asave_share,
)
from core.database.db_threads_control import delete_thread as delete_thread_state
from core.database.db_user_threads import get_thread_row
from core.redis_stream import stream_is_generating
from core.routers.state import assert_thread_access_async
from core.sharing import (
    MAX_ACTIVE_SHARES,
    SHARE_ID_RE,
    ShareError,
    attachment_ids,
    build_snapshot,
    deserialize_state,
    new_share_id,
    remap_attachment_ids,
)
from core.utils import redis_sources, vector_sources
from core.utils.redis_client import get_async_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["shares"])

# A fork re-embeds every source the thread cited, on a shared embedding service.
FORKS_PER_HOUR = 30
_SHARE_ERROR_STATUS = {"empty_thread": 400, "too_large": 413}


def _require_signed_in(user_id: str, action: str) -> None:
    if user_id.startswith("guest_"):
        raise HTTPException(status_code=403, detail=f"Sign in to {action}.")


def _require_share_id(share_id: str) -> None:
    # A malformed id can't exist; refusing it here keeps junk out of the query.
    if not SHARE_ID_RE.match(share_id):
        raise HTTPException(status_code=404, detail="Shared conversation not found.")


@router.post("/api/threads/{thread_id}/share")
async def share_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    _require_signed_in(user_id, "share a conversation")
    await assert_thread_access_async(thread_id, user_id)

    row = await asyncio.to_thread(get_thread_row, thread_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found or access denied.")
    # Voice and scheduled-research threads run other agents; a fork could not
    # resume them. A safety-locked thread is not something to publish.
    if row.get("origin"):
        raise HTTPException(status_code=400, detail="This kind of conversation can't be shared.")
    if row.get("is_locked"):
        raise HTTPException(status_code=400, detail="This conversation can't be shared.")
    # Mid-turn, the checkpoint and the UI history disagree about the last exchange.
    if await stream_is_generating(thread_id):
        raise HTTPException(status_code=409, detail="Wait for the answer to finish before sharing.")
    if await acount_shares(user_id) >= MAX_ACTIVE_SHARES:
        raise HTTPException(
            status_code=429,
            detail=f"You have {MAX_ACTIVE_SHARES} shared links. Revoke one to create another.",
        )

    state = await get_agent(None).aget_state({"configurable": {"thread_id": thread_id}})
    values = state.values or {}
    citations = await redis_sources.load_citations_async(thread_id)
    files_meta = await aget_user_files_meta(user_id, attachment_ids(row["messages"]))
    try:
        snap = build_snapshot(
            ui_messages=row["messages"],
            messages=values.get("messages", []),
            files=values.get("files"),
            citations=citations,
            files_meta=files_meta,
        )
    except ShareError as e:
        raise HTTPException(status_code=_SHARE_ERROR_STATUS.get(e.code, 400), detail=str(e))

    share_id = new_share_id()
    await asave_share(
        share_id=share_id, owner_id=user_id, source_thread_id=thread_id,
        title=row.get("title"), snap=snap,
    )
    logger.info(
        f"[share] created share={share_id} owner={user_id} thread={thread_id} "
        f"msgs={snap.n_messages} bytes={snap.size_bytes}"
    )
    return {"share_id": share_id, "path": f"/s/{share_id}", "n_messages": snap.n_messages}


@router.get("/api/shares")
async def list_shares(user_id: str = Depends(get_current_user)):
    _require_signed_in(user_id, "see your shared links")
    return {"shares": await alist_shares(user_id)}


@router.delete("/api/shares/{share_id}")
async def revoke_share(share_id: str, user_id: str = Depends(get_current_user)):
    _require_share_id(share_id)
    if not await adelete_share(share_id, user_id):
        raise HTTPException(status_code=404, detail="Shared link not found.")
    return {"status": "revoked"}


@router.get("/api/shared/{share_id}")
async def get_shared(share_id: str, response: Response):
    _require_share_id(share_id)
    row = await aget_share_public(share_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Shared conversation not found.")
    # Short, so a revoked link stops resolving within the minute rather than
    # whenever some cache decides to forget it.
    response.headers["Cache-Control"] = "public, max-age=30"
    return {
        "title": row["title"],
        "messages": row["ui_messages"],
        "n_messages": row["n_messages"],
        "created_at": row["created_at"],
    }


async def _check_fork_rate(user_id: str) -> None:
    r = get_async_redis()
    key = f"omni:fork_rate:{user_id}"
    n = await r.incr(key)
    if n == 1:
        await r.expire(key, 3600)
    if n > FORKS_PER_HOUR:
        raise HTTPException(status_code=429, detail="Too many copies in the last hour. Try again later.")


@router.post("/api/shared/{share_id}/fork")
async def fork_shared(share_id: str, user_id: str = Depends(get_current_user)):
    _require_signed_in(user_id, "continue this conversation")
    _require_share_id(share_id)
    await _check_fork_rate(user_id)

    share = await aget_share_full(share_id)
    if share is None:
        raise HTTPException(status_code=404, detail="Shared conversation not found.")

    thread_id = str(uuid.uuid4())
    messages, files = deserialize_state(share["agent_state"])
    citations: list[dict] = share["citations"] or []

    # New ids for the attachment metadata copies, remapped in the visible history.
    id_map = {m["file_id"]: f"shared-copy/{uuid.uuid4()}" for m in share["files_meta"]}
    ui_messages = remap_attachment_ids(share["ui_messages"], id_map)
    file_rows = [(id_map[m["file_id"]], m) for m in share["files_meta"]]

    try:
        # The agent's memory of the conversation: one checkpoint holding the whole
        # state. (Not a replayed history — so turns before this point cannot be
        # rewound; thread_forks.inherited_messages tells the UI which ones.)
        await get_agent(None).aupdate_state(
            {"configurable": {"thread_id": thread_id}},
            {"messages": messages, "files": files},
        )
        for rec in citations:
            await asyncio.to_thread(redis_sources.persist_citation, thread_id, rec)
        await aregister_fork(
            thread_id=thread_id, user_id=user_id, share_id=share_id,
            title=share["title"], ui_messages=ui_messages, files=file_rows,
        )
    except Exception:
        logger.exception(f"[share] fork of {share_id} failed; cleaning up thread {thread_id}")
        # Clears whatever got written: checkpoint, rewind points, citations,
        # vector chunks, and the threads_control row if the transaction got that far.
        await asyncio.to_thread(delete_thread_state, thread_id)
        raise HTTPException(status_code=500, detail="Couldn't copy this conversation. Please try again.")

    # Last, and fire-and-forget: the source chunks behind "check source". The
    # fork is usable before they finish; that feature fills in a few seconds later.
    for rec in citations:
        vector_sources.enqueue_source_indexing(thread_id, rec)

    logger.info(f"[share] forked share={share_id} -> thread={thread_id} user={user_id}")
    return {"thread_id": thread_id, "title": share["title"], "inherited_messages": len(ui_messages)}
