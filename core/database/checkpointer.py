"""LangGraph checkpointer backed by Upstash Redis over its REST API (HTTP only).

Replaces the previous ``AsyncPostgresSaver`` + psycopg TCP pool. On Cloud Run's
scale-to-zero, an idle TCP connection was silently dropped by the egress NAT;
the next request then blocked ~60s on that dead socket before TCP keepalives
declared it dead — surfacing as a long stall *before* the LLM even started
(nothing in the LangSmith trace, because the checkpointer load hadn't returned
yet). Upstash's stateless HTTP REST API keeps no long-lived connection, so every
checkpoint op is an independent HTTPS request with no stale-connection failure
mode.

Credentials come from ``UPSTASH_REDIS_REST_URL`` / ``UPSTASH_REDIS_REST_TOKEN``.
"""
import os

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.upstash_redis import UpstashRedisSaver as _BaseUpstashRedisSaver
from langgraph.checkpoint.upstash_redis.aio import AsyncUpstashRedisSaver as _BaseAsyncUpstashRedisSaver

# Safety-net TTL, refreshed on every checkpoint write (the saver re-sets EXPIRE
# inside aput/aput_writes), so an actively-used thread never expires mid-chat.
# Chat history itself lives in Postgres (user_threads); checkpoints only back
# the rewind/resume feature, so this merely bounds how long an *idle* thread
# stays resumable. 90 days matches the longest thread-retention window
# (db_threads_control). It also plugs a pre-existing leak: cleanup_old_threads
# deletes threads_control rows but never touched checkpoint state, so
# auto-expired threads used to leave their checkpoints behind forever.
_CHECKPOINT_TTL = int(os.getenv("CHECKPOINT_TTL_SECONDS", str(3600 * 24 * 90)))

# ---------------------------------------------------------------------------
# rewind_points map: thread_id -> {turn: checkpoint_id}
# ---------------------------------------------------------------------------
# /api/threads/{id}/rewind used to find its target checkpoint by walking
# `aget_state_history` backwards from the newest checkpoint until it found one
# whose last message is a HumanMessage — an O(checkpoints-in-thread) scan, at
# ~1 sequential Upstash round trip apiece (alist() also fetches pending_writes
# per candidate). Real threads accumulate far more checkpoints than turns:
# every middleware step that doesn't touch `messages` still gets its own
# "loop"-sourced checkpoint, so a single turn can produce several duplicate,
# state-identical "last message is Human" checkpoints in a row. Measured on
# production data: avg ~30 checkpoints/thread, up to 139, and a 136-checkpoint
# scan took 7.5s — all of it before the LLM call even starts.
#
# This hash lets a rewind jump straight to any turn's boundary checkpoint in
# one HTTP round trip instead of scanning. It's maintained here, at the
# storage layer, by overriding aput/put — that's the one place every write
# funnels through regardless of whether it came from a fresh /chat turn, a
# regenerate, or an edit's aupdate_state, so nothing in the request-handling
# code has to remember to keep it in sync.
#
# Every turn produces several near-identical "last message is Human"
# checkpoints in a row (one per middleware step before the model node
# actually runs), all state-identical — so this just keeps overwriting the
# same turn's map entry as those come in. Harmless: last-write-wins lands on
# whichever of the duplicates is chronologically last, which is just as valid
# a fork point as any other. If the same turn is later edited/regenerated
# again, the new write simply replaces the old pointer — this app doesn't
# track a branch tree, so "rewind to turn N" always means "turn N on
# whatever's the current timeline," not a specific historical fork.
#
# `turn` must be threaded into the LangGraph call's config["configurable"]
# for this to have anything to key on (see core/stream.py) — checkpoints
# written without it (e.g. the scheduled-task agent, which has no rewind UI)
# are simply never recorded here, which is fine, nothing reads them.
#
# `messages` is a delta channel: its value is never in aput()'s own
# `checkpoint["channel_values"]` (LangGraph avoids re-snapshotting the whole,
# ever-growing message list into every checkpoint — confirmed by tracing a
# real run: none of a turn's ~7 checkpoints carry a "messages" key). The
# actual HumanMessage/AIMessage content only ever appears as a
# (channel, value) pair passed to `aput_writes`, tied to the checkpoint_id
# it's *pending against* — i.e. the parent of the checkpoint that will
# result once the write is applied. So: catch the fresh-HumanMessage write in
# aput_writes and stash it per-thread; the very next aput() for that same
# thread+ns is exactly the checkpoint the write was just folded into (writes
# for a superstep are always flushed before that superstep's single
# resulting checkpoint — also confirmed by tracing).
_REWIND_POINTS_PREFIX = "omni:rewind_points:"

# (thread_id, checkpoint_ns) -> turn, for a HumanMessage write not yet
# resolved to a checkpoint_id. In-process only: aput_writes and the aput
# that consumes it always happen back-to-back within the same run.
_pending_human_turn: dict[tuple[str, str], int] = {}


def _rewind_points_key(thread_id: str) -> str:
    return f"{_REWIND_POINTS_PREFIX}{thread_id}"


def _note_pending_human_write(config: dict, writes) -> None:
    configurable = config.get("configurable", {})
    if configurable.get("checkpoint_ns", ""):
        return  # subgraph write — not a user-facing turn boundary
    turn = configurable.get("turn")
    if turn is None:
        return
    for channel, value in writes:
        if channel == "messages" and isinstance(value, list) and value and isinstance(value[-1], HumanMessage):
            _pending_human_turn[(configurable["thread_id"], "")] = turn


def _consume_pending_human_write(config: dict, checkpoint: dict) -> tuple[str, int, str] | None:
    configurable = config.get("configurable", {})
    if configurable.get("checkpoint_ns", ""):
        return None
    thread_id = configurable.get("thread_id")
    turn = _pending_human_turn.pop((thread_id, ""), None)
    if turn is None:
        return None
    return thread_id, turn, checkpoint["id"]


async def _arecord_rewind_point(client, ttl, thread_id: str, turn: int, checkpoint_id: str) -> None:
    pipe = client.pipeline()
    pipe.hset(_rewind_points_key(thread_id), values={str(turn): checkpoint_id})
    if ttl is not None:
        pipe.expire(_rewind_points_key(thread_id), ttl)
    await pipe.exec()


def _record_rewind_point(client, ttl, thread_id: str, turn: int, checkpoint_id: str) -> None:
    pipe = client.pipeline()
    pipe.hset(_rewind_points_key(thread_id), values={str(turn): checkpoint_id})
    if ttl is not None:
        pipe.expire(_rewind_points_key(thread_id), ttl)
    pipe.exec()


class AsyncUpstashRedisSaver(_BaseAsyncUpstashRedisSaver):
    async def aput_writes(self, config, writes, task_id, task_path=""):
        _note_pending_human_write(config, writes)
        return await super().aput_writes(config, writes, task_id, task_path)

    async def aput(self, config, checkpoint, metadata, new_versions):
        result = await super().aput(config, checkpoint, metadata, new_versions)
        boundary = _consume_pending_human_write(config, checkpoint)
        if boundary is not None:
            thread_id, turn, checkpoint_id = boundary
            await _arecord_rewind_point(self.client, self.ttl, thread_id, turn, checkpoint_id)
        return result


class UpstashRedisSaver(_BaseUpstashRedisSaver):
    def put_writes(self, config, writes, task_id, task_path=""):
        _note_pending_human_write(config, writes)
        return super().put_writes(config, writes, task_id, task_path)

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        boundary = _consume_pending_human_write(config, checkpoint)
        if boundary is not None:
            thread_id, turn, checkpoint_id = boundary
            _record_rewind_point(self.client, self.ttl, thread_id, turn, checkpoint_id)
        return result


async def get_rewind_checkpoint_id(thread_id: str, turn: int) -> str | None:
    """O(1) lookup: the checkpoint_id for `turn`'s boundary, if recorded."""
    if checkpointer is None:
        return None
    return await checkpointer.client.hget(_rewind_points_key(thread_id), str(turn))


async def backfill_rewind_point(thread_id: str, turn: int, checkpoint_id: str) -> None:
    """Record a turn boundary found via the fallback scan, so the next
    rewind to this turn is O(1). Best-effort — a failed backfill just means
    the next rewind falls back to scanning again, no correctness impact."""
    if checkpointer is None:
        return
    try:
        await _arecord_rewind_point(checkpointer.client, checkpointer.ttl, thread_id, turn, checkpoint_id)
    except Exception:
        pass


def delete_rewind_points(thread_id: str) -> None:
    """Drop a hard-deleted thread's rewind-points map (sync — called from the
    same sync thread-delete path as _delete_checkpoint_state)."""
    sync_checkpointer.client.delete(_rewind_points_key(thread_id))


# Async saver — used by the LangGraph agents (graph.ainvoke / graph.astream).
# Assigned in setup_checkpointer() so main.py's lifespan keeps its existing
# `await setup_checkpointer()` call shape.
checkpointer: AsyncUpstashRedisSaver | None = None

# Sync saver — used only by the synchronous checkpoint-delete path in
# db_threads_control (thread hard-delete), which runs in FastAPI's threadpool
# with no event loop to await an async saver into.
sync_checkpointer: UpstashRedisSaver = UpstashRedisSaver.from_env(ttl=_CHECKPOINT_TTL)


async def setup_checkpointer() -> None:
    global checkpointer
    checkpointer = AsyncUpstashRedisSaver.from_env(ttl=_CHECKPOINT_TTL)


async def teardown_checkpointer() -> None:
    # HTTP REST client — no connection pool to close.
    return None
