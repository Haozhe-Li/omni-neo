"""Redis Stream utilities for background generation buffering.

Events are stored in a Redis Stream keyed by thread_id. A separate status key
tracks whether generation is in progress, done, or errored. Consumers start from
position 0 to replay the full history, then continue live until "done".

Backed by the application's own Redis over TCP (see core/utils/redis_client.py).
That matters here more than anywhere else in the codebase: this is the one
module that needs a *blocking* XREAD. Over Upstash's HTTP REST API there is no
BLOCK, so the live tail had to poll, and every event that landed in a quiet gap
— between tool calls, during reasoning pauses, before the first token — waited
out the poll interval. An adaptive cadence (50 ms right after activity, backing
off to 500 ms once idle) got the median down but could not remove it. A
blocking read removes it entirely: the server pushes the moment XADD lands.
"""
from __future__ import annotations

import time
from typing import AsyncGenerator

import redis.asyncio as aioredis

from core.utils.redis_client import get_async_redis, get_blocking_redis

STREAM_TTL_ACTIVE = 7200   # 2-hour cap while generating (orphan guard)
STREAM_TTL_DONE   = 600    # 10 minutes after completion

# How long one XREAD parks waiting for the next event. This is not a latency
# knob — an event arriving mid-block wakes the read immediately — it only
# bounds how often an *idle* reader loops round to re-check terminal status and
# the orphan deadline.
_BLOCK_MS = 500
_ORPHAN_TIMEOUT = 60.0     # give up on a stream idle this long (backend restart)


def _get_redis() -> aioredis.Redis:
    return get_async_redis()


def _sk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}"


def _stk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}:status"


async def stream_write_batch(thread_id: str, events: list[str]) -> None:
    """Write multiple SSE events to the Redis Stream in a single round trip.

    Combines all xadd calls plus one expire into one pipeline, eliminating the
    per-event RTT that would otherwise throttle fast models like Cerebras.
    """
    if not events:
        return
    r = _get_redis()
    key = _sk(thread_id)
    pipe = r.pipeline()
    for event in events:
        pipe.xadd(key, {"data": event}, maxlen=10000, approximate=True)
    pipe.expire(key, STREAM_TTL_ACTIVE)
    await pipe.execute()


async def stream_write(thread_id: str, event: str) -> None:
    await stream_write_batch(thread_id, [event])


async def stream_set_status(thread_id: str, status: str, ttl: int) -> None:
    await _get_redis().set(_stk(thread_id), status, ex=ttl)


async def stream_get_status(thread_id: str) -> str | None:
    return await _get_redis().get(_stk(thread_id))


async def stream_is_generating(thread_id: str) -> bool:
    return await stream_get_status(thread_id) == "generating"


async def stream_expire(thread_id: str) -> None:
    pipe = _get_redis().pipeline()
    pipe.expire(_sk(thread_id), STREAM_TTL_DONE)
    pipe.expire(_stk(thread_id), STREAM_TTL_DONE)
    await pipe.execute()


async def stream_reset(thread_id: str) -> None:
    """Drop any buffered events + status from a previous turn.

    Each turn reuses the same thread-keyed stream, so a new generation must
    start from an empty stream — otherwise stream_read replays the previous
    turn's events (including its terminal `done`), and the client renders the
    old answer instead of the new one.
    """
    await _get_redis().delete(_sk(thread_id), _stk(thread_id))


async def stream_begin(thread_id: str) -> None:
    """Start a fresh turn in one round trip: drop the previous turn's buffered
    events + status, then mark this turn generating.

    Pipelines the DEL and the SET together (executed in order server-side, so
    the delete can't clobber the status we set right after), replacing the
    separate stream_reset + stream_set_status round trips.
    """
    r = _get_redis()
    pipe = r.pipeline()
    pipe.delete(_sk(thread_id), _stk(thread_id))
    pipe.set(_stk(thread_id), "generating", ex=STREAM_TTL_ACTIVE)
    await pipe.execute()


async def stream_read(thread_id: str) -> AsyncGenerator[str, None]:
    """Yield all buffered SSE strings then live events until generation ends.

    Safe for reconnect: starts from position 0, replays the full buffered
    history. Handles orphaned streams (e.g. backend restart) by timing out
    after 60 s idle.

    The first read is non-blocking so an already-finished stream replays its
    history and terminates without waiting out a block; every read after that
    blocks, so a live event is delivered the instant it is written.

    The XREADs run on the dedicated blocking pool — one connection per
    concurrent reader, held for the length of the block — while the status
    checks stay on the ordinary pool (see core/utils/redis_client.py).
    """
    r = get_blocking_redis()
    key = _sk(thread_id)
    last_id = "0-0"
    idle_deadline = time.monotonic() + _ORPHAN_TIMEOUT
    blocking = False

    while True:
        entries = await r.xread(
            {key: last_id}, count=500, block=_BLOCK_MS if blocking else None
        )
        blocking = True
        if entries:
            idle_deadline = time.monotonic() + _ORPHAN_TIMEOUT
            for _, messages in entries:
                for msg_id, fields in messages:
                    data = fields.get("data")
                    if data is not None:
                        yield data
                    last_id = msg_id
            continue  # read again immediately while the stream is producing

        # Nothing arrived within the block — the only two reasons to stop.
        status = await stream_get_status(thread_id)
        if status in ("done", "error", None):
            # Drain anything written between our last read and the status check.
            tail = await r.xread({key: last_id}, count=1000)
            if tail:
                for _, messages in tail:
                    for msg_id, fields in messages:
                        data = fields.get("data")
                        if data is not None:
                            yield data
                        last_id = msg_id
            break
        if time.monotonic() >= idle_deadline:
            break
