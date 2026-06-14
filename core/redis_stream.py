"""Redis Stream utilities for background generation buffering.

Events are stored in a Redis Stream keyed by thread_id. A separate status key
tracks whether generation is in progress, done, or errored. Consumers start from
position 0 to replay the full history, then continue live until "done".
"""
from __future__ import annotations

import os
from typing import AsyncGenerator

import redis.asyncio as aioredis

_client: aioredis.Redis | None = None

STREAM_TTL_ACTIVE = 7200   # 2-hour cap while generating (orphan guard)
STREAM_TTL_DONE   = 600    # 10 minutes after completion


def _get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _client


def _sk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}"


def _stk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}:status"


async def stream_write(thread_id: str, event: str) -> None:
    r = _get_redis()
    key = _sk(thread_id)
    await r.xadd(key, {"data": event}, maxlen=10000)
    await r.expire(key, STREAM_TTL_ACTIVE)


async def stream_set_status(thread_id: str, status: str, ttl: int) -> None:
    await _get_redis().set(_stk(thread_id), status, ex=ttl)


async def stream_get_status(thread_id: str) -> str | None:
    return await _get_redis().get(_stk(thread_id))


async def stream_is_generating(thread_id: str) -> bool:
    return await stream_get_status(thread_id) == "generating"


async def stream_expire(thread_id: str) -> None:
    r = _get_redis()
    await r.expire(_sk(thread_id), STREAM_TTL_DONE)
    await r.expire(_stk(thread_id), STREAM_TTL_DONE)


async def stream_reset(thread_id: str) -> None:
    """Drop any buffered events + status from a previous turn.

    Each turn reuses the same thread-keyed stream, so a new generation must
    start from an empty stream — otherwise stream_read replays the previous
    turn's events (including its terminal `done`), and the client renders the
    old answer instead of the new one.
    """
    r = _get_redis()
    await r.delete(_sk(thread_id), _stk(thread_id))


async def stream_read(thread_id: str) -> AsyncGenerator[str, None]:
    """Yield all buffered SSE strings then live events until generation ends.

    Safe for reconnect: starts from position 0, replays the full buffered history.
    Handles orphaned streams (e.g. backend restart) by timing out after 60 s idle.
    """
    r = _get_redis()
    key = _sk(thread_id)
    last_id = "0-0"
    idle_ticks = 0  # each tick = 500 ms; 120 ticks = 60 s orphan timeout

    while True:
        entries = await r.xread({key: last_id}, count=50, block=500)
        if entries:
            idle_ticks = 0
            for _, messages in entries:
                for msg_id, fields in messages:
                    yield fields["data"]
                    last_id = msg_id
        else:
            idle_ticks += 1
            status = await stream_get_status(thread_id)
            if status in ("done", "error", None) or idle_ticks >= 120:
                # Drain any events written between our last read and the status check
                tail = await r.xread({key: last_id}, count=1000)
                if tail:
                    for _, messages in tail:
                        for msg_id, fields in messages:
                            yield fields["data"]
                break
