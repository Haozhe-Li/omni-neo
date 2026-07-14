"""Redis Stream utilities for background generation buffering.

Events are stored in a Redis Stream keyed by thread_id. A separate status key
tracks whether generation is in progress, done, or errored. Consumers start from
position 0 to replay the full history, then continue live until "done".

Backed by Upstash Redis over its HTTP REST API (no long-lived TCP connection).
The REST API has no blocking XREAD, so the live tail in `stream_read` polls with
a short sleep between non-blocking reads instead of `XREAD ... BLOCK`.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from upstash_redis.asyncio import Redis

_client: Redis | None = None

STREAM_TTL_ACTIVE = 7200   # 2-hour cap while generating (orphan guard)
STREAM_TTL_DONE   = 600    # 10 minutes after completion

# Upstash's REST API has no blocking XREAD, so `stream_read` tails by polling.
# The cadence is adaptive: poll fast right after an event so the next burst of
# tokens is delivered near-instantly (a fixed 0.5 s poll added up to 0.5 s of
# latency to every token that landed in a quiet gap — between tool calls,
# reasoning pauses, etc. — which compounded to several seconds over a turn),
# then back off once the stream has been quiet for a while (e.g. a long tool
# call produces nothing to deliver anyway) to avoid hammering the REST endpoint.
_POLL_FAST = 0.05          # cadence just after activity (catches the next burst)
_POLL_SLOW = 0.5           # cadence once the stream has gone sustainedly idle
_FAST_POLLS = 20           # keep polling fast for ~1 s (20 × 50 ms) after an event
_STATUS_INTERVAL = 0.5     # min seconds between terminal-status checks while idle
_ORPHAN_TIMEOUT = 60.0     # give up on a stream idle this long (backend restart)


def _get_redis() -> Redis:
    global _client
    if _client is None:
        # Reads UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN. Each call is
        # an independent HTTPS request, so there's no socket to keep alive or
        # retry — a transient blip fails one request and the caller's own
        # retry/idle loop rides over it.
        _client = Redis.from_env()
    return _client


def _sk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}"


def _stk(thread_id: str) -> str:
    return f"omni:stream:{thread_id}:status"


def _entry_data(fields: list) -> str | None:
    """Pull the "data" value out of an Upstash XREAD entry's flat field list.

    Upstash returns each entry's fields as a flat ``[field, value, ...]`` list
    (Redis wire format), not the dict redis-py builds. We only ever write a
    single "data" field, so map it back here.
    """
    for i in range(0, len(fields) - 1, 2):
        if fields[i] == "data":
            return fields[i + 1]
    return None


async def stream_write_batch(thread_id: str, events: list[str]) -> None:
    """Write multiple SSE events to the Redis Stream in a single pipeline round-trip.

    Combines all xadd calls plus one expire into one HTTP request, eliminating the
    per-event RTT that would otherwise throttle fast models like Cerebras.
    """
    if not events:
        return
    r = _get_redis()
    key = _sk(thread_id)
    pipe = r.pipeline()
    for event in events:
        pipe.xadd(key, "*", {"data": event}, maxlen=10000)
    pipe.expire(key, STREAM_TTL_ACTIVE)
    await pipe.exec()


async def stream_write(thread_id: str, event: str) -> None:
    await stream_write_batch(thread_id, [event])


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


async def stream_begin(thread_id: str) -> None:
    """Start a fresh turn in one round trip: drop the previous turn's buffered
    events + status, then mark this turn generating.

    Pipelines the DEL and the SET into a single HTTP request (executed in order
    server-side, so the delete can't clobber the status we set right after),
    replacing the separate stream_reset + stream_set_status round trips.
    """
    r = _get_redis()
    pipe = r.pipeline()
    pipe.delete(_sk(thread_id), _stk(thread_id))
    pipe.set(_stk(thread_id), "generating", ex=STREAM_TTL_ACTIVE)
    await pipe.exec()


async def stream_read(thread_id: str) -> AsyncGenerator[str, None]:
    """Yield all buffered SSE strings then live events until generation ends.

    Safe for reconnect: starts from position 0, replays the full buffered history.
    Handles orphaned streams (e.g. backend restart) by timing out after 60 s idle.

    Upstash's REST API has no blocking XREAD, so this tails by adaptive polling
    (see the cadence constants above): a tight loop while events are flowing,
    fast polling for ~1 s after the last event so the next burst is picked up
    near-instantly, then a slow poll once sustainedly idle. Terminal status is
    checked at most every `_STATUS_INTERVAL`, not on every fast poll.
    """
    r = _get_redis()
    key = _sk(thread_id)
    last_id = "0-0"
    empty_polls = 0
    now = time.monotonic()
    idle_deadline = now + _ORPHAN_TIMEOUT
    last_status_check = 0.0

    while True:
        entries = await r.xread({key: last_id}, count=50)
        if entries:
            empty_polls = 0
            idle_deadline = time.monotonic() + _ORPHAN_TIMEOUT
            for _, messages in entries:
                for msg_id, fields in messages:
                    data = _entry_data(fields)
                    if data is not None:
                        yield data
                    last_id = msg_id
            continue  # read again immediately while the stream is producing

        empty_polls += 1
        now = time.monotonic()
        # Check for terminal status on the first empty poll (fast-path a stream
        # that's already done — e.g. a reconnect after completion) and at most
        # every _STATUS_INTERVAL thereafter.
        if empty_polls == 1 or now - last_status_check >= _STATUS_INTERVAL:
            last_status_check = now
            status = await stream_get_status(thread_id)
            if status in ("done", "error", None):
                # Drain anything written between our last read and the status check.
                tail = await r.xread({key: last_id}, count=1000)
                if tail:
                    for _, messages in tail:
                        for msg_id, fields in messages:
                            data = _entry_data(fields)
                            if data is not None:
                                yield data
                break
        if now >= idle_deadline:
            break
        await asyncio.sleep(_POLL_FAST if empty_polls <= _FAST_POLLS else _POLL_SLOW)
