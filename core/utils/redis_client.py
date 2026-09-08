"""Shared Redis (TCP) clients — one pool per process, sync and async.

Every module that talks to the application's own Redis goes through here, so
connection settings live in one place and the process holds two pools rather
than one per module.

Why TCP again, after the move to HTTP REST: that move existed to survive Cloud
Run's scale-to-zero, where an idle TCP connection is silently dropped by NAT
and takes ~60s of keepalive to be declared dead. The backend no longer scales
to zero, so a long-lived socket is stable — and it buys back the one thing REST
cannot do, a blocking XREAD (see core/redis_stream.py).

`decode_responses=True` throughout: every caller here stores JSON or plain
strings and used to receive `str` from the Upstash REST client, so decoding at
the client keeps those call sites unchanged.

Deliberately NOT here: the Upstash client in core/utils/frontend_redis.py. That
one reads keys the *frontend* writes, which live in a different database
entirely — see that module.
"""

from __future__ import annotations

import os

import redis
import redis.asyncio as aioredis

_URL_ENV = "REDIS_URL"

# Long-lived sockets need liveness checking; without it a connection killed by
# an upstream idle timeout surfaces as a failed command on first reuse rather
# than being reconnected transparently.
_KWARGS = dict(
    decode_responses=True,
    socket_connect_timeout=10,
    socket_keepalive=True,
    health_check_interval=30,
    retry_on_timeout=True,
)

# Two async pools, deliberately. `core/redis_stream.py` tails with a blocking
# XREAD, and a blocked command owns its connection for the whole block — one
# per concurrent SSE reader, held for as long as that generation runs. Sharing
# a pool with ordinary traffic means a burst of concurrent chats can starve the
# cache, citation and status reads behind them, and redis-py's default cap of
# 100 connections is low enough for that to happen in normal use. Blocking
# commands therefore get their own, generously sized pool; nothing else may use
# it.
_MAX_CONNECTIONS = 128
_MAX_BLOCKING_CONNECTIONS = 512

_sync_client: redis.Redis | None = None
_async_client: aioredis.Redis | None = None
_blocking_client: aioredis.Redis | None = None


def _url() -> str:
    url = os.environ.get(_URL_ENV)
    if not url:
        raise RuntimeError(f"{_URL_ENV} is not set")
    return url


def get_redis() -> redis.Redis:
    """Process-wide sync client. Safe to call from any thread."""
    global _sync_client
    if _sync_client is None:
        _sync_client = redis.Redis.from_url(
            _url(), max_connections=_MAX_CONNECTIONS, **_KWARGS
        )
    return _sync_client


def get_async_redis() -> aioredis.Redis:
    """Process-wide async client for ordinary, non-blocking commands."""
    global _async_client
    if _async_client is None:
        _async_client = aioredis.Redis.from_url(
            _url(), max_connections=_MAX_CONNECTIONS, **_KWARGS
        )
    return _async_client


def get_blocking_redis() -> aioredis.Redis:
    """Async client reserved for blocking commands (XREAD BLOCK).

    No `socket_timeout` is set anywhere in `_KWARGS`, which matters most here: a
    socket timeout shorter than the block would abort the read. Liveness is
    handled by `health_check_interval` instead.
    """
    global _blocking_client
    if _blocking_client is None:
        _blocking_client = aioredis.Redis.from_url(
            _url(), max_connections=_MAX_BLOCKING_CONNECTIONS, **_KWARGS
        )
    return _blocking_client


async def close_async_redis() -> None:
    """Release both async pools (app shutdown)."""
    global _async_client, _blocking_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
    if _blocking_client is not None:
        await _blocking_client.aclose()
        _blocking_client = None
