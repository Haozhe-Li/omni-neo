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

from langgraph.checkpoint.upstash_redis import UpstashRedisSaver
from langgraph.checkpoint.upstash_redis.aio import AsyncUpstashRedisSaver

# Safety-net TTL, refreshed on every checkpoint write (the saver re-sets EXPIRE
# inside aput/aput_writes), so an actively-used thread never expires mid-chat.
# Chat history itself lives in Postgres (user_threads); checkpoints only back
# the rewind/resume feature, so this merely bounds how long an *idle* thread
# stays resumable. 90 days matches the longest thread-retention window
# (db_threads_control). It also plugs a pre-existing leak: cleanup_old_threads
# deletes threads_control rows but never touched checkpoint state, so
# auto-expired threads used to leave their checkpoints behind forever.
_CHECKPOINT_TTL = int(os.getenv("CHECKPOINT_TTL_SECONDS", str(3600 * 24 * 90)))

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
