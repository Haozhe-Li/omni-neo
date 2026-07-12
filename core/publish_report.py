"""Publish a scheduled-task report into the same Pages Redis keyspace the
Next.js app writes to from app/api/publish/route.ts, so the emailed link opens
in the exact same Pages UI as a manually-published report.

Written directly from the backend (not by calling back into the Next.js API)
because generate_cover() already lives here — no HTTP hop needed — and this
runs from a background asyncio task with no browser session to authenticate
a call to the Next.js route with anyway.

Deliberately narrower than app/api/publish/route.ts in one way: scheduled
reports are personal by construction, so this only ever adds the page to
omni_pages:user:{userId} — never to the global omni_pages:all sorted set the
public /pages list reads from. (Note: publishToPages=False alone would not
be sufficient — that flag is only honored client-side in pages-client.tsx
today, not by the API routes that read omni_pages:all — so omitting the
global zadd entirely is the actual privacy boundary here.)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import redis.asyncio as aioredis

from core.generate_cover import generate_cover

logger = logging.getLogger(__name__)

_client: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _client


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def publish_report(
    user_id: str,
    thread_id: str,
    title: str,
    report_markdown: str,
    sources: list[dict],
) -> str:
    """Write publish:{id} + omni_pages:user:{userId} and return the page id.

    id is derived from thread_id (not title, unlike the Next.js route) —
    every scheduled run gets its own fresh thread_id, so this is already
    unique per run without risking two runs of the same task (same title)
    colliding on the same page.
    """
    page_id = hashlib.sha256(thread_id.encode()).hexdigest()[:12]
    publish_key = f"publish:{page_id}"

    cover = await asyncio.to_thread(generate_cover, title)

    data = {
        "title": title,
        "answer": report_markdown,
        "sources": sources,
        "userId": user_id,
        "authorName": "Omni AI",
        "authorImage": "",
        "publishedAt": _iso_now(),
        "publishToPages": False,
        "threadId": thread_id,
    }
    if cover.get("image_url"):
        data["coverImage"] = cover["image_url"]

    r = _get_redis()
    now_ms = int(time.time() * 1000)
    pipe = r.pipeline()
    pipe.set(publish_key, json.dumps(data))
    pipe.zadd(f"omni_pages:user:{user_id}", {page_id: now_ms})
    await pipe.execute()

    return page_id
