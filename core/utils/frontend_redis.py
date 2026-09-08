"""The one Redis client that stays on Upstash — the frontend's database.

Two key families live here, both written by the Next.js frontend and only ever
*read* by this backend:

- ``publish:{id}``  — an Omni Page, written by the frontend's /api/publish
  route, with the page's markdown body under ``answer``.
- ``agent_page:…``  — the benchmark llms.txt, written by the frontend's
  lib/llms-txt.ts.

`core/tools/web_page_reader.py` reads them so the agent can quote first-party
omniknows.xyz content without fetching it over the network (the site blocks
crawlers, so a real fetch fails).

This is deliberately a separate module from core/utils/redis_client.py. It used
to be the same client object — web_page_reader imported `r` from
core/utils/redis_cache.py — which was fine only while the application's own
Redis and the frontend's were the same Upstash database. Now that everything
else has moved to Railway over TCP, sharing a client would silently point these
reads at a database that has no `publish:*` keys: `get` would return None,
`first_party_redis_shortcut` would return None, and the caller would fall back
to fetching the page over the network, where it gets blocked. No exception, no
log — just Omni quietly unable to read its own published pages. Hence the
separate module, and hence this comment.

Moving these keys to Railway means migrating the frontend at the same time,
since it is the writer. Until then, Upstash stays.
"""

from __future__ import annotations

from upstash_redis import Redis

_client: Redis | None = None


def get_frontend_redis() -> Redis:
    """Upstash HTTP REST client (UPSTASH_REDIS_REST_URL / _TOKEN)."""
    global _client
    if _client is None:
        _client = Redis.from_env()
    return _client
