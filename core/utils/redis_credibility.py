"""Async Redis-backed domain -> credibility label cache.

Deliberately NOT built on `redis_cache.l1cache`: l1cache memoizes a whole
function call under an opaque md5-of-arguments key, which makes the cache
impossible to inspect or purge by domain. This module keys directly on the
domain under its own human-readable prefix (`omni:credibility:domain:<host>`)
and exposes batched get/set so `source_credibility.py` can resolve a whole
search result page with one round trip instead of one per source.
"""
from __future__ import annotations

import os

import redis.asyncio as aioredis

_PREFIX = "omni:credibility:domain:"

# Whitelist-tier domains (regex hits + seed list) are rarely worth
# re-checking. Junk gets a much shorter TTL: a bad domain's ownership or
# content can change (or the classification can simply have been wrong),
# and a stale "junk" verdict is more harmful to leave uncorrected than a
# stale "trusted" one.
TTL_TRUSTED = 3600 * 24 * 365
TTL_JUNK = 3600 * 24 * 10


class CredibilityRedis:
    def __init__(self, prefix: str = _PREFIX):
        self._prefix = prefix
        self._client: aioredis.Redis | None = None

    def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.Redis.from_url(
                os.environ["REDIS_URL"], decode_responses=True
            )
        return self._client

    def _key(self, domain: str) -> str:
        return f"{self._prefix}{domain}"

    async def get_many(self, domains: list[str]) -> dict[str, str]:
        """Batched lookup. Returns only the domains that were cached."""
        if not domains:
            return {}
        keys = [self._key(d) for d in domains]
        values = await self._get_client().mget(keys)
        return {domain: value for domain, value in zip(domains, values) if value}

    async def set_many(self, entries: dict[str, str], ttl: int) -> None:
        """Write every entry with the same `ttl` in one pipelined round trip.

        Callers batch trusted/junk writes separately since the two labels
        use different TTLs.
        """
        if not entries:
            return
        pipe = self._get_client().pipeline(transaction=False)
        for domain, label in entries.items():
            pipe.set(self._key(domain), label, ex=ttl)
        await pipe.execute()


credibility_redis = CredibilityRedis()
