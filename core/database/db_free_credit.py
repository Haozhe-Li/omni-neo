"""
Backend for the "get free credit" self-serve page (/get-free-credit on the
frontend): a user-exclusive, one-time redeem code handed out after a risk
check, capped to one real check per user per day via a Redis cooldown.

Flow: request_free_credit() first replays any cached result from the last
check (see the Redis key below) — a grant is replayed until it, and the code
it names, expire; a denial is replayed until the next day. Only a cache miss
actually calls _evaluate_risk() and, if approved, mints a code via
create_restricted_code() (core/database/db_redeem_codes.py).

_evaluate_risk is a v1 placeholder by explicit product decision (2026-09-21):
approve every check, first time and every repeat after cooldown. Real signals
(account age, spend history, guest vs signed-in, device/IP dedup, etc.) land
here later without touching request_free_credit or the router.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from core.database import pg
from core.database.db_redeem_codes import create_restricted_code, normalize_code
from core.utils.redis_client import get_async_redis

logger = logging.getLogger(__name__)

_COOLDOWN_KEY_PREFIX = "free_credit:cooldown:"
_DENY_COOLDOWN_S = 24 * 60 * 60           # "no" stands for 1 day
_GRANT_COOLDOWN_S = 3 * 24 * 60 * 60      # "yes" stands for 3 days — matches
                                           # the minted code's own expires_in_days below,
                                           # so the cache never outlives what it names

# Tunable, nothing else in the codebase depends on either value.
FREE_CREDIT_AMOUNT = 20.0
FREE_CREDIT_EXPIRES_DAYS = 3


async def _evaluate_risk(user_id: str) -> bool:
    """v1: always approve. See module docstring."""
    return True


async def request_free_credit(user_id: str) -> dict:
    """Returns one of:
      {"status": "ok", "code": str, "expires_at": iso str}
      {"status": "denied"}
      {"status": "error"}

    Safe to call on every button click — a cached grant or denial from
    today's (or the last 3 days', for a grant) check is replayed without
    touching the risk pipeline again.
    """
    r = get_async_redis()
    key = _COOLDOWN_KEY_PREFIX + user_id

    cached_raw = await r.get(key)
    if cached_raw:
        try:
            cached = json.loads(cached_raw)
        except ValueError:
            cached = None
        if cached and cached.get("status") in ("ok", "denied"):
            return cached

    approved = await _evaluate_risk(user_id)
    if not approved:
        payload = {"status": "denied"}
        await r.set(key, json.dumps(payload), ex=_DENY_COOLDOWN_S)
        return payload

    minted = await asyncio.to_thread(
        create_restricted_code, user_id, FREE_CREDIT_AMOUNT, FREE_CREDIT_EXPIRES_DAYS
    )
    if minted is None:
        # Deliberately not cached: a DB hiccup shouldn't cost the user their
        # once-a-day check, and there's no code/denial to replay anyway.
        logger.error(f"[db_free_credit] mint failed for {user_id}")
        return {"status": "error"}

    code, expires_at = minted
    payload = {"status": "ok", "code": code, "expires_at": expires_at}
    await r.set(key, json.dumps(payload), ex=_GRANT_COOLDOWN_S)
    return payload


async def get_free_credit_status(user_id: str) -> dict:
    """Read-only view of the cooldown, for the page to render before the user
    clicks anything. Never mints and never runs the risk check. Returns one of:

      {"status": "available"}                                   — no check on file; the button may be offered
      {"status": "cooldown", "code": str, "expires_at": iso,
       "redeemed": bool, "available_at": iso}                   — a grant is on file; show its code
      {"status": "denied", "available_at": iso}                 — a denial is on file

    `available_at` is when the cached result lapses and a new check may run
    (the Redis key's remaining TTL), `redeemed` whether the cached code has
    already been claimed by this user.
    """
    r = get_async_redis()
    key = _COOLDOWN_KEY_PREFIX + user_id

    cached_raw = await r.get(key)
    if not cached_raw:
        return {"status": "available"}
    try:
        cached = json.loads(cached_raw)
    except ValueError:
        return {"status": "available"}

    ttl = await r.ttl(key)
    available_at = (datetime.now(timezone.utc) + timedelta(seconds=max(ttl, 0))).isoformat()

    if cached.get("status") == "denied":
        return {"status": "denied", "available_at": available_at}
    if cached.get("status") != "ok" or not cached.get("code"):
        return {"status": "available"}

    redeemed = await asyncio.to_thread(
        pg.fetch_one,
        "SELECT 1 FROM code_redemptions WHERE code = %s AND user_id = %s",
        (normalize_code(cached["code"]), user_id),
    )
    return {
        "status": "cooldown",
        "code": cached["code"],
        "expires_at": cached.get("expires_at"),
        "redeemed": redeemed is not None,
        "available_at": available_at,
    }
