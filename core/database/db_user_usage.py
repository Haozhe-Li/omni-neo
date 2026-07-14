"""
Database operations for the user_usage table — a unified credit ledger for
both guests and signed-in users, replacing the old pro-only `guest_usage`
table.

Table schema:
    CREATE TABLE IF NOT EXISTS user_usage (
        user_id VARCHAR(255) PRIMARY KEY,
        day DATE NOT NULL DEFAULT CURRENT_DATE,
        day_used NUMERIC(10,2) NOT NULL DEFAULT 0,
        month VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
        month_used NUMERIC(10,2) NOT NULL DEFAULT 0,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

Model:
    - Every /chat or /rewind call spends credits: fast=1, pro=4.7 — and a
      scheduled research run (core/routers/scheduled_tasks.py) also spends
      4.7, charged against the same ledger (MODE_CREDIT_COST). Costs are
      fractional, hence NUMERIC columns rather than INT.
    - Two independent caps apply at once: a daily one and a calendar-month one.
      Both are tracked in the same row so a single charge is one round trip.
    - Limits differ for guests vs signed-in users (see *_CREDIT_LIMIT below),
      keyed the same way every other per-user table in this codebase is:
      `user_id` is either a Clerk sub or a `guest_<uuid>` string.
    - Charging is all-or-nothing: if applying the cost would push either the
      day or month counter over its limit, nothing is charged at all (the
      request should be rejected outright, not partially billed).

Note: NUMERIC values can come back over PostgREST as strings (to preserve
precision), so every value read from a day_used/month_used column is cast to
float immediately (see _rolled_over_usage) so the rest of this module and its
callers can treat usage as plain floats throughout.
"""

import asyncio
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

from core.database.supabase_client import supabase, get_async_supabase, utcnow_iso

logger = logging.getLogger(__name__)

USER_DAILY_CREDIT_LIMIT: int = int(os.getenv("USER_DAILY_CREDIT_LIMIT", "100"))
USER_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("USER_MONTHLY_CREDIT_LIMIT", "3000"))
GUEST_DAILY_CREDIT_LIMIT: int = int(os.getenv("GUEST_DAILY_CREDIT_LIMIT", "20"))
GUEST_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("GUEST_MONTHLY_CREDIT_LIMIT", "300"))

MODE_CREDIT_COST: dict[str, float] = {"fast": 1.0, "pro": 4.7, "scheduled": 4.7}


def _limits(user_id: str) -> tuple[int, int]:
    """Return (day_limit, month_limit) for this user_id's tier."""
    if user_id.startswith("guest_"):
        return GUEST_DAILY_CREDIT_LIMIT, GUEST_MONTHLY_CREDIT_LIMIT
    return USER_DAILY_CREDIT_LIMIT, USER_MONTHLY_CREDIT_LIMIT


def _reset_times(today: date) -> tuple[str, str]:
    """ISO-8601 UTC timestamps for the next daily and monthly rollover."""
    tomorrow = today + timedelta(days=1)
    resets_day_at = datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=timezone.utc
    ).isoformat()
    if today.month == 12:
        next_month_first = date(today.year + 1, 1, 1)
    else:
        next_month_first = date(today.year, today.month + 1, 1)
    resets_month_at = datetime(
        next_month_first.year, next_month_first.month, next_month_first.day,
        tzinfo=timezone.utc,
    ).isoformat()
    return resets_day_at, resets_month_at


def _rolled_over_usage(row: dict | None, today_iso: str, month: str) -> tuple[float, float]:
    """Current day/month usage from a stored row, reset to 0 if its stored
    day/month has since rolled over."""
    if not row:
        return 0.0, 0.0
    day_used = float(row["day_used"]) if row.get("day") == today_iso else 0.0
    month_used = float(row["month_used"]) if row.get("month") == month else 0.0
    return day_used, month_used


def _charge_context(user_id: str, mode: str) -> dict:
    """Per-call constants shared by the sync and async charge paths."""
    today = datetime.now(timezone.utc).date()
    resets_day_at, resets_month_at = _reset_times(today)
    day_limit, month_limit = _limits(user_id)
    return {
        "cost": MODE_CREDIT_COST[mode],
        "day_limit": day_limit, "month_limit": month_limit,
        "today_iso": today.isoformat(), "month": today.strftime("%Y-%m"),
        "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
    }


def _charge_decision(user_id: str, row: dict | None, ctx: dict) -> tuple[dict | None, dict]:
    """Pure decision: given the current row, return (write_payload | None, result).

    Shared by charge_credits and charge_credits_async so the limit/rollover
    logic lives in exactly one place; only the I/O around it differs.
    """
    day_used, month_used = _rolled_over_usage(row, ctx["today_iso"], ctx["month"])
    new_day = day_used + ctx["cost"]
    new_month = month_used + ctx["cost"]
    common = {
        "day_limit": ctx["day_limit"], "month_limit": ctx["month_limit"],
        "resets_day_at": ctx["resets_day_at"], "resets_month_at": ctx["resets_month_at"],
    }
    if new_day <= ctx["day_limit"] and new_month <= ctx["month_limit"]:
        write = {
            "user_id": user_id,
            "day": ctx["today_iso"], "day_used": new_day,
            "month": ctx["month"], "month_used": new_month,
            "updated_at": utcnow_iso(),
        }
        return write, {"charged": True, "day_used": new_day, "month_used": new_month,
                       "exceeded_scope": None, **common}
    day_over = new_day > ctx["day_limit"]
    month_over = new_month > ctx["month_limit"]
    scope = "both" if (day_over and month_over) else ("day" if day_over else "month")
    return None, {"charged": False, "day_used": day_used, "month_used": month_used,
                  "exceeded_scope": scope, **common}


def _charge_failopen(ctx: dict) -> dict:
    """Fail-open result on a DB error — a usage-ledger hiccup should degrade
    gracefully (let the turn through), not take chat down with it."""
    return {
        "charged": True, "day_used": 0.0, "month_used": 0.0,
        "day_limit": ctx["day_limit"], "month_limit": ctx["month_limit"],
        "resets_day_at": ctx["resets_day_at"], "resets_month_at": ctx["resets_month_at"],
        "exceeded_scope": None,
    }


def charge_credits(user_id: str, mode: str) -> dict:
    """
    Charge `mode`'s credit cost against user_id's daily and monthly usage,
    charging only if BOTH limits still hold after the charge (rollover-aware).

    Over PostgREST there is no single-statement atomic upsert-with-guard the
    way the old psycopg version had, so this is read-modify-write: read the
    row, decide, then write. The window is tiny and same-user charges are
    effectively never concurrent (a user's requests are serialized by the UI),
    so worst case is a bounded, self-correcting over-count, never a hard fail.
    Sync variant — use `charge_credits_async` from the async chat path.
    """
    ctx = _charge_context(user_id, mode)
    try:
        res = (
            supabase.table("user_usage")
            .select("day, day_used, month, month_used")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        write, result = _charge_decision(user_id, res.data[0] if res.data else None, ctx)
        if write is not None:
            supabase.table("user_usage").upsert(write, on_conflict="user_id").execute()
        return result
    except Exception as e:
        logger.error(f"[db_user_usage] charge_credits error for {user_id}: {e}")
        return _charge_failopen(ctx)


async def evaluate_charge_async(user_id: str, mode: str) -> tuple[dict, dict | None]:
    """Read usage + decide, WITHOUT committing the write. Returns
    ``(result, write_payload)`` where `write_payload` is None when nothing
    should be written (not charged, or fail-open).

    Splitting the read from the write lets the chat handler fold the read into
    one concurrent batch with the other pre-LLM lookups, then defer the write
    off the critical path — or skip it entirely (e.g. on a reconnect, which
    must not charge)."""
    ctx = _charge_context(user_id, mode)
    try:
        sb = await get_async_supabase()
        res = (
            await sb.table("user_usage")
            .select("day, day_used, month, month_used")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        write, result = _charge_decision(user_id, res.data[0] if res.data else None, ctx)
        return result, write
    except Exception as e:
        logger.error(f"[db_user_usage] evaluate_charge_async error for {user_id}: {e}")
        return _charge_failopen(ctx), None


async def commit_charge_async(write: dict | None) -> None:
    """Persist a charge decided by evaluate_charge_async. Safe to fire-and-forget
    (usage accuracy is best-effort); a no-op when `write` is None."""
    if not write:
        return
    try:
        sb = await get_async_supabase()
        await sb.table("user_usage").upsert(write, on_conflict="user_id").execute()
    except Exception as e:
        logger.error(f"[db_user_usage] commit_charge_async error: {e}")


async def charge_credits_async(user_id: str, mode: str) -> dict:
    """True-async read-decide-write charge for callers that want it in one shot
    (the read/write don't block the event loop). The chat handler instead uses
    evaluate_charge_fast + commit_charge_fast to keep the DB off the hot path."""
    result, write = await evaluate_charge_async(user_id, mode)
    await commit_charge_async(write)
    return result


# ---------------------------------------------------------------------------
# Latency-first charge path: gate on a local usage snapshot, reconcile in the
# background. Limits become approximate — a burst within the snapshot window can
# slip over — which is an accepted trade for keeping Supabase off the /chat
# critical path (usage accounting is best-effort here).
# ---------------------------------------------------------------------------
_USAGE_SNAPSHOT_TTL = 60.0
_usage_snapshot: dict[str, dict] = {}
_usage_lock = threading.Lock()


def _snapshot_get(user_id: str, today_iso: str, month: str) -> tuple[float, float] | None:
    ent = _usage_snapshot.get(user_id)
    if not ent or ent["expiry"] <= time.monotonic():
        return None
    day_used = ent["day_used"] if ent["day"] == today_iso else 0.0
    month_used = ent["month_used"] if ent["month"] == month else 0.0
    return day_used, month_used


def _snapshot_put(user_id: str, today_iso: str, day_used: float, month: str, month_used: float) -> None:
    with _usage_lock:
        _usage_snapshot[user_id] = {
            "day": today_iso, "day_used": day_used,
            "month": month, "month_used": month_used,
            "expiry": time.monotonic() + _USAGE_SNAPSHOT_TTL,
        }


async def evaluate_charge_fast(user_id: str, mode: str) -> dict:
    """Decide the limit gate against the local usage snapshot — no DB round trip
    on a hit. On a miss (cold process / expired), do one real read to seed the
    snapshot. Returns the usual charge result dict. Pair with commit_charge_fast,
    which the caller invokes only when it actually proceeds (so a reconnect,
    which never calls it, never charges)."""
    ctx = _charge_context(user_id, mode)
    snap = _snapshot_get(user_id, ctx["today_iso"], ctx["month"])
    if snap is not None:
        row = {"day": ctx["today_iso"], "day_used": snap[0],
               "month": ctx["month"], "month_used": snap[1]}
        _, result = _charge_decision(user_id, row, ctx)
        # Optimistic local bump so a rapid burst sees rising usage; the
        # background reconcile re-anchors to DB truth right after.
        if result["charged"]:
            _snapshot_put(user_id, ctx["today_iso"], result["day_used"], ctx["month"], result["month_used"])
        return result
    result, _write = await evaluate_charge_async(user_id, mode)
    _snapshot_put(user_id, ctx["today_iso"], result["day_used"], ctx["month"], result["month_used"])
    return result


def commit_charge_fast(user_id: str, mode: str) -> None:
    """Fire-and-forget reconcile: do the real read+increment+write against the
    DB, then re-anchor the local snapshot to the post-write truth."""
    async def _run():
        try:
            ctx = _charge_context(user_id, mode)
            result, write = await evaluate_charge_async(user_id, mode)
            await commit_charge_async(write)
            _snapshot_put(user_id, ctx["today_iso"], result["day_used"], ctx["month"], result["month_used"])
        except Exception as e:
            logger.error(f"[db_user_usage] commit_charge_fast reconcile error for {user_id}: {e}")

    asyncio.create_task(_run())


def get_usage(user_id: str) -> dict:
    """Read-only snapshot of a user's current usage — no mutation, no charge."""
    day_limit, month_limit = _limits(user_id)
    today = datetime.now(timezone.utc).date()
    today_iso = today.isoformat()
    month = today.strftime("%Y-%m")
    resets_day_at, resets_month_at = _reset_times(today)

    day_used = month_used = 0.0
    try:
        res = (
            supabase.table("user_usage")
            .select("day, day_used, month, month_used")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        row = res.data[0] if res.data else None
        day_used, month_used = _rolled_over_usage(row, today_iso, month)
    except Exception as e:
        logger.error(f"[db_user_usage] get_usage error for {user_id}: {e}")

    return {
        "is_guest": user_id.startswith("guest_"),
        "day_used": day_used, "day_limit": day_limit,
        "day_remaining": max(day_limit - day_used, 0),
        "month_used": month_used, "month_limit": month_limit,
        "month_remaining": max(month_limit - month_used, 0),
        "mode_cost": MODE_CREDIT_COST,
        "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
    }


def delete_user_usage(user_id: str) -> bool:
    """Delete a user's usage row (account purge / guest-merge cleanup)."""
    try:
        res = supabase.table("user_usage").delete().eq("user_id", user_id).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_user_usage] delete_user_usage error: {e}")
        return False
