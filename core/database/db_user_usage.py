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

import logging
import os
from datetime import date, datetime, timedelta, timezone

from core.database.supabase_client import supabase, utcnow_iso

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


def charge_credits(user_id: str, mode: str) -> dict:
    """
    Charge `mode`'s credit cost against user_id's daily and monthly usage.
    Charges only if the totals stay within BOTH limits after the charge
    (day/month rollover-aware) — if either would be exceeded, nothing is
    charged and `charged=False` is returned along with which scope(s) were
    exceeded.

    Over PostgREST there is no single-statement atomic upsert-with-guard the
    way the old psycopg version had, so this is read-modify-write: read the
    row, decide, then write. The window is tiny and same-user charges are
    effectively never concurrent (a user's requests are serialized by the UI),
    so worst case two truly-simultaneous requests could each let the other
    slip just over a limit — a bounded, self-correcting over-count, never a
    hard failure.

    Fails open on a DB error (logs loudly, reports charged=True with zeroed
    usage) — a usage-ledger hiccup should degrade gracefully, not take chat
    down with it.
    """
    cost = MODE_CREDIT_COST[mode]
    day_limit, month_limit = _limits(user_id)
    today = datetime.now(timezone.utc).date()
    today_iso = today.isoformat()
    month = today.strftime("%Y-%m")
    resets_day_at, resets_month_at = _reset_times(today)

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
        new_day = day_used + cost
        new_month = month_used + cost

        if new_day <= day_limit and new_month <= month_limit:
            supabase.table("user_usage").upsert(
                {
                    "user_id": user_id,
                    "day": today_iso, "day_used": new_day,
                    "month": month, "month_used": new_month,
                    "updated_at": utcnow_iso(),
                },
                on_conflict="user_id",
            ).execute()
            return {
                "charged": True,
                "day_used": new_day, "day_limit": day_limit,
                "month_used": new_month, "month_limit": month_limit,
                "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
                "exceeded_scope": None,
            }

        day_over = new_day > day_limit
        month_over = new_month > month_limit
        scope = "both" if (day_over and month_over) else ("day" if day_over else "month")
        return {
            "charged": False,
            "day_used": day_used, "day_limit": day_limit,
            "month_used": month_used, "month_limit": month_limit,
            "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
            "exceeded_scope": scope,
        }
    except Exception as e:
        logger.error(f"[db_user_usage] charge_credits error for {user_id}: {e}")
        return {
            "charged": True,
            "day_used": 0.0, "day_limit": day_limit,
            "month_used": 0.0, "month_limit": month_limit,
            "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
            "exceeded_scope": None,
        }


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
