"""
Database operations for the user_usage table — a unified credit ledger for
both guests and signed-in users, replacing the old pro-only `guest_usage`
table.

Table schema:
    CREATE TABLE IF NOT EXISTS user_usage (
        user_id VARCHAR(255) PRIMARY KEY,
        day DATE NOT NULL DEFAULT CURRENT_DATE,
        day_used INT NOT NULL DEFAULT 0,
        month VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
        month_used INT NOT NULL DEFAULT 0,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

Model:
    - Every /chat or /rewind call spends credits: fast=1, pro=3 (MODE_CREDIT_COST).
    - Two independent caps apply at once: a daily one and a calendar-month one.
      Both are tracked in the same row so a single charge is one round trip.
    - Limits differ for guests vs signed-in users (see *_CREDIT_LIMIT below),
      keyed the same way every other per-user table in this codebase is:
      `user_id` is either a Clerk sub or a `guest_<uuid>` string.
    - Charging is all-or-nothing: if applying the cost would push either the
      day or month counter over its limit, nothing is charged at all (the
      request should be rejected outright, not partially billed).
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone

from core.database.postgresql_saver import sync_pool as pool

logger = logging.getLogger(__name__)

USER_DAILY_CREDIT_LIMIT: int = int(os.getenv("USER_DAILY_CREDIT_LIMIT", "100"))
USER_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("USER_MONTHLY_CREDIT_LIMIT", "3000"))
GUEST_DAILY_CREDIT_LIMIT: int = int(os.getenv("GUEST_DAILY_CREDIT_LIMIT", "20"))
GUEST_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("GUEST_MONTHLY_CREDIT_LIMIT", "300"))

MODE_CREDIT_COST: dict[str, int] = {"fast": 1, "pro": 3}


def setup_user_usage_table() -> None:
    """
    Create user_usage (idempotent) and drop the old guest_usage table it
    replaces — usage is now tracked uniformly for guests and signed-in users
    alike, across both fast and pro modes, so the old pro-only/guest-only
    table no longer serves a purpose. Safe to call on every startup.
    """
    ddl = [
        "DROP TABLE IF EXISTS guest_usage;",
        """
        CREATE TABLE IF NOT EXISTS user_usage (
            user_id VARCHAR(255) PRIMARY KEY,
            day DATE NOT NULL DEFAULT CURRENT_DATE,
            day_used INT NOT NULL DEFAULT 0,
            month VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
            month_used INT NOT NULL DEFAULT 0,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ]
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in ddl:
                    cur.execute(stmt)
    except Exception as e:
        logger.error(f"[db_user_usage] setup_user_usage_table error: {e}")


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


def charge_credits(user_id: str, mode: str) -> dict:
    """
    Atomically charge `mode`'s credit cost against user_id's daily and
    monthly usage, in one round trip. Charges only if the totals stay within
    BOTH limits after the charge (day/month rollover-aware) — if either
    would be exceeded, nothing is charged and `charged=False` is returned
    along with which scope(s) were exceeded.

    Fails open on a DB error (logs loudly, reports charged=True with zeroed
    usage) — a usage-ledger hiccup should degrade gracefully, not take chat
    down with it.
    """
    cost = MODE_CREDIT_COST[mode]
    day_limit, month_limit = _limits(user_id)
    today = datetime.now(timezone.utc).date()
    month = today.strftime("%Y-%m")
    resets_day_at, resets_month_at = _reset_times(today)

    # The CASE branches roll each counter over to `cost` if the stored
    # day/month has moved on, otherwise add `cost` to what's already there.
    # The WHERE clause re-evaluates those same rolled-over totals against the
    # limits — if either would be exceeded, the UPDATE (and therefore the
    # whole statement) simply matches no row, and RETURNING yields nothing.
    # This is single-statement atomic: Postgres locks the row for the
    # duration of this one UPDATE, so concurrent charges from the same user
    # serialize correctly with no separate transaction/lock needed. The
    # INSERT branch (brand-new user) is never blocked by the WHERE guard —
    # harmless since cost (<=3) is always far below any real limit.
    sql = """
        INSERT INTO user_usage (user_id, day, day_used, month, month_used)
        VALUES (%(user_id)s, %(today)s, %(cost)s, %(month)s, %(cost)s)
        ON CONFLICT (user_id) DO UPDATE SET
            day = %(today)s,
            day_used = CASE WHEN user_usage.day = %(today)s
                             THEN user_usage.day_used + %(cost)s ELSE %(cost)s END,
            month = %(month)s,
            month_used = CASE WHEN user_usage.month = %(month)s
                               THEN user_usage.month_used + %(cost)s ELSE %(cost)s END,
            updated_at = NOW()
        WHERE
            (CASE WHEN user_usage.day = %(today)s
                  THEN user_usage.day_used + %(cost)s ELSE %(cost)s END) <= %(day_limit)s
            AND
            (CASE WHEN user_usage.month = %(month)s
                  THEN user_usage.month_used + %(cost)s ELSE %(cost)s END) <= %(month_limit)s
        RETURNING day_used, month_used
    """
    params = {
        "user_id": user_id, "today": today, "month": month, "cost": cost,
        "day_limit": day_limit, "month_limit": month_limit,
    }
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if row is not None:
                    return {
                        "charged": True,
                        "day_used": row["day_used"], "day_limit": day_limit,
                        "month_used": row["month_used"], "month_limit": month_limit,
                        "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
                        "exceeded_scope": None,
                    }
                # Rejected — read back the current (unchanged) totals to report.
                cur.execute(
                    "SELECT day, day_used, month, month_used FROM user_usage WHERE user_id = %s",
                    (user_id,),
                )
                current = cur.fetchone()
                day_used = current["day_used"] if current and current["day"] == today else 0
                month_used = current["month_used"] if current and current["month"] == month else 0
                day_over = day_used + cost > day_limit
                month_over = month_used + cost > month_limit
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
            "day_used": 0, "day_limit": day_limit,
            "month_used": 0, "month_limit": month_limit,
            "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
            "exceeded_scope": None,
        }


def get_usage(user_id: str) -> dict:
    """Read-only snapshot of a user's current usage — no mutation, no charge."""
    day_limit, month_limit = _limits(user_id)
    today = datetime.now(timezone.utc).date()
    month = today.strftime("%Y-%m")
    resets_day_at, resets_month_at = _reset_times(today)

    day_used = month_used = 0
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT day, day_used, month, month_used FROM user_usage WHERE user_id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
                if row:
                    day_used = row["day_used"] if row["day"] == today else 0
                    month_used = row["month_used"] if row["month"] == month else 0
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
    sql = "DELETE FROM user_usage WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_usage] delete_user_usage error: {e}")
        return False
