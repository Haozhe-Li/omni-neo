"""
Database operations for the user_usage table — a unified credit ledger for
both guests and signed-in users, replacing the old pro-only `guest_usage`
table.

Table schema (see schema.sql):
    CREATE TABLE IF NOT EXISTS user_usage (
        user_id VARCHAR(255) PRIMARY KEY,
        day DATE NOT NULL DEFAULT CURRENT_DATE,
        day_used NUMERIC(10,2) NOT NULL DEFAULT 0,
        month VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
        month_used NUMERIC(10,2) NOT NULL DEFAULT 0,
        extra_granted NUMERIC(10,2) NOT NULL DEFAULT 0,
        extra_used NUMERIC(10,2) NOT NULL DEFAULT 0,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

Model:
    - Every /chat or /rewind call spends credits, priced by the model the user
      picked: 1 for the fine-tune, 3 for the frontier models (MODE_CREDIT_COST
      below). A scheduled research run (core/routers/scheduled_tasks.py) spends
      4.7 against the same ledger. Costs are fractional, hence NUMERIC columns
      rather than INT.
    - Two independent caps apply at once: a daily one and a calendar-month one.
      Both are tracked in the same row so a single charge is one round trip.
    - Limits differ for guests vs signed-in users (see *_CREDIT_LIMIT below),
      keyed the same way every other per-user table in this codebase is:
      `user_id` is either a Clerk sub or a `guest_<uuid>` string.
    - On top of the two caps sits a THIRD bucket: extra credits redeemed with
      a code (see db_redeem_codes.py). It is a balance, not a cap — it never
      resets, it is spent before the day/month allowances, and what it pays
      for does not count against either cap. A charge takes as much as it can
      from the extra balance and bills only the remainder to day/month, so a
      balance smaller than one pro turn (4.7) can't get stranded.
    - Charging is all-or-nothing: if the part left for day/month would push
      either counter over its limit, nothing is charged at all (the request
      should be rejected outright, not partially billed).

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

# Cost of one charge, keyed by whatever the caller passes as `mode`.
#
# Interactive turns are keyed by **model id** now (see core/chat_models.py) —
# `rix` and a plain `best` turn are 1 credit, every other model is 3, and
# `best-vision` is what the chat router passes when this turn carries an image
# and will therefore be re-routed to gemma. The old `fast`/`pro` keys are kept
# because a client on a stale bundle, or a rewind of a thread created before
# this change, still sends them; both bill as `best` did.
#
# `scheduled` is unrelated to the picker and unchanged: an unattended research
# run is a much bigger job than one chat turn.
MODE_CREDIT_COST: dict[str, float] = {
    "best": 1.0,
    "best-vision": 3.0,
    "rix": 1.0,
    "gemma": 3.0,
    "luna": 3.0,
    "gemini": 3.0,
    "fast": 1.0,
    "pro": 1.0,
    "scheduled": 4.7,
}

# A charge key the table doesn't know must not take chat down: an unpriced
# model bills at the interactive default rather than raising a KeyError deep
# inside the usage path.
_DEFAULT_CREDIT_COST = 3.0


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


# The ledger's columns are NUMERIC(10,2), so Postgres rounds every write to 2dp
# anyway. Rounding here too keeps the in-process snapshot and the stored row
# agreeing to the cent, and stops float residue (4.7 - 2.0 = 2.7000000000000002)
# from accumulating in the extra balance across many charges — an epsilon left
# behind there would otherwise be permanently unspendable.
_EPSILON = 0.005


def _q2(x: float) -> float:
    return round(x + 0.0, 2)


def _extra_balance(row: dict | None) -> tuple[float, float]:
    """(granted, used) for the redeemed-credit bucket.

    No rollover logic on purpose — unlike day_used/month_used these do not
    reset when the stored day/month goes stale. Missing keys read as 0 so a
    row written before the columns existed still charges correctly.
    """
    if not row:
        return 0.0, 0.0
    return float(row.get("extra_granted") or 0), float(row.get("extra_used") or 0)


# ---------------------------------------------------------------------------
# Reading the row. Two shapes, because DDL here is applied by hand in the
# Supabase SQL editor (see schema.sql) and therefore does NOT land atomically
# with a deploy. If code that selects extra_granted/extra_used ships before the
# ALTER runs, PostgREST answers every read with 42703 (undefined_column) — and
# since a read failure fails OPEN, that would silently hand every user
# unlimited free usage until someone noticed. So: try the wide select, and on
# 42703 fall back to the pre-migration columns for the rest of the process's
# life (a restart after the migration picks the wide shape back up).
# ---------------------------------------------------------------------------
_USAGE_COLUMNS_WITH_EXTRA = "day, day_used, month, month_used, extra_granted, extra_used"
_USAGE_COLUMNS_LEGACY = "day, day_used, month, month_used"
_extra_columns_present = True


def _usage_columns() -> str:
    return _USAGE_COLUMNS_WITH_EXTRA if _extra_columns_present else _USAGE_COLUMNS_LEGACY


def _is_missing_extra_columns(e: Exception) -> bool:
    """Whether this error is 'the extra_* columns aren't in the DB yet'."""
    if not _extra_columns_present:
        return False
    code = getattr(e, "code", None)
    text = str(e)
    return (code == "42703" or "42703" in text) and "extra_" in text


def _note_missing_extra_columns() -> None:
    global _extra_columns_present
    if _extra_columns_present:
        _extra_columns_present = False
        logger.warning(
            "[db_user_usage] extra_granted/extra_used missing from user_usage — "
            "falling back to legacy columns. Run the ALTER TABLE statements in "
            "schema.sql; redeemed credits are inert until then."
        )


def _read_usage_row(user_id: str) -> dict | None:
    """Blocking read of a user's ledger row, with the column fallback above."""
    try:
        res = (
            supabase.table("user_usage").select(_usage_columns())
            .eq("user_id", user_id).limit(1).execute()
        )
    except Exception as e:
        if not _is_missing_extra_columns(e):
            raise
        _note_missing_extra_columns()
        res = (
            supabase.table("user_usage").select(_usage_columns())
            .eq("user_id", user_id).limit(1).execute()
        )
    return res.data[0] if res.data else None


async def _read_usage_row_async(user_id: str) -> dict | None:
    """Async twin of `_read_usage_row`."""
    sb = await get_async_supabase()
    try:
        res = (
            await sb.table("user_usage").select(_usage_columns())
            .eq("user_id", user_id).limit(1).execute()
        )
    except Exception as e:
        if not _is_missing_extra_columns(e):
            raise
        _note_missing_extra_columns()
        res = (
            await sb.table("user_usage").select(_usage_columns())
            .eq("user_id", user_id).limit(1).execute()
        )
    return res.data[0] if res.data else None


def _strip_extra(write: dict | None) -> dict | None:
    """Drop extra_used from a write payload when the column doesn't exist yet —
    otherwise the write would fail the same way the read did."""
    if write is None or _extra_columns_present:
        return write
    return {k: v for k, v in write.items() if k != "extra_used"}


def _charge_context(user_id: str, mode: str) -> dict:
    """Per-call constants shared by the sync and async charge paths."""
    today = datetime.now(timezone.utc).date()
    resets_day_at, resets_month_at = _reset_times(today)
    day_limit, month_limit = _limits(user_id)
    return {
        "cost": MODE_CREDIT_COST.get(mode, _DEFAULT_CREDIT_COST),
        "day_limit": day_limit, "month_limit": month_limit,
        "today_iso": today.isoformat(), "month": today.strftime("%Y-%m"),
        "resets_day_at": resets_day_at, "resets_month_at": resets_month_at,
    }


def _charge_decision(user_id: str, row: dict | None, ctx: dict) -> tuple[dict | None, dict]:
    """Pure decision: given the current row, return (write_payload | None, result).

    Shared by charge_credits and charge_credits_async so the limit/rollover
    logic lives in exactly one place; only the I/O around it differs.

    Redeemed extra credits are spent first: `from_extra` comes off that
    balance and is invisible to both caps, and only `remainder` is billed to
    the day/month counters — which is what makes a leftover balance smaller
    than one turn's cost still usable instead of stranded.
    """
    day_used, month_used = _rolled_over_usage(row, ctx["today_iso"], ctx["month"])
    extra_granted, extra_used = _extra_balance(row)
    extra_remaining = max(extra_granted - extra_used, 0.0)
    if extra_remaining < _EPSILON:
        extra_remaining = 0.0  # sub-cent dust is spent, not a fractional charge

    from_extra = min(ctx["cost"], extra_remaining)
    remainder = _q2(ctx["cost"] - from_extra)
    new_day = _q2(day_used + remainder)
    new_month = _q2(month_used + remainder)
    new_extra_used = _q2(extra_used + from_extra)

    common = {
        "day_limit": ctx["day_limit"], "month_limit": ctx["month_limit"],
        "resets_day_at": ctx["resets_day_at"], "resets_month_at": ctx["resets_month_at"],
        "extra_granted": extra_granted,
    }
    if new_day <= ctx["day_limit"] and new_month <= ctx["month_limit"]:
        # The full row is written even when extra paid for everything: `day`
        # and `month` still have to be re-stamped so a rolled-over counter is
        # persisted as reset rather than left stale for the next read.
        write = {
            "user_id": user_id,
            "day": ctx["today_iso"], "day_used": new_day,
            "month": ctx["month"], "month_used": new_month,
            "extra_used": new_extra_used,
            "updated_at": utcnow_iso(),
        }
        return write, {"charged": True, "day_used": new_day, "month_used": new_month,
                       "extra_used": new_extra_used,
                       "extra_remaining": _q2(extra_granted - new_extra_used),
                       "exceeded_scope": None, **common}
    day_over = new_day > ctx["day_limit"]
    month_over = new_month > ctx["month_limit"]
    scope = "both" if (day_over and month_over) else ("day" if day_over else "month")
    return None, {"charged": False, "day_used": day_used, "month_used": month_used,
                  "extra_used": extra_used, "extra_remaining": extra_remaining,
                  "exceeded_scope": scope, **common}


def _charge_failopen(ctx: dict) -> dict:
    """Fail-open result on a DB error — a usage-ledger hiccup should degrade
    gracefully (let the turn through), not take chat down with it."""
    return {
        "charged": True, "day_used": 0.0, "month_used": 0.0,
        "extra_granted": 0.0, "extra_used": 0.0, "extra_remaining": 0.0,
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
        write, result = _charge_decision(user_id, _read_usage_row(user_id), ctx)
        write = _strip_extra(write)
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
        write, result = _charge_decision(user_id, await _read_usage_row_async(user_id), ctx)
        return result, _strip_extra(write)
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


def _snapshot_get(user_id: str, today_iso: str, month: str) -> dict | None:
    """The cached row for this user, shaped like a DB row so it can be handed
    straight to `_charge_decision`, or None on a miss/expiry."""
    ent = _usage_snapshot.get(user_id)
    if not ent or ent["expiry"] <= time.monotonic():
        return None
    return {
        "day": today_iso,
        "day_used": ent["day_used"] if ent["day"] == today_iso else 0.0,
        "month": month,
        "month_used": ent["month_used"] if ent["month"] == month else 0.0,
        # Not rollover-gated, same as the real row: an extra balance survives
        # the day/month flip untouched.
        "extra_granted": ent["extra_granted"],
        "extra_used": ent["extra_used"],
    }


def _snapshot_put(user_id: str, today_iso: str, month: str, result: dict) -> None:
    """Cache a charge result as this user's next gate input."""
    with _usage_lock:
        _usage_snapshot[user_id] = {
            "day": today_iso, "day_used": result["day_used"],
            "month": month, "month_used": result["month_used"],
            "extra_granted": result.get("extra_granted", 0.0),
            "extra_used": result.get("extra_used", 0.0),
            "expiry": time.monotonic() + _USAGE_SNAPSHOT_TTL,
        }


def invalidate_usage_snapshot(user_id: str) -> None:
    """Drop this user's cached gate input so the next charge re-reads the DB.

    Redeeming a code MUST call this. Without it the snapshot keeps answering
    with the pre-redemption balance for up to `_USAGE_SNAPSHOT_TTL` seconds —
    so a user who just redeemed watches the UI show a full extra bar while
    /chat keeps 429-ing them, with no way to tell that it will fix itself.

    Only clears this process's copy. Under multiple instances the others keep
    their own stale entries until TTL, which is the same bounded staleness the
    snapshot already accepts everywhere else.
    """
    with _usage_lock:
        _usage_snapshot.pop(user_id, None)


async def evaluate_charge_fast(user_id: str, mode: str) -> dict:
    """Decide the limit gate against the local usage snapshot — no DB round trip
    on a hit. On a miss (cold process / expired), do one real read to seed the
    snapshot. Returns the usual charge result dict. Pair with commit_charge_fast,
    which the caller invokes only when it actually proceeds (so a reconnect,
    which never calls it, never charges)."""
    ctx = _charge_context(user_id, mode)
    row = _snapshot_get(user_id, ctx["today_iso"], ctx["month"])
    if row is not None:
        _, result = _charge_decision(user_id, row, ctx)
        # Optimistic local bump so a rapid burst sees rising usage; the
        # background reconcile re-anchors to DB truth right after.
        if result["charged"]:
            _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
        return result
    result, _write = await evaluate_charge_async(user_id, mode)
    _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
    return result


def usage_snapshot_hit(user_id: str) -> bool:
    """Whether a live usage-snapshot entry exists (for timing/observability logs)."""
    today = datetime.now(timezone.utc)
    return _snapshot_get(user_id, today.date().isoformat(), today.strftime("%Y-%m")) is not None


def commit_charge_fast(user_id: str, mode: str) -> None:
    """Fire-and-forget reconcile: do the real read+increment+write against the
    DB, then re-anchor the local snapshot to the post-write truth."""
    async def _run():
        try:
            ctx = _charge_context(user_id, mode)
            result, write = await evaluate_charge_async(user_id, mode)
            await commit_charge_async(write)
            _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
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
    extra_granted = extra_used = 0.0
    try:
        row = _read_usage_row(user_id)
        day_used, month_used = _rolled_over_usage(row, today_iso, month)
        extra_granted, extra_used = _extra_balance(row)
    except Exception as e:
        logger.error(f"[db_user_usage] get_usage error for {user_id}: {e}")

    return {
        "is_guest": user_id.startswith("guest_"),
        "day_used": day_used, "day_limit": day_limit,
        "day_remaining": max(day_limit - day_used, 0),
        "month_used": month_used, "month_limit": month_limit,
        "month_remaining": max(month_limit - month_used, 0),
        # Redeemed-code balance. `extra_granted` is 0 for the vast majority of
        # users; the frontend hides the whole meter in that case.
        "extra_granted": extra_granted, "extra_used": extra_used,
        "extra_remaining": max(_q2(extra_granted - extra_used), 0.0),
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
