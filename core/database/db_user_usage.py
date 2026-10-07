"""
Database operations for the user_usage table — a unified credit ledger for
both guests and signed-in users.

Table schema: see schema.sql.

Model:
    - Every /chat or /rewind call spends credits, priced by the model the user
      picked: 1 for `best` and the fine-tune, 3 for a frontier model asked for
      by name (MODE_CREDIT_COST below). A scheduled research run (core/routers/scheduled_tasks.py) spends
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

Atomicity: a charge is one transaction that locks the user's row
(SELECT ... FOR UPDATE), decides, and writes. Two concurrent charges for the same
user therefore serialise instead of overwriting each other.
"""

import asyncio
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

from core.database import pg

logger = logging.getLogger(__name__)

USER_DAILY_CREDIT_LIMIT: int = int(os.getenv("USER_DAILY_CREDIT_LIMIT", "100"))
USER_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("USER_MONTHLY_CREDIT_LIMIT", "3000"))
GUEST_DAILY_CREDIT_LIMIT: int = int(os.getenv("GUEST_DAILY_CREDIT_LIMIT", "20"))
GUEST_MONTHLY_CREDIT_LIMIT: int = int(os.getenv("GUEST_MONTHLY_CREDIT_LIMIT", "300"))

# Cost of one charge, keyed by whatever the caller passes as `mode`.
#
# Interactive turns are keyed by **model id** now (see core/chat_models.py) —
# `rix` and `best` are 1 credit and every other model is 3. `best-vision` is
# what the chat router passes when this turn carries an image; it is priced the
# same as `best` while `best` is luna on both paths, and stays a key of its own
# so image turns remain countable in the usage rows and so there is one line to
# put back to 3.0 when the fine-tune takes the text path again. The old
# `fast`/`pro` keys are kept because a client on a stale bundle, or a rewind of
# a thread created before this change, still sends them; both bill as `best`
# did.
#
# `scheduled` is unrelated to the picker and unchanged: an unattended research
# run is a much bigger job than one chat turn.
#
# `voice`/`voice-text` are core/voice/session.py's live call and
# core/routers/voice.py's typed-continuation endpoint — priced apart from
# everything above since neither goes through model selection at all (both
# always run the same fixed voice agent). A spoken turn costs more than a
# typed one in the same thread because it also pays for STT (OpenAI realtime)
# and TTS (Fish Audio) on top of the LLM call a typed turn alone makes.
MODE_CREDIT_COST: dict[str, float] = {
    "best": 1.0,
    "best-vision": 1.0,
    "rix": 1.0,
    "luna": 3.0,
    "gemini": 3.0,
    "fast": 1.0,
    "pro": 1.0,
    "scheduled": 4.7,
    "voice": 5.0,
    "voice-text": 1.0,
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
# Reading and writing the row
# ---------------------------------------------------------------------------
_USAGE_COLUMNS = "day, day_used, month, month_used, extra_granted, extra_used"
_ENSURE_ROW = "INSERT INTO user_usage (user_id) VALUES (%s) ON CONFLICT (user_id) DO NOTHING"
_LOCK_ROW = f"SELECT {_USAGE_COLUMNS} FROM user_usage WHERE user_id = %s FOR UPDATE"
_WRITE_ROW = (
    "UPDATE user_usage SET day = %(day)s, day_used = %(day_used)s, month = %(month)s, "
    "month_used = %(month_used)s, extra_used = %(extra_used)s, updated_at = now() "
    "WHERE user_id = %(user_id)s"
)


def _read_usage_row(user_id: str) -> dict | None:
    return pg.fetch_one(f"SELECT {_USAGE_COLUMNS} FROM user_usage WHERE user_id = %s", (user_id,))


async def _read_usage_row_async(user_id: str) -> dict | None:
    return await pg.afetch_one(f"SELECT {_USAGE_COLUMNS} FROM user_usage WHERE user_id = %s", (user_id,))


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

    One transaction: make sure the row exists, lock it, decide, write. Sync
    variant — use `charge_credits_async` from the async chat path.
    """
    ctx = _charge_context(user_id, mode)
    try:
        with pg.pool().connection() as conn, conn.transaction():
            conn.execute(_ENSURE_ROW, (user_id,))
            row = conn.execute(_LOCK_ROW, (user_id,)).fetchone()
            write, result = _charge_decision(user_id, row, ctx)
            if write is not None:
                conn.execute(_WRITE_ROW, write)
        return result
    except Exception as e:
        logger.error(f"[db_user_usage] charge_credits error for {user_id}: {e}")
        return _charge_failopen(ctx)


async def charge_credits_async(user_id: str, mode: str) -> dict:
    """Async twin of `charge_credits`: same single locking transaction, without
    blocking the event loop."""
    ctx = _charge_context(user_id, mode)
    try:
        async with (await pg.apool()).connection() as conn, conn.transaction():
            await conn.execute(_ENSURE_ROW, (user_id,))
            cur = await conn.execute(_LOCK_ROW, (user_id,))
            write, result = _charge_decision(user_id, await cur.fetchone(), ctx)
            if write is not None:
                await conn.execute(_WRITE_ROW, write)
        return result
    except Exception as e:
        logger.error(f"[db_user_usage] charge_credits_async error for {user_id}: {e}")
        return _charge_failopen(ctx)


async def evaluate_charge_async(user_id: str, mode: str) -> dict:
    """Read usage and decide, WITHOUT writing anything. Used to gate a request on
    a snapshot miss; the actual charge is `charge_credits_async`."""
    ctx = _charge_context(user_id, mode)
    try:
        _, result = _charge_decision(user_id, await _read_usage_row_async(user_id), ctx)
        return result
    except Exception as e:
        logger.error(f"[db_user_usage] evaluate_charge_async error for {user_id}: {e}")
        return _charge_failopen(ctx)


# ---------------------------------------------------------------------------
# Latency-first charge path: gate on a local usage snapshot, reconcile in the
# background. Limits become approximate — a burst within the snapshot window can
# slip over — which is an accepted trade for keeping the database off the /chat
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
        # background charge re-anchors to DB truth right after.
        if result["charged"]:
            _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
        return result
    result = await evaluate_charge_async(user_id, mode)
    _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
    return result


def usage_snapshot_hit(user_id: str) -> bool:
    """Whether a live usage-snapshot entry exists (for timing/observability logs)."""
    today = datetime.now(timezone.utc)
    return _snapshot_get(user_id, today.date().isoformat(), today.strftime("%Y-%m")) is not None


def commit_charge_fast(user_id: str, mode: str) -> None:
    """Fire-and-forget: do the real, atomic charge against the DB, then re-anchor
    the local snapshot to the post-write truth."""
    async def _run():
        try:
            ctx = _charge_context(user_id, mode)
            result = await charge_credits_async(user_id, mode)
            _snapshot_put(user_id, ctx["today_iso"], ctx["month"], result)
        except Exception as e:
            logger.error(f"[db_user_usage] commit_charge_fast error for {user_id}: {e}")

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
        return pg.execute("DELETE FROM user_usage WHERE user_id = %s", (user_id,)) > 0
    except Exception as e:
        logger.error(f"[db_user_usage] delete_user_usage error: {e}")
        return False
