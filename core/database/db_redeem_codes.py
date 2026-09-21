"""
Database operations for prepaid credit codes — `redeem_codes` (the codes) and
`code_redemptions` (who redeemed what).

Redeeming grants a permanent extra-credit balance on the user's `user_usage`
row (`extra_granted`). That bucket is spent before the daily/monthly
allowances and never counts against either cap — see db_user_usage.py for the
charge-side half of this.

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE IF NOT EXISTS redeem_codes (
        code VARCHAR(64) PRIMARY KEY,
        credits NUMERIC(10,2) NOT NULL DEFAULT 1000,
        max_uses INT NOT NULL DEFAULT 1,   -- <=0 = unlimited, see redeem_code()
        used_count INT NOT NULL DEFAULT 0,
        expires_at TIMESTAMPTZ,
        active BOOLEAN NOT NULL DEFAULT TRUE,
        note TEXT,
        restricted_user_id VARCHAR(255),   -- NULL for every ordinary code; set only
                                            -- by create_restricted_code, see below
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS code_redemptions (
        id BIGSERIAL PRIMARY KEY,
        code VARCHAR(64) NOT NULL REFERENCES redeem_codes(code),
        user_id VARCHAR(255) NOT NULL,
        credits NUMERIC(10,2) NOT NULL,
        redeemed_at TIMESTAMPTZ DEFAULT NOW(),
        CONSTRAINT uq_code_redemptions_code_user UNIQUE (code, user_id)
    );

On atomicity: PostgREST gives no transactions, so `redeem_code` below leans on
the UNIQUE (code, user_id) index as its real guard — the redemption row is
inserted BEFORE any credit is granted, and losing that insert is what "already
redeemed" means. The `used_count` bump afterwards is best-effort, so a burst of
concurrent redemptions of one shared campaign code can overshoot `max_uses` by
the number of racers. That matches the accuracy the rest of the credit system
already accepts (see the read-modify-write note in db_user_usage.charge_credits);
make it exact with a Postgres RPC doing a guarded single-statement UPDATE if a
campaign ever needs a hard cap.
"""

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from core.database.supabase_client import supabase, utcnow_iso
from core.database.db_user_usage import invalidate_usage_snapshot

logger = logging.getLogger(__name__)

# Codes are stored normalized, so "omni-1000-abcd", "OMNI 1000 ABCD" and
# "OMNI1000ABCD" are all the same primary-key lookup.
_NORMALIZE_STRIP = re.compile(r"[\s\-_]+")
MAX_CODE_LENGTH = 64


def normalize_code(code: str) -> str:
    return _NORMALIZE_STRIP.sub("", (code or "")).upper()[:MAX_CODE_LENGTH]


# No I/O/0/1 — the alphabet is deliberately unambiguous, because these get read
# off a screen, written down, and typed back in by hand. Shared by
# scripts/gen_redeem_codes.py (hand-minted campaign codes) and
# create_restricted_code below (the "get free credit" flow) so both draw codes
# from the same format.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PREFIX = "OMNI"


def generate_code(groups: int = 3, group_len: int = 4) -> str:
    """e.g. OMNI-K7QX-2M9P-TRWD. Stored normalized (no dashes) — the dashes are
    purely for readability; normalize_code strips them on redeem."""
    body = ["".join(secrets.choice(_ALPHABET) for _ in range(group_len)) for _ in range(groups)]
    return "-".join([_PREFIX, *body])


def _is_duplicate_error(e: Exception) -> bool:
    """Whether an insert failed on a UNIQUE violation rather than anything else.

    supabase-py surfaces PostgREST errors as APIError with the Postgres SQLSTATE
    in `.code`; the string check is a fallback for versions/transports that only
    carry the message.
    """
    code = getattr(e, "code", None)
    if code == "23505":
        return True
    msg = str(e).lower()
    return "23505" in msg or "duplicate key" in msg


def _expired(expires_at: str | None) -> bool:
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at.replace("Z", "+00:00")) <= datetime.now(timezone.utc)
    except ValueError:
        logger.warning(f"[db_redeem_codes] unparseable expires_at: {expires_at!r}")
        return False


def redeem_code(user_id: str, code: str) -> dict:
    """
    Redeem `code` for `user_id`, granting its credits as extra balance.

    Returns ``{"status": "ok", "credits_added": float, "extra_granted": float,
    "extra_remaining": float}`` on success, or ``{"status": <reason>}`` where
    reason is one of: invalid_code, code_expired, code_exhausted,
    already_redeemed, error.

    Order matters and is the whole correctness story: validate → insert the
    redemption row (the double-redeem guard) → grant → bump used_count →
    invalidate the charge snapshot. A crash between the insert and the grant
    leaves a redemption on record with no credits attached, which is the safe
    direction to fail (nobody gets free credits twice); `repair_grant` covers
    fixing one up by hand if it ever happens.
    """
    code = normalize_code(code)
    if not code:
        return {"status": "invalid_code"}

    try:
        res = (
            supabase.table("redeem_codes")
            .select("code, credits, max_uses, used_count, expires_at, active, restricted_user_id")
            .eq("code", code)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.error(f"[db_redeem_codes] lookup error for {code}: {e}")
        return {"status": "error"}

    row = res.data[0] if res.data else None
    # An inactive code, or a code minted for a different user (see
    # create_restricted_code / the "get free credit" flow), is reported as
    # invalid rather than as its own state: whether a code exists but doesn't
    # belong to you isn't something a stranger guessing codes should be able
    # to learn.
    if not row or not row.get("active"):
        return {"status": "invalid_code"}
    restricted_to = row.get("restricted_user_id")
    if restricted_to and restricted_to != user_id:
        return {"status": "invalid_code"}
    if _expired(row.get("expires_at")):
        return {"status": "code_expired"}
    # max_uses <= 0 means unlimited — an evergreen/default code anyone can
    # claim. Note what "unlimited" does NOT mean: the UNIQUE (code, user_id)
    # guard still applies, so it's unlimited *users*, one redemption each. A
    # code that the same person could redeem repeatedly would be an infinite
    # credit tap.
    max_uses = int(row.get("max_uses") or 0)
    if max_uses > 0 and int(row.get("used_count") or 0) >= max_uses:
        return {"status": "code_exhausted"}

    credits = float(row["credits"])

    # Read the balance BEFORE recording the redemption. If the ledger read is
    # going to fail — most likely because the extra_granted/extra_used ALTERs
    # haven't been run yet — it's much better to fail here, with nothing
    # written, than to leave a redemption row behind that makes every retry
    # answer already_redeemed for a code that never granted anything.
    current = _read_extra(user_id)
    if current is None:
        logger.error(f"[db_redeem_codes] ledger unreadable, refusing to redeem {code} for {user_id}")
        return {"status": "error"}

    try:
        supabase.table("code_redemptions").insert({
            "code": code,
            "user_id": user_id,
            "credits": credits,
            "redeemed_at": utcnow_iso(),
        }).execute()
    except Exception as e:
        if _is_duplicate_error(e):
            return {"status": "already_redeemed"}
        logger.error(f"[db_redeem_codes] redemption insert error for {user_id}/{code}: {e}")
        return {"status": "error"}

    granted = _grant_extra_credits(user_id, credits, current)
    if granted is None:
        # The redemption is on record but the grant failed — surfacing an error
        # is honest, and a retry will (correctly) say already_redeemed rather
        # than granting twice. See repair_grant.
        logger.error(f"[db_redeem_codes] grant failed after redemption row for {user_id}/{code}")
        return {"status": "error"}

    try:
        supabase.table("redeem_codes").update(
            {"used_count": int(row.get("used_count") or 0) + 1}
        ).eq("code", code).execute()
    except Exception as e:
        # Best-effort: the user already has their credits, and used_count is
        # only a campaign counter. Log and move on rather than failing a
        # redemption that actually succeeded.
        logger.warning(f"[db_redeem_codes] used_count bump failed for {code}: {e}")

    # Must come last, and must not be skipped: the charge path gates on a 60s
    # in-process snapshot that still holds the pre-redemption balance.
    invalidate_usage_snapshot(user_id)

    extra_granted, extra_used = granted
    return {
        "status": "ok",
        "credits_added": credits,
        "extra_granted": extra_granted,
        "extra_remaining": max(round(extra_granted - extra_used, 2), 0.0),
    }


def _read_extra(user_id: str) -> tuple[float, float] | None:
    """This user's (extra_granted, extra_used), (0, 0) if they have no ledger
    row yet, or None if the read itself failed."""
    try:
        res = (
            supabase.table("user_usage")
            .select("extra_granted, extra_used")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.error(f"[db_redeem_codes] _read_extra error for {user_id}: {e}")
        return None
    row = res.data[0] if res.data else None
    if not row:
        return 0.0, 0.0
    return float(row.get("extra_granted") or 0), float(row.get("extra_used") or 0)


def _grant_extra_credits(
    user_id: str, credits: float, current: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    """Add `credits` to this user's extra balance. Returns the post-write
    (extra_granted, extra_used), or None if the write failed. `current` is the
    already-read balance, so the caller can validate the ledger before it
    commits to anything.

    Read-modify-write, like every other update in this module's neighbourhood —
    a user redeeming two codes in the same instant could lose one grant, which
    is vanishingly unlikely (it needs two codes typed into two tabs within the
    same round trip) and recoverable from `code_redemptions`, which holds the
    authoritative list of what was actually redeemed.
    """
    if current is None:
        current = _read_extra(user_id)
        if current is None:
            return None
    extra_granted, extra_used = current
    new_granted = round(extra_granted + credits, 2)
    try:
        # Partial payload: PostgREST's upsert only writes the columns present,
        # so an existing row keeps its day/month counters untouched and a new
        # row picks up the schema defaults for them.
        supabase.table("user_usage").upsert(
            {"user_id": user_id, "extra_granted": new_granted, "updated_at": utcnow_iso()},
            on_conflict="user_id",
        ).execute()
        return new_granted, extra_used
    except Exception as e:
        logger.error(f"[db_redeem_codes] _grant_extra_credits error for {user_id}: {e}")
        return None


def repair_grant(user_id: str, code: str) -> dict:
    """Re-apply the grant for a redemption that was recorded but never credited.

    Manual recovery tool for the crash window in `redeem_code`, not called by
    any endpoint: verify against `code_redemptions` first that the user really
    is missing the credit, because this grants unconditionally.
    """
    code = normalize_code(code)
    try:
        res = (
            supabase.table("code_redemptions")
            .select("credits")
            .eq("code", code)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.error(f"[db_redeem_codes] repair_grant lookup error: {e}")
        return {"status": "error"}
    if not res.data:
        return {"status": "not_redeemed"}
    granted = _grant_extra_credits(user_id, float(res.data[0]["credits"]))
    if granted is None:
        return {"status": "error"}
    invalidate_usage_snapshot(user_id)
    return {"status": "ok", "extra_granted": granted[0]}


def create_restricted_code(user_id: str, credits: float, expires_in_days: int) -> tuple[str, str] | None:
    """Mint a fresh code that only `user_id` can redeem (single use, expires in
    `expires_in_days`) — the code-minting half of the "get free credit" flow;
    see core/database/db_free_credit.py for the risk-check + cooldown half that
    decides whether to call this at all.

    Returns (code, expires_at) on success, None if the insert failed. Retries
    once on a PK collision (astronomically unlikely at this alphabet/length,
    same odds scripts/gen_redeem_codes.py already accepts) before giving up —
    a second collision is treated as a real error rather than looped on.
    """
    expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()
    for _ in range(2):
        code = generate_code()
        try:
            supabase.table("redeem_codes").insert({
                "code": code,
                "credits": credits,
                "max_uses": 1,
                "expires_at": expires_at,
                "note": "free_credit self-serve grant",
                "restricted_user_id": user_id,
            }).execute()
            return code, expires_at
        except Exception as e:
            if _is_duplicate_error(e):
                logger.warning(f"[db_redeem_codes] code collision minting for {user_id}, retrying: {code}")
                continue
            logger.error(f"[db_redeem_codes] create_restricted_code insert error for {user_id}: {e}")
            return None
    logger.error(f"[db_redeem_codes] create_restricted_code gave up after repeated collisions for {user_id}")
    return None
