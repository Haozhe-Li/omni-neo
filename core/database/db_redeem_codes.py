"""
Database operations for prepaid credit codes — `redeem_codes` (the codes) and
`code_redemptions` (who redeemed what).

Redeeming grants a permanent extra-credit balance on the user's `user_usage`
row (`extra_granted`). That bucket is spent before the daily/monthly
allowances and never counts against either cap — see db_user_usage.py for the
charge-side half of this.

Table schema: see schema.sql.

Atomicity: a redemption is ONE transaction. It locks the code's row
(SELECT ... FOR UPDATE), checks it, records the redemption, bumps used_count and
adds the credits to the user's balance — all or nothing. So a shared campaign
code can never exceed `max_uses`, a crash can't leave a redemption recorded
without its credits, and the UNIQUE (code, user_id) constraint guarantees one
redemption per user however many times they try.
"""

import logging
import re
import secrets
from datetime import datetime, timedelta, timezone

from core.database import pg
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


def redeem_code(user_id: str, code: str) -> dict:
    """
    Redeem `code` for `user_id`, granting its credits as extra balance.

    Returns ``{"status": "ok", "credits_added": float, "extra_granted": float,
    "extra_remaining": float}`` on success, or ``{"status": <reason>}`` where
    reason is one of: invalid_code, code_expired, code_exhausted,
    already_redeemed, error.
    """
    code = normalize_code(code)
    if not code:
        return {"status": "invalid_code"}

    try:
        with pg.pool().connection() as conn, conn.transaction():
            row = conn.execute(
                "SELECT credits, max_uses, used_count, active, restricted_user_id, "
                "       (expires_at IS NOT NULL AND expires_at <= now()) AS expired "
                "FROM redeem_codes WHERE code = %s FOR UPDATE",
                (code,),
            ).fetchone()
            # An inactive code, or a code minted for a different user (see
            # create_restricted_code / the "get free credit" flow), is reported
            # as invalid rather than as its own state: whether a code exists but
            # doesn't belong to you isn't something a stranger guessing codes
            # should be able to learn.
            if not row or not row["active"]:
                return {"status": "invalid_code"}
            if row["restricted_user_id"] and row["restricted_user_id"] != user_id:
                return {"status": "invalid_code"}
            if row["expired"]:
                return {"status": "code_expired"}
            # Checked before exhaustion so that someone retrying a code they already
            # used hears "already redeemed", not "code exhausted".
            if conn.execute(
                "SELECT 1 FROM code_redemptions WHERE code = %s AND user_id = %s", (code, user_id)
            ).fetchone():
                return {"status": "already_redeemed"}
            # max_uses <= 0 means unlimited — an evergreen/default code anyone can
            # claim. It is unlimited *users*, one redemption each: the UNIQUE
            # (code, user_id) guard below still applies.
            if row["max_uses"] > 0 and row["used_count"] >= row["max_uses"]:
                return {"status": "code_exhausted"}

            credits = row["credits"]
            won = conn.execute(
                "INSERT INTO code_redemptions (code, user_id, credits) VALUES (%s, %s, %s) "
                "ON CONFLICT (code, user_id) DO NOTHING RETURNING id",
                (code, user_id, credits),
            ).fetchone()
            if won is None:
                return {"status": "already_redeemed"}

            conn.execute("UPDATE redeem_codes SET used_count = used_count + 1 WHERE code = %s", (code,))
            ledger = conn.execute(
                "INSERT INTO user_usage (user_id, extra_granted) VALUES (%s, %s) "
                "ON CONFLICT (user_id) DO UPDATE "
                "   SET extra_granted = user_usage.extra_granted + EXCLUDED.extra_granted, "
                "       updated_at = now() "
                "RETURNING extra_granted, extra_used",
                (user_id, credits),
            ).fetchone()
    except Exception as e:
        logger.error(f"[db_redeem_codes] redeem error for {user_id}/{code}: {e}")
        return {"status": "error"}

    # Must come after the commit, and must not be skipped: the charge path gates
    # on a 60s in-process snapshot that still holds the pre-redemption balance.
    invalidate_usage_snapshot(user_id)

    return {
        "status": "ok",
        "credits_added": credits,
        "extra_granted": ledger["extra_granted"],
        "extra_remaining": max(round(ledger["extra_granted"] - ledger["extra_used"], 2), 0.0),
    }


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
    expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
    for _ in range(2):
        code = generate_code()
        try:
            inserted = pg.fetch_one(
                "INSERT INTO redeem_codes (code, credits, max_uses, expires_at, note, restricted_user_id) "
                "VALUES (%s, %s, 1, %s, 'free_credit self-serve grant', %s) "
                "ON CONFLICT (code) DO NOTHING RETURNING code",
                # Stored normalized, like every other code: redeem_code looks the
                # user's input up after normalize_code, so a row keyed by the
                # dashed display form could never be found.
                (normalize_code(code), credits, expires_at, user_id),
            )
        except Exception as e:
            logger.error(f"[db_redeem_codes] create_restricted_code insert error for {user_id}: {e}")
            return None
        if inserted:
            return code, expires_at.isoformat()  # `code` is the readable, dashed form
        logger.warning(f"[db_redeem_codes] code collision minting for {user_id}, retrying: {code}")
    logger.error(f"[db_redeem_codes] create_restricted_code gave up after repeated collisions for {user_id}")
    return None
