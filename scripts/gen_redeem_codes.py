"""
Mint prepaid credit codes into the `redeem_codes` table.

There's no admin UI, so this is how codes come into existence. Prints the
generated codes to stdout as CSV (code,credits) so a batch can be pasted
straight into whatever is handing them out.

    python3 scripts/gen_redeem_codes.py --count 20
    python3 scripts/gen_redeem_codes.py --count 1 --credits 5000 --note "launch giveaway"
    python3 scripts/gen_redeem_codes.py --count 1 --max-uses 500 --expires-in-days 30 \
        --note "conference booth"
    # An evergreen code with a name people can remember, claimable once per
    # user by any number of users:
    python3 scripts/gen_redeem_codes.py --code OMNIKNOWSXYZ --max-uses 0 --note "default code"

Requires the same SUPABASE_URL / SUPABASE_KEY the backend uses (loaded from
.env, as main.py does).
"""

import argparse
import secrets
import sys
from datetime import datetime, timedelta, timezone

import dotenv

dotenv.load_dotenv()

from core.database.supabase_client import supabase  # noqa: E402  (after load_dotenv)
# Same normalization the redeem endpoint applies, imported rather than
# reimplemented — a code minted under different rules than it's looked up
# under would simply never be redeemable.
from core.database.db_redeem_codes import normalize_code  # noqa: E402

# No I/O/0/1 — the alphabet is deliberately unambiguous, because these get read
# off a screen, written down, and typed back in by hand.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_PREFIX = "OMNI"


def generate_code(groups: int = 3, group_len: int = 4) -> str:
    """e.g. OMNI-K7QX-2M9P-TRWD. Stored normalized (no dashes) — the dashes are
    purely for readability; `normalize_code` strips them on redeem."""
    body = ["".join(secrets.choice(_ALPHABET) for _ in range(group_len)) for _ in range(groups)]
    return "-".join([_PREFIX, *body])


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate and insert redeem codes.")
    ap.add_argument("--count", type=int, default=10, help="how many codes to mint")
    ap.add_argument("--code", default=None,
                    help="mint this exact code instead of a random one (implies --count 1) — "
                         "for a memorable, publishable code like a default/promo one")
    ap.add_argument("--credits", type=float, default=1000, help="credits each code grants")
    ap.add_argument("--max-uses", type=int, default=1,
                    help="redemptions allowed per code (1 = single-use; >1 = shared campaign "
                         "code; 0 or less = unlimited users, still one redemption each)")
    ap.add_argument("--expires-in-days", type=int, default=None, help="omit for no expiry")
    ap.add_argument("--note", default=None, help="free-form label, e.g. the campaign name")
    ap.add_argument("--dry-run", action="store_true", help="print codes without inserting")
    args = ap.parse_args()

    expires_at = None
    if args.expires_in_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=args.expires_in_days)).isoformat()

    # An explicit --code is stored exactly as the user will type it (after the
    # same normalization redeem_code applies), so it stays memorable.
    if args.code:
        displays = [normalize_code(args.code)]
        if not displays[0]:
            print("--code normalized to an empty string", file=sys.stderr)
            return 1
    else:
        displays = [generate_code() for _ in range(args.count)]

    rows = [{
        "code": normalize_code(display),
        "credits": args.credits,
        "max_uses": args.max_uses,
        "expires_at": expires_at,
        "note": args.note,
    } for display in displays]
    printable = displays

    if not args.dry_run:
        try:
            supabase.table("redeem_codes").insert(rows).execute()
        except Exception as e:
            print(f"insert failed: {e}", file=sys.stderr)
            return 1

    print("code,credits")
    for display in printable:
        print(f"{display},{args.credits:g}")
    if args.dry_run:
        print("(dry run — nothing was inserted)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
