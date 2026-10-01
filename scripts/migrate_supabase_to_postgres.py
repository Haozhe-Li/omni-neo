"""One-off: copy every table from the old Supabase project into the new Postgres.

Reads Supabase over its REST API (read-only; SUPABASE_URL + SUPABASE_KEY, the
service_role key) and upserts into DATABASE_URL by primary key, so it is safe to
re-run: run it once ahead of time, then again at cutover to pick up whatever
changed in between. Nothing is ever deleted from either side.

  python -m scripts.init_db                              # schema first
  python -m scripts.migrate_supabase_to_postgres --dry-run
  python -m scripts.migrate_supabase_to_postgres
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import httpx
import psycopg

from core.database import pg

# (table, primary key columns, REST page size). Parents before children.
TABLES = [
    ("threads_control", ["thread_id"], 1000),
    ("user_threads", ["thread_id"], 100),
    ("user_memories", ["user_id"], 500),
    ("user_usage", ["user_id"], 1000),
    ("redeem_codes", ["code"], 1000),
    ("code_redemptions", ["id"], 1000),
    ("user_files", ["file_id"], 200),
    ("scheduled_tasks", ["task_id"], 500),
    ("scheduled_task_runs", ["run_id"], 50),
    ("eval_cases", ["case_id"], 200),
    ("eval_runs", ["run_id"], 200),
    ("eval_results", ["result_id"], 20),
    ("eval_checks", ["check_id"], 500),
    ("eval_pricing", ["pricing_id"], 500),
    ("eval_case_scores", ["run_id", "case_id"], 500),
    ("eval_oracle", ["family", "scope", "key"], 500),
]
SERIALS = {"code_redemptions": "id", "eval_checks": "check_id", "eval_pricing": "pricing_id"}


def normalize_code(code: str) -> str:
    """Same rule as db_redeem_codes.normalize_code (inlined: this script must not
    import the app). Codes minted by the old "get free credit" flow were stored in
    their dashed display form, which redeem_code could never look up."""
    import re

    return re.sub(r"[\s\-_]+", "", code or "").upper()[:64]


def fetch_pages(client: httpx.Client, base: str, table: str, order: list[str], page: int):
    offset = 0
    while True:
        r = client.get(
            f"{base}/rest/v1/{table}",
            params={"select": "*", "order": ",".join(order), "limit": page, "offset": offset},
        )
        if r.status_code == 404:  # table never existed in the old project
            return
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return
        yield rows
        if len(rows) < page:
            return
        offset += page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="read and check everything, write nothing")
    ap.add_argument("--tables", help="comma separated subset")
    args = ap.parse_args()

    base = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_KEY"]
    http = httpx.Client(headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=120)
    wanted = set(args.tables.split(",")) if args.tables else None

    with psycopg.connect(pg._url(), autocommit=False) as conn:
        cols = defaultdict(set)
        for t, c in conn.execute("SELECT table_name, column_name FROM information_schema.columns WHERE table_schema='public'"):
            cols[t].add(c)
        notnull_default = set()
        for t, c in conn.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND is_nullable='NO' AND column_default IS NOT NULL"
        ):
            notnull_default.add((t, c))

        problems = 0
        for table, pk, page in TABLES:
            if wanted and table not in wanted:
                continue
            n = 0
            ignored: Counter = Counter()
            for batch in fetch_pages(http, base, table, pk, page):
                groups: dict[tuple, list] = defaultdict(list)
                for row in batch:
                    clean = {}
                    for c, v in row.items():
                        if c not in cols[table]:
                            ignored[c] += 1
                            continue
                        # NULL in a NOT NULL ... DEFAULT column: let the default apply.
                        if v is None and (table, c) in notnull_default:
                            continue
                        clean[c] = v
                    if table in ("redeem_codes", "code_redemptions") and clean.get("code"):
                        clean["code"] = normalize_code(clean["code"])
                    groups[tuple(sorted(clean))].append(clean)
                n += len(batch)
                if args.dry_run:
                    continue
                for keyset, rows in groups.items():
                    stmt, params, upd = pg_upsert_sql(table, keyset, pk, rows)
                    try:
                        with conn.transaction():
                            conn.execute(stmt, params)
                    except psycopg.Error as e:
                        problems += 1
                        print(f"  !! {table}: {type(e).__name__}: {str(e).splitlines()[0]}")
            if ignored:
                print(f"  note: {table}: source columns not in the new schema were skipped: {dict(ignored)}")
            print(f"{table:<22} {n:>6} rows" + (" (dry run)" if args.dry_run else ""))

        if not args.dry_run:
            for table, col in SERIALS.items():
                conn.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', '{col}'), "
                    f"COALESCE((SELECT max({col}) FROM {table}), 0) + 1, false)"
                )
            conn.commit()
            print("\nrow counts in the new database:")
            for table, _, _ in TABLES:
                if wanted and table not in wanted:
                    continue
                print(f"  {table:<22} {conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]:>6}")
        if problems:
            print(f"\n{problems} batch(es) failed — see the lines marked !! above")
            sys.exit(1)


def pg_upsert_sql(table: str, keyset: tuple, pk: list[str], rows: list[dict]):
    from psycopg import sql

    cols = list(keyset)
    ph = sql.SQL("({})").format(sql.SQL(", ").join(sql.Placeholder() for _ in cols))
    upd = [c for c in cols if c not in pk]
    stmt = sql.SQL("INSERT INTO {} ({}) VALUES {} ON CONFLICT ({}) ").format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        sql.SQL(", ").join(ph for _ in rows),
        sql.SQL(", ").join(sql.Identifier(c) for c in pk),
    )
    stmt += (
        sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in upd)
        )
        if upd else sql.SQL("DO NOTHING")
    )
    params = [pg.adapt(r[c]) for r in rows for c in cols]
    return stmt, params, upd


if __name__ == "__main__":
    main()
