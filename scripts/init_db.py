"""Create or update the database schema (schema.sql + evals/schema_evals.sql).

Both files are idempotent, so this is safe to run on an empty database, on an
existing one, and on every deploy.

  python -m scripts.init_db            # uses DATABASE_URL
"""
import pathlib
import sys

import psycopg

from core.database import pg

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = ["schema.sql", "evals/schema_evals.sql"]


def main() -> None:
    url = pg._url()
    with psycopg.connect(url, autocommit=True) as conn:
        for name in FILES:
            # One transaction per file: a half-applied schema is worse than none.
            with conn.transaction():
                conn.execute((ROOT / name).read_text())
            print(f"applied {name}")
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        ]
    print(f"{len(tables)} tables/views: {', '.join(tables)}")


if __name__ == "__main__":
    sys.exit(main())
