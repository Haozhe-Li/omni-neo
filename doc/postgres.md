# PostgreSQL

The backend talks to PostgreSQL directly (psycopg 3, `core/database/pg.py`) using `DATABASE_URL`.
On Railway use the database service's private URL (`postgres.railway.internal`).

## Schema

`schema.sql` (product tables) and `evals/schema_evals.sql` (evaluation tables and views) are
idempotent. Apply both with

    python -m scripts.init_db

Run it after changing either file, and on first setup. The app itself never runs DDL.

## Moving data in from the old Supabase project (one-off)

    python -m scripts.migrate_supabase_to_postgres --dry-run
    python -m scripts.migrate_supabase_to_postgres

Needs `SUPABASE_URL`, `SUPABASE_KEY` (service_role) and `DATABASE_URL`. It upserts by primary key and
never deletes, so it can be repeated: run it once ahead of time, and again just before switching the
app over to pick up whatever changed in between. Delete the script once the old project is retired.

## Pool settings

`PG_POOL_SYNC_MAX` (default 20) and `PG_POOL_ASYNC_MAX` (default 10) cap connections per process.
Sync handlers use the first pool, `async def` handlers the second.
