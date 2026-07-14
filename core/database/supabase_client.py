"""Shared Supabase client — all DB access goes over PostgREST (HTTP only).

Replaces the psycopg ``sync_pool`` that talked to Supabase over a direct
Postgres TCP connection. Same motivation as the checkpointer move: on Cloud
Run's scale-to-zero, idle TCP connections got silently dropped and the next
request stalled ~60s before the keepalive killed the dead socket. supabase-py
speaks HTTP via httpx, so a dropped keep-alive connection surfaces as a clean
reset and reconnects in ~100ms instead of black-holing.

Credentials come from ``SUPABASE_URL`` / ``SUPABASE_KEY``.

Note on the key: server-side code here writes/reads arbitrary user_ids (keyed
off a verified Clerk JWT, not Supabase Auth), so it needs to bypass RLS — that
requires a *service_role* key. A publishable/anon key only works if RLS is
disabled on these tables. See the migration notes for the implications.
"""
import os
from datetime import datetime, timezone

from supabase import Client, create_client

supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def utcnow_iso() -> str:
    """A literal UTC ISO-8601 timestamp for `updated_at`-style columns.

    The DB's ``NOW()`` isn't reachable over PostgREST, so writes that used to
    say ``updated_at = NOW()`` compute the timestamp here instead.
    """
    return datetime.now(timezone.utc).isoformat()
