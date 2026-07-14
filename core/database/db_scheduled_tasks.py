"""
Database operations for scheduled research tasks (the "run this prompt on a
schedule, email me the report" feature). Over Supabase PostgREST (HTTP).

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE IF NOT EXISTS scheduled_tasks (
        task_id TEXT PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        email TEXT NOT NULL,
        prompt TEXT NOT NULL,
        cron_schedule TEXT NOT NULL,
        qstash_schedule_id TEXT,
        status VARCHAR(20) NOT NULL DEFAULT 'active',  -- active | paused | deleted
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS scheduled_task_runs (
        run_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
        thread_id TEXT,
        publish_id TEXT,
        title TEXT,
        report_markdown TEXT,
        sources JSONB,
        summary TEXT,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | running | success | failed
        error TEXT,
        qstash_message_id TEXT,   -- idempotency guard (unique index, see schema.sql)
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

Each logged-in user may have at most MAX_ACTIVE_TASKS non-deleted tasks
(enforced in core/routers/scheduled_tasks.py at creation time, not here).
Each firing gets its own row in scheduled_task_runs plus its own thread_id.

A run's report is private by construction: it lives only in this table, gated
by scheduled_tasks.user_id (see api_get_run in core/routers/scheduled_tasks.py).
"""

import logging

from core.database.supabase_client import supabase, utcnow_iso

logger = logging.getLogger(__name__)

MAX_ACTIVE_TASKS = 3


# ---------------------------------------------------------------------------
# scheduled_tasks
# ---------------------------------------------------------------------------

def count_active_tasks(user_id: str) -> int:
    """Tasks counting against the per-user MAX_ACTIVE_TASKS cap (active + paused, not deleted)."""
    try:
        res = (
            supabase.table("scheduled_tasks")
            .select("task_id", count="exact")
            .eq("user_id", user_id)
            .in_("status", ["active", "paused"])
            .execute()
        )
        return res.count or 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] count_active_tasks error: {e}")
        return 0


def create_task(
    task_id: str,
    user_id: str,
    name: str,
    email: str,
    prompt: str,
    cron_schedule: str,
    qstash_schedule_id: str | None,
) -> bool:
    try:
        res = supabase.table("scheduled_tasks").insert({
            "task_id": task_id,
            "user_id": user_id,
            "name": name,
            "email": email,
            "prompt": prompt,
            "cron_schedule": cron_schedule,
            "qstash_schedule_id": qstash_schedule_id,
        }).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] create_task error: {e}")
        return False


def update_task(
    task_id: str,
    user_id: str,
    *,
    name: str | None = None,
    prompt: str | None = None,
    cron_schedule: str | None = None,
) -> bool:
    """Edit a task's content (not its status — see update_task_status).
    The caller is responsible for updating the QStash schedule to match."""
    payload = {
        col: val
        for col, val in (("name", name), ("prompt", prompt), ("cron_schedule", cron_schedule))
        if val is not None
    }
    if not payload:
        return False
    payload["updated_at"] = utcnow_iso()
    try:
        res = (
            supabase.table("scheduled_tasks")
            .update(payload)
            .eq("task_id", task_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task error: {e}")
        return False


def get_task(task_id: str) -> dict | None:
    try:
        res = (
            supabase.table("scheduled_tasks")
            .select("*")
            .eq("task_id", task_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] get_task error: {e}")
        return None


def list_tasks_for_user(user_id: str) -> list[dict]:
    try:
        res = (
            supabase.table("scheduled_tasks")
            .select("*")
            .eq("user_id", user_id)
            .neq("status", "deleted")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_tasks_for_user error: {e}")
        return []


def update_task_status(task_id: str, user_id: str, status: str) -> bool:
    """status: 'active' | 'paused' | 'deleted'. Scoped to user_id to enforce ownership."""
    try:
        res = (
            supabase.table("scheduled_tasks")
            .update({"status": status, "updated_at": utcnow_iso()})
            .eq("task_id", task_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task_status error: {e}")
        return False


# ---------------------------------------------------------------------------
# scheduled_task_runs
# ---------------------------------------------------------------------------

def create_run(run_id: str, task_id: str, qstash_message_id: str | None = None) -> bool:
    """Insert a new run row. Returns False (no-op) if qstash_message_id already
    exists — that's a QStash retry of a delivery we already started, not a new firing."""
    try:
        # QStash retries redeliver the same message id; skip if we've seen it.
        if qstash_message_id is not None:
            existing = (
                supabase.table("scheduled_task_runs")
                .select("run_id")
                .eq("qstash_message_id", qstash_message_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                return False
        res = supabase.table("scheduled_task_runs").insert({
            "run_id": run_id,
            "task_id": task_id,
            "status": "pending",
            "qstash_message_id": qstash_message_id,
        }).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] create_run error: {e}")
        return False


def update_run(
    run_id: str,
    *,
    status: str | None = None,
    thread_id: str | None = None,
    title: str | None = None,
    report_markdown: str | None = None,
    sources: list[dict] | None = None,
    summary: str | None = None,
    error: str | None = None,
) -> None:
    payload = {
        col: val
        for col, val in (
            ("status", status),
            ("thread_id", thread_id),
            ("title", title),
            ("report_markdown", report_markdown),
            ("summary", summary),
            ("error", error),
        )
        if val is not None
    }
    if sources is not None:
        payload["sources"] = sources
    if not payload:
        return
    payload["updated_at"] = utcnow_iso()
    try:
        supabase.table("scheduled_task_runs").update(payload).eq("run_id", run_id).execute()
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_run error: {e}")


def get_run(run_id: str) -> dict | None:
    try:
        res = (
            supabase.table("scheduled_task_runs")
            .select("*")
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] get_run error: {e}")
        return None


def list_runs_for_task(task_id: str, limit: int = 20) -> list[dict]:
    """Excludes report_markdown/sources — those are only needed by the single
    -run detail view (get_run), not the run-history list in Settings, so
    leaving them out keeps this payload small."""
    try:
        res = (
            supabase.table("scheduled_task_runs")
            .select("run_id, task_id, thread_id, summary, status, error, "
                    "qstash_message_id, created_at, updated_at")
            .eq("task_id", task_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_runs_for_task error: {e}")
        return []
