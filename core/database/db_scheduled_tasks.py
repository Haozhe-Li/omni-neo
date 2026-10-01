"""
Database operations for scheduled research tasks (the "run this prompt on a
schedule, email me the report" feature). Direct Postgres access (see pg.py);
table schema in schema.sql (scheduled_tasks, scheduled_task_runs).

Each logged-in user may have at most MAX_ACTIVE_TASKS non-deleted tasks
(enforced in core/routers/scheduled_tasks.py at creation time, not here).
Each firing gets its own row in scheduled_task_runs plus its own thread_id.

A run's report is private by construction: it lives only in this table, gated
by scheduled_tasks.user_id (see api_get_run in core/routers/scheduled_tasks.py).
"""

import logging

from core.database import pg

logger = logging.getLogger(__name__)

MAX_ACTIVE_TASKS = 3


# ---------------------------------------------------------------------------
# scheduled_tasks
# ---------------------------------------------------------------------------

def count_active_tasks(user_id: str) -> int:
    """Tasks counting against the per-user MAX_ACTIVE_TASKS cap (active + paused, not deleted)."""
    try:
        row = pg.fetch_one(
            "SELECT count(*) AS n FROM scheduled_tasks "
            "WHERE user_id = %s AND status IN ('active', 'paused')",
            (user_id,),
        )
        return int(row["n"])
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
        pg.execute(
            "INSERT INTO scheduled_tasks "
            "(task_id, user_id, name, email, prompt, cron_schedule, qstash_schedule_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (task_id, user_id, name, email, prompt, cron_schedule, qstash_schedule_id),
        )
        return True
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
    patch = {
        col: val
        for col, val in (("name", name), ("prompt", prompt), ("cron_schedule", cron_schedule))
        if val is not None
    }
    if not patch:
        return False
    try:
        # now() is set in SQL; the patch only carries the caller's columns.
        sets = ", ".join(f"{col} = %s" for col in patch)
        return pg.execute(
            f"UPDATE scheduled_tasks SET {sets}, updated_at = now() WHERE task_id = %s AND user_id = %s",
            (*patch.values(), task_id, user_id),
        ) > 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task error: {e}")
        return False


def get_task(task_id: str) -> dict | None:
    try:
        return pg.fetch_one("SELECT * FROM scheduled_tasks WHERE task_id = %s", (task_id,))
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] get_task error: {e}")
        return None


def list_tasks_for_user(user_id: str) -> list[dict]:
    try:
        return pg.fetch_all(
            "SELECT * FROM scheduled_tasks WHERE user_id = %s AND status <> 'deleted' "
            "ORDER BY created_at DESC",
            (user_id,),
        )
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_tasks_for_user error: {e}")
        return []


def update_task_status(task_id: str, user_id: str, status: str) -> bool:
    """status: 'active' | 'paused' | 'deleted'. Scoped to user_id to enforce ownership."""
    try:
        return pg.execute(
            "UPDATE scheduled_tasks SET status = %s, updated_at = now() "
            "WHERE task_id = %s AND user_id = %s",
            (status, task_id, user_id),
        ) > 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task_status error: {e}")
        return False


# ---------------------------------------------------------------------------
# scheduled_task_runs
# ---------------------------------------------------------------------------

def create_run(run_id: str, task_id: str, qstash_message_id: str | None = None) -> bool:
    """Insert a new run row. Returns False (no-op) if qstash_message_id already
    exists — that's a QStash retry of a delivery we already started, not a new
    firing. The check and the insert are one statement, guarded by the partial
    unique index on qstash_message_id."""
    try:
        row = pg.fetch_one(
            "INSERT INTO scheduled_task_runs (run_id, task_id, status, qstash_message_id) "
            "VALUES (%s, %s, 'pending', %s) "
            "ON CONFLICT (qstash_message_id) WHERE qstash_message_id IS NOT NULL DO NOTHING "
            "RETURNING run_id",
            (run_id, task_id, qstash_message_id),
        )
        return row is not None
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
    patch = {
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
        patch["sources"] = pg.adapt(sources)
    if not patch:
        return
    try:
        sets = ", ".join(f"{col} = %s" for col in patch)
        pg.execute(
            f"UPDATE scheduled_task_runs SET {sets}, updated_at = now() WHERE run_id = %s",
            (*patch.values(), run_id),
        )
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_run error: {e}")


def get_run(run_id: str) -> dict | None:
    try:
        return pg.fetch_one("SELECT * FROM scheduled_task_runs WHERE run_id = %s", (run_id,))
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] get_run error: {e}")
        return None


def list_runs_for_task(task_id: str, limit: int = 20) -> list[dict]:
    """Excludes report_markdown/sources — those are only needed by the single
    -run detail view (get_run), not the run-history list in Settings, so
    leaving them out keeps this payload small."""
    try:
        return pg.fetch_all(
            "SELECT run_id, task_id, thread_id, summary, status, error, "
            "qstash_message_id, created_at, updated_at "
            "FROM scheduled_task_runs WHERE task_id = %s ORDER BY created_at DESC LIMIT %s",
            (task_id, limit),
        )
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_runs_for_task error: {e}")
        return []
