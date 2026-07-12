"""
Database operations for scheduled research tasks (the "run this prompt on a
schedule, email me the report" feature).

Table schema:
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
        thread_id TEXT,            -- fresh chat thread created for this run
        publish_id TEXT,           -- Pages publish:{id} for the generated report
        summary TEXT,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | running | success | failed
        error TEXT,
        qstash_message_id TEXT,    -- idempotency guard: QStash retries redeliver
                                    -- the same message id, so a unique index here
                                    -- turns a retry into a no-op insert instead of
                                    -- a second run.
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

Each logged-in user may have at most MAX_ACTIVE_TASKS non-deleted tasks
(enforced in core/routers/scheduled_tasks.py at creation time, not here).
Each task fires repeatedly on its cron schedule; every firing gets its own
row in scheduled_task_runs plus its own thread_id, so the user can continue
chatting in that thread afterward exactly like any other conversation.
"""

import json
import logging

from core.database.postgresql_saver import sync_pool as pool

logger = logging.getLogger(__name__)

MAX_ACTIVE_TASKS = 3


def setup_scheduled_tasks_table() -> None:
    """Idempotently create the scheduled_tasks / scheduled_task_runs tables. Safe on every startup."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            task_id TEXT PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            email TEXT NOT NULL,
            prompt TEXT NOT NULL,
            cron_schedule TEXT NOT NULL,
            qstash_schedule_id TEXT,
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_user_id ON scheduled_tasks(user_id);",
        # Table pre-dates the `name` column (added once the frontend needed a
        # separate task name from the prompt) — idempotent for existing rows.
        "ALTER TABLE scheduled_tasks ADD COLUMN IF NOT EXISTS name TEXT NOT NULL DEFAULT '';",
        """
        CREATE TABLE IF NOT EXISTS scheduled_task_runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
            thread_id TEXT,
            publish_id TEXT,
            summary TEXT,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            error TEXT,
            qstash_message_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_task_id ON scheduled_task_runs(task_id);",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_runs_qstash_msg
            ON scheduled_task_runs(qstash_message_id) WHERE qstash_message_id IS NOT NULL;
        """,
    ]
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                for stmt in ddl:
                    cur.execute(stmt)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] setup_scheduled_tasks_table error: {e}")


# ---------------------------------------------------------------------------
# scheduled_tasks
# ---------------------------------------------------------------------------

def count_active_tasks(user_id: str) -> int:
    """Tasks counting against the per-user MAX_ACTIVE_TASKS cap (active + paused, not deleted)."""
    sql = "SELECT COUNT(*) AS cnt FROM scheduled_tasks WHERE user_id = %s AND status IN ('active', 'paused')"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
                return row["cnt"] if row else 0
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
    sql = """
        INSERT INTO scheduled_tasks (task_id, user_id, name, email, prompt, cron_schedule, qstash_schedule_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (task_id, user_id, name, email, prompt, cron_schedule, qstash_schedule_id))
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
    fields = []
    values = []
    for col, val in (("name", name), ("prompt", prompt), ("cron_schedule", cron_schedule)):
        if val is not None:
            fields.append(f"{col} = %s")
            values.append(val)
    if not fields:
        return False
    fields.append("updated_at = NOW()")
    sql = f"UPDATE scheduled_tasks SET {', '.join(fields)} WHERE task_id = %s AND user_id = %s"
    values += [task_id, user_id]
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task error: {e}")
        return False


def get_task(task_id: str) -> dict | None:
    sql = "SELECT * FROM scheduled_tasks WHERE task_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (task_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] get_task error: {e}")
        return None


def list_tasks_for_user(user_id: str) -> list[dict]:
    sql = """
        SELECT * FROM scheduled_tasks
        WHERE user_id = %s AND status != 'deleted'
        ORDER BY created_at DESC
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_tasks_for_user error: {e}")
        return []


def update_task_status(task_id: str, user_id: str, status: str) -> bool:
    """status: 'active' | 'paused' | 'deleted'. Scoped to user_id to enforce ownership."""
    sql = """
        UPDATE scheduled_tasks SET status = %s, updated_at = NOW()
        WHERE task_id = %s AND user_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, task_id, user_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_task_status error: {e}")
        return False


# ---------------------------------------------------------------------------
# scheduled_task_runs
# ---------------------------------------------------------------------------

def create_run(run_id: str, task_id: str, qstash_message_id: str | None = None) -> bool:
    """Insert a new run row. Returns False (no-op) if qstash_message_id already
    exists — that's a QStash retry of a delivery we already started, not a new firing."""
    sql = """
        INSERT INTO scheduled_task_runs (run_id, task_id, status, qstash_message_id)
        VALUES (%s, %s, 'pending', %s)
        ON CONFLICT (qstash_message_id) WHERE qstash_message_id IS NOT NULL DO NOTHING
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_id, task_id, qstash_message_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] create_run error: {e}")
        return False


def update_run(
    run_id: str,
    *,
    status: str | None = None,
    thread_id: str | None = None,
    publish_id: str | None = None,
    summary: str | None = None,
    error: str | None = None,
) -> None:
    fields = []
    values = []
    for col, val in (
        ("status", status),
        ("thread_id", thread_id),
        ("publish_id", publish_id),
        ("summary", summary),
        ("error", error),
    ):
        if val is not None:
            fields.append(f"{col} = %s")
            values.append(val)
    if not fields:
        return
    fields.append("updated_at = NOW()")
    sql = f"UPDATE scheduled_task_runs SET {', '.join(fields)} WHERE run_id = %s"
    values.append(run_id)
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, values)
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] update_run error: {e}")


def list_runs_for_task(task_id: str, limit: int = 20) -> list[dict]:
    sql = """
        SELECT * FROM scheduled_task_runs
        WHERE task_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (task_id, limit))
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"[db_scheduled_tasks] list_runs_for_task error: {e}")
        return []
