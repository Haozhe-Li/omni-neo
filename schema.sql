-- Omni Supabase schema (reference).
--
-- Since the backend now talks to Supabase over PostgREST (HTTP), it can no
-- longer run DDL at startup the way the old psycopg setup_*_table() functions
-- did. Table/column/index changes are managed here and applied in the Supabase
-- SQL editor. This file reproduces the schema those startup functions used to
-- create; the live database already has it (data was not migrated, only the
-- access layer), so this is for fresh deploys and as documentation.
--
-- LangGraph checkpoint state is NOT here — it lives in Upstash Redis now
-- (see core/database/checkpointer.py).

-- ---------------------------------------------------------------------------
-- threads_control: ownership + retention bookkeeping (parent of user_threads)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS threads_control (
    thread_id  TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_pinned  BOOLEAN DEFAULT FALSE,
    user_id    VARCHAR(255)   -- NULL=unclaimed, 'guest_xxx'=guest, Clerk id=user
);

-- ---------------------------------------------------------------------------
-- user_threads: chat history + search text (child of threads_control)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_threads (
    thread_id   VARCHAR(255) PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    title       VARCHAR(255),
    ui_messages JSONB DEFAULT '[]',
    search_text TEXT NOT NULL DEFAULT '',
    is_pinned   BOOLEAN DEFAULT FALSE,
    origin      VARCHAR(20),   -- NULL=chat, 'scheduled_task'=scheduled research run
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_threads_thread
        FOREIGN KEY (thread_id) REFERENCES threads_control(thread_id)
);
CREATE INDEX IF NOT EXISTS idx_user_threads_user_id ON user_threads(user_id);
-- Thread search runs in the app now (substring + fuzzy rank), so the old
-- pg_trgm GIN indexes are no longer required. Kept here (commented) for
-- reference in case server-side ranking is reintroduced:
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- CREATE INDEX IF NOT EXISTS idx_user_threads_search_trgm ON user_threads USING GIN (search_text gin_trgm_ops);
-- CREATE INDEX IF NOT EXISTS idx_user_threads_title_trgm  ON user_threads USING GIN (title gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- user_memories: one freeform markdown doc per user
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_memories (
    user_id    VARCHAR(255) PRIMARY KEY,
    content    TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- user_usage: unified credit ledger (guests + signed-in)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_usage (
    user_id    VARCHAR(255) PRIMARY KEY,
    day        DATE NOT NULL DEFAULT CURRENT_DATE,
    day_used   NUMERIC(10,2) NOT NULL DEFAULT 0,
    month      VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
    month_used NUMERIC(10,2) NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- user_files: uploaded file metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_files (
    file_id           VARCHAR(255) PRIMARY KEY,
    user_id           VARCHAR(255) NOT NULL,
    thread_id         VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type         VARCHAR(255) NOT NULL,
    file_size_bytes   BIGINT DEFAULT 0,
    status            VARCHAR(50) DEFAULT 'pending',
    s3_bucket         VARCHAR(255),
    category          VARCHAR(50) NOT NULL,
    extracted_text    TEXT,
    created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(thread_id);
CREATE INDEX IF NOT EXISTS idx_user_files_user_id   ON user_files(user_id);

-- ---------------------------------------------------------------------------
-- scheduled_tasks / scheduled_task_runs: scheduled research feature
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    task_id            TEXT PRIMARY KEY,
    user_id            VARCHAR(255) NOT NULL,
    name               TEXT NOT NULL DEFAULT '',
    email              TEXT NOT NULL,
    prompt             TEXT NOT NULL,
    cron_schedule      TEXT NOT NULL,
    qstash_schedule_id TEXT,
    status             VARCHAR(20) NOT NULL DEFAULT 'active',  -- active | paused | deleted
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    updated_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_user_id ON scheduled_tasks(user_id);

CREATE TABLE IF NOT EXISTS scheduled_task_runs (
    run_id            TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES scheduled_tasks(task_id) ON DELETE CASCADE,
    thread_id         TEXT,
    publish_id        TEXT,
    title             TEXT,
    report_markdown   TEXT,
    sources           JSONB,
    summary           TEXT,
    status            VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | running | success | failed
    error             TEXT,
    qstash_message_id TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_task_id ON scheduled_task_runs(task_id);
-- Idempotency guard: QStash retries redeliver the same message id; a duplicate
-- create_run insert hits this unique index and is treated as a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_runs_qstash_msg
    ON scheduled_task_runs(qstash_message_id) WHERE qstash_message_id IS NOT NULL;
