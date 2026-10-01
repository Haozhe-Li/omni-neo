-- Omni database schema (PostgreSQL).
--
-- Idempotent: safe to run on an empty database and safe to re-run. Apply it with
--
--     python -m scripts.init_db
--
-- which also applies evals/schema_evals.sql. The backend talks to Postgres over a
-- direct connection (DATABASE_URL, see core/database/pg.py), so there is no REST
-- layer and no Row Level Security: access control is the Clerk-verified user_id
-- the application puts in each WHERE clause.
--
-- LangGraph checkpoint state is NOT here — it lives in Redis
-- (see core/database/checkpointer.py).

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- threads_control: ownership + retention bookkeeping (parent of user_threads)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS threads_control (
    thread_id     TEXT PRIMARY KEY,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_pinned     BOOLEAN NOT NULL DEFAULT FALSE,
    user_id       VARCHAR(255),   -- NULL=unclaimed, 'guest_xxx'=guest, Clerk id=user
    -- Safety lock: set once a turn on this thread trips SAFETY_TERMINATED
    -- (core/utils/errors.py). locked_reason is always "safety_terminated" —
    -- never the more granular harmful_query/prompt_leakage cause, which stays
    -- server-log-only so a locked thread's own owner can't use this field to
    -- learn which guard they tripped (see core/utils/errors.py's docstring on
    -- why that distinction is never surfaced past the log).
    is_locked     BOOLEAN NOT NULL DEFAULT FALSE,
    locked_reason VARCHAR(50),
    locked_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_threads_control_user_id ON threads_control (user_id);
-- Serves the retention sweep (cleanup_old_threads): only unpinned rows are ever deleted.
CREATE INDEX IF NOT EXISTS idx_threads_control_expiry ON threads_control (updated_at) WHERE NOT is_pinned;

-- ---------------------------------------------------------------------------
-- user_threads: chat history + search text (child of threads_control)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_threads (
    thread_id     VARCHAR(255) PRIMARY KEY
                  REFERENCES threads_control (thread_id) ON DELETE CASCADE,
    user_id       VARCHAR(255) NOT NULL,
    title         VARCHAR(255),
    ui_messages   JSONB NOT NULL DEFAULT '[]',
    search_text   TEXT NOT NULL DEFAULT '',
    is_pinned     BOOLEAN NOT NULL DEFAULT FALSE,
    origin        VARCHAR(20),   -- NULL=chat, 'voice', 'scheduled_task'=scheduled research run
    -- Mirrors threads_control.is_locked so GET /api/threads/{id} and the thread
    -- list can read lock state from the same row as ui_messages.
    is_locked     BOOLEAN NOT NULL DEFAULT FALSE,
    locked_reason VARCHAR(50),
    locked_at     TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- The sidebar list: one user's threads, pinned first, newest first.
CREATE INDEX IF NOT EXISTS idx_user_threads_list
    ON user_threads (user_id, is_pinned DESC, updated_at DESC);
-- Thread search (core/database/db_user_threads.py): substring and fuzzy matches on
-- title and body are served by trigram indexes.
CREATE INDEX IF NOT EXISTS idx_user_threads_title_trgm
    ON user_threads USING GIN (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_user_threads_search_trgm
    ON user_threads USING GIN (search_text gin_trgm_ops);

-- A short excerpt of `txt` centred on the first case-insensitive occurrence of
-- `q` (the first 80 characters when there is none), with an ellipsis on each cut
-- side. Done in SQL so a search never ships whole 50k-character bodies to Python.
CREATE OR REPLACE FUNCTION make_snippet(txt TEXT, q TEXT, radius INT DEFAULT 40)
RETURNS TEXT
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE
        WHEN txt IS NULL OR txt = '' THEN ''
        WHEN s.i < 0 THEN left(txt, 80)
        ELSE (CASE WHEN greatest(0, s.i - radius) > 0 THEN '…' ELSE '' END)
             || substr(txt, greatest(0, s.i - radius) + 1,
                       least(length(txt), s.i + length(q) + radius) - greatest(0, s.i - radius))
             || (CASE WHEN least(length(txt), s.i + length(q) + radius) < length(txt) THEN '…' ELSE '' END)
    END
    FROM (SELECT strpos(lower(txt), lower(q)) - 1 AS i) AS s
$$;

-- ---------------------------------------------------------------------------
-- user_memories: one freeform markdown doc per user
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_memories (
    user_id    VARCHAR(255) PRIMARY KEY,
    content    TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- user_usage: unified credit ledger (guests + signed-in)
-- ---------------------------------------------------------------------------
-- extra_granted/extra_used: the redeemed-code balance. Deliberately in this
-- same row rather than a table of its own — the charge path reads its whole
-- decision input in one select. These two NEVER roll over: day_used/month_used
-- reset when `day`/`month` go stale, extra credits are a permanent balance
-- until spent.
CREATE TABLE IF NOT EXISTS user_usage (
    user_id       VARCHAR(255) PRIMARY KEY,
    day           DATE NOT NULL DEFAULT CURRENT_DATE,
    day_used      NUMERIC(10,2) NOT NULL DEFAULT 0,
    month         VARCHAR(7) NOT NULL DEFAULT to_char(CURRENT_DATE, 'YYYY-MM'),
    month_used    NUMERIC(10,2) NOT NULL DEFAULT 0,
    extra_granted NUMERIC(10,2) NOT NULL DEFAULT 0,
    extra_used    NUMERIC(10,2) NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- redeem_codes: prepaid credit codes (see core/database/db_redeem_codes.py)
-- ---------------------------------------------------------------------------
-- `code` is stored normalized (upper-cased, dashes/spaces stripped) so lookup
-- is a plain primary-key hit and users can type it however they like.
CREATE TABLE IF NOT EXISTS redeem_codes (
    code               VARCHAR(64) PRIMARY KEY,
    credits            NUMERIC(10,2) NOT NULL DEFAULT 1000,
    max_uses           INT NOT NULL DEFAULT 1,   -- 1 = single-use; >1 = shared campaign code;
                                                  -- <=0 = unlimited users (still once each,
                                                  -- enforced by code_redemptions' unique index)
    used_count         INT NOT NULL DEFAULT 0,
    expires_at         TIMESTAMPTZ,              -- NULL = never expires
    active             BOOLEAN NOT NULL DEFAULT TRUE,
    note               TEXT,                     -- free-form: what campaign this was for
    -- Set only by the "get free credit" self-serve flow (core/database/db_free_credit.py):
    -- when non-NULL, redeem_code() refuses anyone but this user_id, as if the
    -- code didn't exist. NULL for every ordinary/campaign code.
    restricted_user_id VARCHAR(255),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- code_redemptions: who redeemed what
-- ---------------------------------------------------------------------------
-- UNIQUE (code, user_id): whatever the code's max_uses, one user redeems it once.
CREATE TABLE IF NOT EXISTS code_redemptions (
    id          BIGSERIAL PRIMARY KEY,
    code        VARCHAR(64) NOT NULL REFERENCES redeem_codes (code),
    user_id     VARCHAR(255) NOT NULL,
    credits     NUMERIC(10,2) NOT NULL,
    redeemed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_code_redemptions_code_user UNIQUE (code, user_id)
);
CREATE INDEX IF NOT EXISTS idx_code_redemptions_user_id ON code_redemptions (user_id);

-- ---------------------------------------------------------------------------
-- user_files: uploaded file metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_files (
    file_id           VARCHAR(255) PRIMARY KEY,
    user_id           VARCHAR(255) NOT NULL,
    thread_id         VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_type         VARCHAR(255) NOT NULL,
    file_size_bytes   BIGINT NOT NULL DEFAULT 0,
    status            VARCHAR(50) NOT NULL DEFAULT 'pending',   -- pending | ready | failed
    s3_bucket         VARCHAR(255),
    category          VARCHAR(50) NOT NULL,
    extracted_text    TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files (user_id);
-- count_prior_ready_files_with_name: one thread, one filename, ordered by creation.
CREATE INDEX IF NOT EXISTS idx_user_files_thread_name
    ON user_files (thread_id, original_filename, created_at, file_id);

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
    status             VARCHAR(20) NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active', 'paused', 'deleted')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_user_id ON scheduled_tasks (user_id);

CREATE TABLE IF NOT EXISTS scheduled_task_runs (
    run_id            TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL REFERENCES scheduled_tasks (task_id) ON DELETE CASCADE,
    thread_id         TEXT,
    publish_id        TEXT,
    title             TEXT,
    report_markdown   TEXT,
    sources           JSONB,
    summary           TEXT,
    status            VARCHAR(20) NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'running', 'success', 'failed')),
    error             TEXT,
    qstash_message_id TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Run history for one task, newest first.
CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_task
    ON scheduled_task_runs (task_id, created_at DESC);
-- Idempotency guard: QStash retries redeliver the same message id; a duplicate
-- create_run insert hits this unique index and is treated as a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_runs_qstash_msg
    ON scheduled_task_runs (qstash_message_id) WHERE qstash_message_id IS NOT NULL;
