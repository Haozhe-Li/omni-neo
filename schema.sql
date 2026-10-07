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

-- ---------------------------------------------------------------------------
-- sft_examples / harness_snapshots / sft_images: thumbs-up -> distillation data
-- ---------------------------------------------------------------------------
-- A thumbs-up on a teacher-model (luna) answer captures the whole conversation
-- up to and including that turn, already in the shape the fine-tune consumes
-- (OpenAI-style messages; see core/sft_capture.py). finetune/rix_gemma/
-- build_dataset.py turns the rows into the training JSONL.
--
-- Deliberately NOT tied to threads_control/user_threads: deleting a thread does
-- not delete the example. Erasing a user (DELETE /user-data) does.
--
-- harness_snapshots: the assembled system prompt + tool schemas the turn was
-- served under. A LoRA is keyed to exactly one prompt, so every example records
-- which one; the text is stored once per hash rather than once per example.
CREATE TABLE IF NOT EXISTS harness_snapshots (
    harness_hash       VARCHAR(32) PRIMARY KEY,
    system_prompt      TEXT NOT NULL,
    tools              JSONB NOT NULL,
    deepagents_version VARCHAR(32),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- sft_images: image bytes pulled out of the stored messages (which reference
-- them as omni-image://<sha256>) so a multi-turn thread's repeated image is one
-- row, and so the example survives the S3 upload being deleted.
CREATE TABLE IF NOT EXISTS sft_images (
    sha256     VARCHAR(64) PRIMARY KEY,
    mime       VARCHAR(100) NOT NULL,
    data       BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sft_examples (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    thread_id         VARCHAR(255) NOT NULL,
    -- The frontend's turn number for the *user* message of this exchange
    -- (1-indexed, odd) — the same value QueryRequest.turn carries.
    turn              INTEGER NOT NULL,
    user_id           VARCHAR(255) NOT NULL,
    harness_hash      VARCHAR(32) NOT NULL REFERENCES harness_snapshots (harness_hash),
    -- Model names reported (response_metadata.model_name) by every assistant
    -- message in the captured prefix. The thumbed turn itself is always the
    -- teacher — the endpoint refuses otherwise — but earlier turns of a thread
    -- may have come from another model, and the loss mask covers every
    -- assistant turn, so the builder filters on this.
    models_seen       TEXT[] NOT NULL DEFAULT '{}',
    -- [{role: user|assistant|tool, ...}], no system message (that is the
    -- snapshot's). Ends on the thumbed assistant answer.
    messages          JSONB NOT NULL,
    n_assistant_turns INTEGER NOT NULL DEFAULT 0,
    n_tool_calls      INTEGER NOT NULL DEFAULT 0,
    tools_used        TEXT[] NOT NULL DEFAULT '{}',
    approx_chars      INTEGER NOT NULL DEFAULT 0,
    -- Privacy / trainability flags, so the builder can exclude without parsing.
    has_image         BOOLEAN NOT NULL DEFAULT FALSE,
    has_memory        BOOLEAN NOT NULL DEFAULT FALSE,   -- <user_memory> block present
    has_attachments   BOOLEAN NOT NULL DEFAULT FALSE,   -- <attached_files> block present
    compacted         BOOLEAN NOT NULL DEFAULT FALSE,   -- deepagents summarised history: the model saw less than `messages`
    -- A thumbs-up is a noisy label; this is where a human overrules it.
    status            VARCHAR(16) NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'accepted', 'rejected')),
    status_note       TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (thread_id, turn)
);
CREATE INDEX IF NOT EXISTS idx_sft_examples_user ON sft_examples (user_id);
CREATE INDEX IF NOT EXISTS idx_sft_examples_build ON sft_examples (harness_hash, status);

-- Rows written by the training-data collector (core/routers/collector.py, the
-- password-protected /collect page) rather than by a thumbs-up. Same table, same
-- `messages` shape, same harness snapshot: a collected example is built from the
-- real LangGraph checkpoint of a real agent run, so the builder treats both alike.
--   source              'thumbs' (default, every pre-existing row) | 'collector'
--   edited              a human rewrote at least one final answer in the thread
--   original_final_text the model's last answer before editing (NULL if unedited)
--   collect_meta        per-turn inputs the annotator chose + the edit log
-- Collector rows may carry a <user_memory> block (has_memory) — memory a human
-- wrote for the occasion, not a real person's — so the builder exempts them from
-- its has_memory filter. A thumbs-up still never records one.
ALTER TABLE sft_examples ADD COLUMN IF NOT EXISTS source VARCHAR(16) NOT NULL DEFAULT 'thumbs';
ALTER TABLE sft_examples ADD COLUMN IF NOT EXISTS edited BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE sft_examples ADD COLUMN IF NOT EXISTS original_final_text TEXT;
ALTER TABLE sft_examples ADD COLUMN IF NOT EXISTS collect_meta JSONB;
CREATE INDEX IF NOT EXISTS idx_sft_examples_source ON sft_examples (source);

-- collector_turns: what the annotator asked for on each turn of a collector
-- thread, and the edit they made to its answer. One row per (thread, turn); the
-- turn is the frontend-style odd number (1, 3, 5 ...). Written by
-- POST /api/collector/generate and PUT /api/collector/threads/{id}/final, read at
-- submit to fill sft_examples.collect_meta and to refuse a thread whose turns
-- did not all go through the collector.
CREATE TABLE IF NOT EXISTS collector_turns (
    thread_id           VARCHAR(255) NOT NULL,
    turn                INTEGER NOT NULL,
    user_id             VARCHAR(255) NOT NULL,
    model               VARCHAR(32),
    personalization     JSONB NOT NULL,          -- exactly what was sent: datetime / location / language
    memory              TEXT,                    -- first turn only
    original_final_text TEXT,                    -- set the first time this turn's answer is edited
    edited_final_text   TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, turn)
);
CREATE INDEX IF NOT EXISTS idx_collector_turns_user ON collector_turns (user_id);

-- ---------------------------------------------------------------------------
-- shared_threads / thread_forks: share a conversation by link
-- ---------------------------------------------------------------------------
-- A share is a frozen, Postgres-only snapshot of a thread (core/sharing.py): what
-- the UI renders, the serialized agent state, the citations, and attachment
-- metadata. Viewing one needs nothing else. Continuing one forks it into a
-- private thread for the viewer, which is when its Redis checkpoint, citation
-- list and Qdrant chunks are built (POST /api/shared/{share_id}/fork).
--
-- Deliberately not tied to the source thread: no foreign key, so the owner
-- deleting or continuing the original leaves the link intact. Revoking a share
-- deletes the row; forks already made are independent copies and are not touched.
-- Guests cannot own shares, so these rows are exempt from the guest/user thread
-- retention sweep by construction.
CREATE TABLE IF NOT EXISTS shared_threads (
    share_id         VARCHAR(64) PRIMARY KEY,           -- random, unguessable
    owner_id         VARCHAR(255) NOT NULL,
    source_thread_id VARCHAR(255),                       -- informational only
    title            VARCHAR(255),
    ui_messages      JSONB NOT NULL,                     -- what the public page renders
    agent_state      JSONB NOT NULL,                     -- {"messages": [...], "files": {...}}, memory/location stripped
    citations        JSONB NOT NULL DEFAULT '[]',
    files_meta       JSONB NOT NULL DEFAULT '[]',        -- user_files metadata for the attachments
    n_messages       INTEGER NOT NULL DEFAULT 0,
    size_bytes       INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shared_threads_owner ON shared_threads (owner_id, created_at DESC);

-- thread_forks: marks a thread as a copy of a shared one. `inherited_messages` is
-- how many ui_messages came with it — the turns before that cannot be
-- regenerated or edited (a fork has one checkpoint, not a history). Cascades with
-- the thread; share_id has no foreign key so revoking the share leaves forks alone.
CREATE TABLE IF NOT EXISTS thread_forks (
    thread_id          VARCHAR(255) PRIMARY KEY REFERENCES threads_control (thread_id) ON DELETE CASCADE,
    share_id           VARCHAR(64) NOT NULL,
    inherited_messages INTEGER NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
