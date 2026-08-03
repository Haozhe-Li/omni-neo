-- Omni evaluation schema (Supabase / PostgREST).
--
-- Same conventions as the main schema.sql: DDL is applied by hand in the
-- Supabase SQL editor, the backend only reads/writes rows over PostgREST.
-- Kept in its own file because eval data has a different lifecycle from
-- product data — it is append-only, safe to truncate, and only ever written
-- by `python -m evals.cli`, never by a request handler.
--
-- Read path is the frontend eval dashboard; the views at the bottom exist so
-- that dashboard never has to aggregate client-side.
--
-- ---------------------------------------------------------------------------
-- Applying this file
-- ---------------------------------------------------------------------------
-- Paste the whole thing into the Supabase SQL editor and run it once. Order
-- matters (tables, then views) and is already correct, so run it top to bottom
-- rather than in pieces.
--
-- Re-running: tables use CREATE TABLE IF NOT EXISTS and are safe to re-apply.
-- The views are NOT: `CREATE OR REPLACE VIEW` fails if a view's column list or
-- types changed, since it can only replace a view with an identical output
-- shape. If a view definition here has changed since you last applied it,
-- DROP VIEW that one first:
--
--   DROP VIEW IF EXISTS v_eval_model_leaderboard;
--   DROP VIEW IF EXISTS v_eval_family_grid;
--   DROP VIEW IF EXISTS v_eval_run_summary;
--   DROP VIEW IF EXISTS v_eval_check_failures;
--
-- After the DDL, PostgREST needs to see the new tables. Supabase usually
-- reloads on its own within a few seconds; if the first write 404s with
-- "relation does not exist", force it:
--
--   NOTIFY pgrst, 'reload schema';
--
-- RLS: plain CREATE TABLE leaves row-level security OFF, which is what the
-- writer needs (it uses the service_role key and writes arbitrary rows). It
-- also leaves these tables readable by the anon key — fine for an internal
-- dashboard, but decide that deliberately before pointing a public frontend at
-- them. To lock reads down instead, enable RLS per table and add a read policy
-- for whichever role the dashboard authenticates as; the service_role writer
-- bypasses RLS either way and needs no policy.
--
-- Cost stays NULL until eval_pricing has rows — see the seed template at the
-- bottom of this file.

-- ---------------------------------------------------------------------------
-- eval_cases: the case registry (what we test, and what "good" means)
--
-- Written by the CLI on every run via upsert, so the table always mirrors
-- evals/cases.yaml. It exists separately from the config so the dashboard can
-- render a case's prompt and rubric without parsing Python — and so a rubric
-- edit is visible as a `rubric_version` bump rather than silently changing
-- what old scores meant.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_cases (
    case_id        TEXT PRIMARY KEY,        -- stable slug: "web-research/sea-lions-vs-seals"
    suite          TEXT NOT NULL,           -- "web-research" | "charting" | "general" | ...
    skill          TEXT,                    -- primary skill under test, NULL for general/adversarial
    title          TEXT NOT NULL,
    lang           VARCHAR(8) NOT NULL DEFAULT 'zh',
    -- Ordered scripted user turns. Turn 0 is the real query; later turns are
    -- fixed replies that stand in for the user answering a <question> block.
    -- Scripted rather than LLM-simulated on purpose: a simulated user is a
    -- second source of variance sitting between the model and its score.
    turns          JSONB NOT NULL,          -- ["深度研究一下海狮和海豹的区别", ...]
    -- Declarative rubric, both layers, as authored:
    --   [{layer:"deterministic", key:"skill_loaded:web-research", label, weight, turn, args:{}}, ...]
    rubric         JSONB NOT NULL DEFAULT '[]',
    rubric_version INT NOT NULL DEFAULT 1,
    -- TRUE for "should NOT trigger" cases (no report / no chart / no tools).
    -- Called out as a column because over-triggering is pro mode's main
    -- failure mode and the dashboard needs to chart it separately.
    is_negative    BOOLEAN NOT NULL DEFAULT FALSE,
    weight         NUMERIC NOT NULL DEFAULT 1,
    enabled        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eval_cases_suite ON eval_cases(suite);

-- ---------------------------------------------------------------------------
-- eval_runs: one invocation of the CLI, for one model
--
-- One row per (model, batch). Comparing two models means comparing two runs,
-- which is why model identity, git sha, judge model and tool-cache mode all
-- live here — a score is only meaningful alongside the conditions that
-- produced it. `tool_cache` especially: cached and uncached runs are NOT
-- comparable, since uncached runs let search luck move the score.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label          TEXT,                    -- optional human name: "baseline 2026-08-02"
    mode           VARCHAR(16) NOT NULL DEFAULT 'pro',
    model_label    TEXT NOT NULL,           -- "gemma-4-31b-high"
    model_id       TEXT NOT NULL,           -- provider model id actually bound
    -- Split out of the label rather than left for the dashboard to parse:
    -- gpt-oss-120b is tested as a fully crossed provider x effort 2x3 grid
    -- (cerebras/groq times low/medium/high), and pivoting that grid is the
    -- point of running all six. String-slicing "gpt-oss-120b-medium-groq" in
    -- the frontend to recover the axes would be guesswork.
    provider        TEXT,                   -- cerebras | groq | google_genai
    reasoning_effort TEXT,                  -- low | medium | high | NULL when the model has none
    model_family    TEXT,                   -- "gpt-oss-120b" — groups variants for within-family comparison
    judge_model    TEXT,                    -- NULL when judging was skipped
    git_sha        VARCHAR(40),
    prompt_sha     VARCHAR(64),             -- sha256(PRO_PROMPT), so prompt edits are attributable
    skills_sha     VARCHAR(64),             -- sha256 of all SKILL.md, same reason
    repeats        INT NOT NULL DEFAULT 1,
    tool_cache     BOOLEAN NOT NULL DEFAULT FALSE,
    pricing_version INT,                    -- which eval_pricing snapshot cost_usd was computed from
    suites         JSONB NOT NULL DEFAULT '[]',   -- which suites this run covered
    status         VARCHAR(16) NOT NULL DEFAULT 'running',  -- running|done|failed
    -- Aggregates, filled in when the run finishes. Denormalized so the
    -- dashboard's run list is a single select with no joins.
    score          NUMERIC,                 -- 0..1, mean over suite means
    pass_rate      NUMERIC,                 -- 0..1, over weight>=2 checks only
    suite_scores   JSONB,                   -- {"web-research": 0.82, ...}
    n_cases        INT,
    n_errors       INT NOT NULL DEFAULT 0,
    total_latency_ms  BIGINT,
    total_cost_usd NUMERIC(12, 6),
    notes          TEXT,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_started ON eval_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_runs_model   ON eval_runs(model_label, started_at DESC);

-- ---------------------------------------------------------------------------
-- eval_results: one (run, case, repeat) — a single agent execution
--
-- `final_text` / `report_md` are stored raw so the dashboard can show exactly
-- what the model produced next to the checks that graded it. `trace` is the
-- compacted step list (tool name, arg summary, truncated result), not the raw
-- LangGraph messages — full messages routinely exceed a megabyte per pro run
-- and PostgREST would choke on them.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_results (
    result_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    case_id        TEXT NOT NULL REFERENCES eval_cases(case_id),
    repeat_idx     INT NOT NULL DEFAULT 0,
    status         VARCHAR(16) NOT NULL,    -- ok|error|timeout
    error          TEXT,

    score          NUMERIC,                 -- 0..1, weighted mean of this case's checks
    passed_hard    BOOLEAN,                 -- every weight>=2 check passed

    -- Per-turn outputs. Arrays because a multi-turn case (trip-advisor,
    -- guided-learning) produces one assistant message per turn and the rubric
    -- targets specific turns.
    final_texts    JSONB NOT NULL DEFAULT '[]',  -- ["turn 0 answer", "turn 1 answer"]
    report_md      TEXT,                    -- extracted <report> body, last turn that had one
    report_title   TEXT,

    -- Cheap counters, denormalized off the trace for charting without a scan.
    n_tool_calls   INT,
    n_searches     INT,
    n_pages_read   INT,
    n_charts       INT,
    n_maps         INT,
    has_report     BOOLEAN,
    has_question   BOOLEAN,
    word_count     INT,                     -- CJK-aware: CJK chars + latin word tokens
    skills_loaded  JSONB DEFAULT '[]',      -- ["web-research", "charting", "report-writing"]
    hit_run_limit  BOOLEAN DEFAULT FALSE,   -- ToolCallLimitMiddleware capped the run

    -- ── latency ────────────────────────────────────────────────────────────
    -- Three separate time-to-first-token measures, because in an agent run
    -- they differ by an order of magnitude and answer different questions.
    -- Collapsing them into one number would hide exactly what pro mode is
    -- judged on: it starts "thinking" fast but the prose can be 2 minutes out.
    ttft_ms        INT,   -- first chunk of ANY kind (reasoning / tool_call / text)
    ttft_answer_ms INT,   -- first token of the final answer prose
    ttft_report_ms INT,   -- moment "<report" appears in the text stream
    latency_ms     INT,   -- whole run, all turns
    per_turn_latency_ms JSONB,   -- [12043, 88120] for multi-turn cases

    n_llm_turns    INT,   -- AIMessage count = agent-loop round trips

    -- ── tokens ─────────────────────────────────────────────────────────────
    -- Summed over every LLM call in the run (billing semantics), NOT the last
    -- call. Providers disagree on streaming usage semantics — Cerebras/Groq
    -- emit one total on the final chunk, Gemini emits per-chunk deltas whose
    -- final chunk is all zeros — so these are accumulated across chunks, the
    -- only rule that is correct for both. See evals/PLAN.md §4.
    input_tokens   INT,
    output_tokens  INT,
    cached_input_tokens INT,  -- usage_metadata.input_token_details.cache_read
    reasoning_tokens    INT,  -- usage_metadata.output_token_details.reasoning
    peak_context_tokens INT,  -- largest single-call input = context high-water mark

    -- Computed at write time from eval_pricing. NULL — never 0 — when the
    -- model has no price on file, so an unpriced model reads as "unknown"
    -- instead of "free" in every downstream average.
    cost_usd       NUMERIC(12, 6),

    trace          JSONB,                   -- [{i, kind, name, args, result_head}, ...]
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_eval_results UNIQUE (run_id, case_id, repeat_idx)
);
CREATE INDEX IF NOT EXISTS idx_eval_results_run  ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_case ON eval_results(case_id);

-- ---------------------------------------------------------------------------
-- eval_checks: one rubric item, evaluated
--
-- Both layers land here under one shape (`kind` distinguishes them) so the
-- dashboard renders a single checklist per result instead of stitching two
-- sources together. Normalized rather than a jsonb blob on eval_results
-- because the useful cross-cutting query is per-check: "which rubric item
-- fails on every model", "did skill triggering regress after the prompt edit".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_checks (
    check_id       BIGSERIAL PRIMARY KEY,
    result_id      UUID NOT NULL REFERENCES eval_results(result_id) ON DELETE CASCADE,
    run_id         UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,  -- denormalized for cross-run rollups
    case_id        TEXT NOT NULL,
    kind           VARCHAR(16) NOT NULL,    -- deterministic|judge
    key            TEXT NOT NULL,           -- "skill_loaded:web-research", "judge:answers_question"
    label          TEXT NOT NULL,           -- human string shown in the dashboard
    turn           INT,                     -- which turn it graded, NULL = whole conversation
    passed         BOOLEAN,                 -- judge items: score >= threshold
    score          NUMERIC NOT NULL,        -- raw
    max_score      NUMERIC NOT NULL DEFAULT 1,
    weight         NUMERIC NOT NULL DEFAULT 1,
    -- Evidence is required for judge items (the judge must quote the text it
    -- scored) and holds the matched span / offending substring for
    -- deterministic ones — without it a red row in the UI is unactionable.
    evidence       TEXT,
    reason         TEXT,
    detail         JSONB,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_eval_checks_result ON eval_checks(result_id);
CREATE INDEX IF NOT EXISTS idx_eval_checks_run_key ON eval_checks(run_id, key);
CREATE INDEX IF NOT EXISTS idx_eval_checks_key    ON eval_checks(key, passed);

-- ---------------------------------------------------------------------------
-- eval_pricing: per-model token prices
--
-- Prices live in a table rather than in Python because they change while a
-- run's token counts never do: keeping them here means cost is recomputable
-- in SQL against historical runs instead of requiring a re-run. Each price
-- change inserts a new row (new `version`, new `effective_from`) rather than
-- updating in place, so an old run's cost stays reproducible via
-- eval_runs.pricing_version.
--
-- A model with no row here yields cost_usd = NULL, which is the intended
-- behaviour — see the note on eval_results.cost_usd.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_pricing (
    pricing_id     BIGSERIAL PRIMARY KEY,
    version        INT NOT NULL,
    model_label    TEXT NOT NULL,           -- matches eval_runs.model_label
    provider       TEXT NOT NULL,           -- cerebras | groq | google_genai
    -- USD per 1M tokens.
    usd_per_1m_input        NUMERIC(10, 4) NOT NULL,
    usd_per_1m_output       NUMERIC(10, 4) NOT NULL,
    -- Cached input is billed at a discount by every provider that reports
    -- cache_read. NULL means "not discounted / unknown" — fall back to the
    -- full input price rather than assuming free.
    usd_per_1m_cached_input NUMERIC(10, 4),
    effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source         TEXT,                    -- where the number came from, for auditability
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_eval_pricing UNIQUE (version, model_label)
);
CREATE INDEX IF NOT EXISTS idx_eval_pricing_model ON eval_pricing(model_label, version DESC);

-- ---------------------------------------------------------------------------
-- eval_case_scores: (run, case) rolled up across repeats
--
-- Materialized rather than a view: the dashboard's headline is a case × model
-- matrix, and it needs the mean/spread over repeats, not per-repeat rows.
-- `score_stdev` is the point of running repeats at all — a case whose score
-- swings between runs of the same model is telling you the rubric or the
-- prompt is unstable, not that the model got worse.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_case_scores (
    run_id         UUID NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    case_id        TEXT NOT NULL REFERENCES eval_cases(case_id),
    suite          TEXT NOT NULL,
    model_label    TEXT NOT NULL,
    n_repeats      INT NOT NULL,
    score_mean     NUMERIC,
    score_min      NUMERIC,
    score_max      NUMERIC,
    score_stdev    NUMERIC,
    pass_rate      NUMERIC,
    n_errors       INT NOT NULL DEFAULT 0,
    -- Metric medians rather than means: one 300s timeout would drag a mean
    -- latency into meaninglessness, and with 2-3 repeats the median is the
    -- more honest summary of a typical run.
    ttft_ms_p50        INT,
    ttft_answer_ms_p50 INT,
    latency_ms_p50     INT,
    n_llm_turns_p50    INT,
    total_tokens_p50   INT,
    cost_usd_mean      NUMERIC(12, 6),
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_eval_case_scores_case ON eval_case_scores(case_id, model_label);

-- ---------------------------------------------------------------------------
-- Views for the dashboard
-- ---------------------------------------------------------------------------

-- Per-run, per-suite rollup — the run detail page's top section.
CREATE OR REPLACE VIEW v_eval_run_summary AS
SELECT
    r.run_id,
    r.label,
    r.model_label,
    r.started_at,
    r.status,
    cs.suite,
    COUNT(*)                       AS n_cases,
    AVG(cs.score_mean)             AS suite_score,
    AVG(cs.pass_rate)              AS suite_pass_rate,
    SUM(cs.n_errors)               AS n_errors
FROM eval_runs r
JOIN eval_case_scores cs ON cs.run_id = r.run_id
GROUP BY r.run_id, r.label, r.model_label, r.started_at, r.status, cs.suite;

-- Which rubric items fail most often, across every run — the "what should I
-- fix next" list. Ordered so the worst offenders come first.
--
-- Grouped by (key, kind) and NOT by label. `label` embeds the check's
-- arguments, so grouping on it splits one rubric across every threshold the
-- suite happens to use — `charts_valid` once per `min_series` value,
-- `word_count` once per min/max pair — leaving several partial rows where none
-- of them answers "how often does this check fail". `key` is the stable
-- identity: config.py already folds the discriminating argument into it where
-- that matters (`skill_loaded:charting`, `tool_called:google_search`), so
-- collapsing on it merges thresholds without merging different checks.
-- `label` is kept as a representative sample, purely so the view's column list
-- is unchanged and CREATE OR REPLACE still applies over the old definition.
CREATE OR REPLACE VIEW v_eval_check_failures AS
SELECT
    c.key,
    MIN(c.label)                                      AS label,
    c.kind,
    COUNT(*)                                          AS n_evaluated,
    COUNT(*) FILTER (WHERE c.passed IS FALSE)         AS n_failed,
    ROUND(
        COUNT(*) FILTER (WHERE c.passed IS FALSE)::NUMERIC
        / NULLIF(COUNT(*), 0), 3
    )                                                 AS failure_rate
FROM eval_checks c
GROUP BY c.key, c.kind
ORDER BY failure_rate DESC, n_failed DESC;

-- Model leaderboard: quality next to what it costs and how long it makes the
-- user wait. This is the view the whole metrics layer exists for — a model is
-- only worth picking if its score justifies its latency and price, and those
-- three numbers are useless apart from each other.
--
-- Restricted to status='ok' rows for the metric columns (a timeout's 300s
-- would poison every latency percentile) while `error_rate` keeps the failures
-- visible — a model that scores well on the third of cases it survives is not
-- a model that scores well.
CREATE OR REPLACE VIEW v_eval_model_leaderboard AS
SELECT
    r.run_id,
    r.model_label,
    r.model_family,
    r.provider,
    r.reasoning_effort,
    r.started_at,
    r.score,
    r.pass_rate,
    COUNT(*)                                                   AS n_results,
    ROUND(
        COUNT(*) FILTER (WHERE res.status <> 'ok')::NUMERIC
        / NULLIF(COUNT(*), 0), 3
    )                                                          AS error_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY res.ttft_ms) FILTER (WHERE res.status = 'ok') AS ttft_ms_p50,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY res.ttft_answer_ms) FILTER (WHERE res.status = 'ok') AS ttft_answer_ms_p50,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY res.latency_ms) FILTER (WHERE res.status = 'ok')     AS latency_ms_p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY res.latency_ms) FILTER (WHERE res.status = 'ok')     AS latency_ms_p95,
    AVG(res.n_llm_turns)  FILTER (WHERE res.status = 'ok')     AS turns_mean,
    AVG(res.input_tokens) FILTER (WHERE res.status = 'ok')     AS input_tokens_mean,
    AVG(res.output_tokens) FILTER (WHERE res.status = 'ok')    AS output_tokens_mean,
    AVG(res.reasoning_tokens) FILTER (WHERE res.status = 'ok') AS reasoning_tokens_mean,
    SUM(res.cost_usd)                                          AS cost_usd_total,
    AVG(res.cost_usd)                                          AS cost_usd_per_case
FROM eval_runs r
JOIN eval_results res ON res.run_id = r.run_id
GROUP BY r.run_id, r.model_label, r.model_family, r.provider,
         r.reasoning_effort, r.started_at, r.score, r.pass_rate;

-- The gpt-oss-120b 2x3 (and gemma's low/high pair) as an actual grid: one row
-- per family x provider x effort, so the two factors can be read apart —
-- along a row, what extra reasoning effort buys; down a column, what the
-- provider changes at identical effort.
--
-- The provider comparison doubles as this eval's own control group: same
-- weights, same prompt, same effort, so quality scores SHOULD land on top of
-- each other and only latency/cost should move. A large quality gap down a
-- column is evidence of a harness or serving bug (quantization, tool-call
-- formatting, reasoning leaking into content), not of model ability — which
-- is why score and latency sit side by side here rather than in two views.
CREATE OR REPLACE VIEW v_eval_family_grid AS
SELECT
    r.model_family,
    r.provider,
    r.reasoning_effort,
    COUNT(DISTINCT r.run_id)                                    AS n_runs,
    AVG(r.score)                                                AS score,
    AVG(r.pass_rate)                                            AS pass_rate,
    AVG(res.reasoning_tokens) FILTER (WHERE res.status = 'ok')  AS reasoning_tokens_mean,
    AVG(res.output_tokens)    FILTER (WHERE res.status = 'ok')  AS output_tokens_mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY res.ttft_ms) FILTER (WHERE res.status = 'ok')  AS ttft_ms_p50,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY res.latency_ms) FILTER (WHERE res.status = 'ok') AS latency_ms_p50,
    AVG(res.cost_usd)                                           AS cost_usd_per_case,
    ROUND(
        COUNT(*) FILTER (WHERE res.status <> 'ok')::NUMERIC
        / NULLIF(COUNT(*), 0), 3
    )                                                           AS error_rate
FROM eval_runs r
JOIN eval_results res ON res.run_id = r.run_id
WHERE r.model_family IS NOT NULL
GROUP BY r.model_family, r.provider, r.reasoning_effort;

-- ---------------------------------------------------------------------------
-- Seed template for eval_pricing
-- ---------------------------------------------------------------------------
-- Cost is NULL — deliberately, never 0 — for any model without a row here, so
-- an unpriced model reads as "unknown" instead of winning every cost
-- comparison it appears in. Fill in the real per-1M-token prices from each
-- provider's pricing page before the first run you care about cost for; the
-- numbers below are placeholders, NOT quoted prices.
--
-- To change a price later, INSERT a new row with a higher `version` rather than
-- UPDATE-ing this one: eval_runs.pricing_version pins each historical run to
-- the prices it was costed with, so old numbers stay reproducible.
--
-- model_label must match evals/models.py exactly — run `python -m evals.models`
-- to print the list.

-- INSERT INTO eval_pricing
--   (version, model_label, provider, usd_per_1m_input, usd_per_1m_output,
--    usd_per_1m_cached_input, source)
-- VALUES
--   (1, 'gpt-oss-120b-low',         'cerebras',     0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-120b-medium',      'cerebras',     0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-120b-high',        'cerebras',     0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-120b-low-groq',    'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-120b-medium-groq', 'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-120b-high-groq',   'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gpt-oss-20b',              'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'qwen-3-6-27b',             'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'llama3-1-8b',              'groq',         0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gemma-4-31b',              'cerebras',     0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gemma-4-31b-high',         'cerebras',     0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gemini-flash',             'google_genai', 0.00, 0.00, NULL, 'FILL ME'),
--   (1, 'gemini-flash-lite-latest', 'google_genai', 0.00, 0.00, NULL, 'FILL ME');

-- ---------------------------------------------------------------------------
-- Smoke test: run after applying, before the first eval run.
-- ---------------------------------------------------------------------------
-- Exercises the two constructs most likely to differ by Postgres version —
-- FILTER on an ordered-set aggregate, and gen_random_uuid() — against empty
-- tables. Both views returning zero rows without error means the schema is
-- good; an error here is far cheaper to find than after a 40-minute matrix run.
--
--   SELECT * FROM v_eval_model_leaderboard;
--   SELECT * FROM v_eval_family_grid;
--   SELECT * FROM v_eval_run_summary;
--   SELECT * FROM v_eval_check_failures;
