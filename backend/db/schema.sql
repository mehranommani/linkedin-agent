-- LinkedIn AI Content Agent - SQLite Schema
-- ============================================

-- Posts (replaces linkedin_content_bank.csv)
CREATE TABLE IF NOT EXISTS posts (
    id                TEXT PRIMARY KEY,
    created_at        DATETIME DEFAULT (datetime('now')),
    title             TEXT NOT NULL,
    original_url      TEXT NOT NULL UNIQUE,
    source            TEXT NOT NULL,
    source_summary    TEXT,
    image_url         TEXT,
    source_metrics    TEXT,
    extracted_content TEXT,
    post_text         TEXT NOT NULL,
    char_count        INTEGER,
    hashtags          TEXT,
    relevance_score   REAL NOT NULL,
    quality_score     REAL NOT NULL,
    faithfulness_score REAL DEFAULT 0.0,
    trending_boost    REAL DEFAULT 0.0,
    final_score       REAL,
    is_used           INTEGER DEFAULT 0,
    content_fingerprint TEXT,
    generation_attempts INTEGER DEFAULT 1,
    pipeline_run_id   TEXT,
    human_rating_avg  REAL DEFAULT 0.0,
    human_feedback_count INTEGER DEFAULT 0,
    detailed_eval_id  TEXT
);

-- Evaluations (per-post quality history)
CREATE TABLE IF NOT EXISTS evaluations (
    id                TEXT PRIMARY KEY,
    post_id           TEXT REFERENCES posts(id),
    evaluated_at      DATETIME DEFAULT (datetime('now')),
    relevance_score   REAL,
    quality_score     REAL,
    faithfulness_score REAL,
    confidence_score  REAL,
    ragas_faithfulness REAL,
    ragas_relevancy   REAL,
    passed            INTEGER,
    issues            TEXT,
    recommendations   TEXT,
    judge_model       TEXT,
    evaluation_type   TEXT
);

-- Pipeline Runs (self-improvement tracking)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              TEXT PRIMARY KEY,
    started_at      DATETIME DEFAULT (datetime('now')),
    completed_at    DATETIME,
    status          TEXT DEFAULT 'running',
    config          TEXT,
    items_fetched   INTEGER DEFAULT 0,
    items_filtered  INTEGER DEFAULT 0,
    posts_accepted  INTEGER DEFAULT 0,
    posts_rejected  INTEGER DEFAULT 0,
    avg_quality     REAL,
    avg_relevance   REAL,
    pass_rate       REAL,
    common_issues   TEXT
);

-- Sources
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    source_type     TEXT NOT NULL,
    name            TEXT NOT NULL,
    url             TEXT,
    category        TEXT,
    params          TEXT,
    is_active       INTEGER DEFAULT 1,
    priority        INTEGER DEFAULT 5,
    total_fetched   INTEGER DEFAULT 0,
    total_accepted  INTEGER DEFAULT 0,
    avg_quality     REAL DEFAULT 0.0,
    last_fetched_at DATETIME,
    last_error      TEXT,
    last_error_at   DATETIME,
    consecutive_failures INTEGER DEFAULT 0
);

-- Source Cache (TTL-based, prevents re-fetching)
CREATE TABLE IF NOT EXISTS source_cache (
    url             TEXT PRIMARY KEY,
    source_type     TEXT,
    content_hash    TEXT,
    fetched_at      DATETIME DEFAULT (datetime('now')),
    ttl_hours       INTEGER DEFAULT 6,
    raw_data        TEXT
);

-- Config (replaces ALL hardcoded values)
CREATE TABLE IF NOT EXISTS config (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    description     TEXT,
    updated_at      DATETIME DEFAULT (datetime('now'))
);

-- Trending Topics
CREATE TABLE IF NOT EXISTS trending_topics (
    id              TEXT PRIMARY KEY,
    detected_at     DATETIME DEFAULT (datetime('now')),
    topic           TEXT NOT NULL,
    mentions        INTEGER DEFAULT 0,
    momentum        REAL DEFAULT 0.0,
    sources         TEXT
);

-- Human feedback table (RLHF)
CREATE TABLE IF NOT EXISTS human_feedback (
    id              TEXT PRIMARY KEY,
    post_id         TEXT,
    created_at      DATETIME DEFAULT (datetime('now')),
    rating          INTEGER CHECK (rating BETWEEN 1 AND 5),
    text_feedback   TEXT,
    is_good_example INTEGER DEFAULT 0,
    feedback_weight REAL DEFAULT 1.0,
    feedback_source TEXT DEFAULT 'human'
);

-- Detailed per-post evaluations (6 RAGAS/DeepEval-equivalent metrics)
CREATE TABLE IF NOT EXISTS detailed_evaluations (
    id                  TEXT PRIMARY KEY,
    post_id             TEXT,
    evaluated_at        DATETIME DEFAULT (datetime('now')),
    pipeline_run_id     TEXT,
    answer_relevancy    REAL,
    faithfulness        REAL,
    hallucination       REAL,
    bias                REAL,
    toxicity            REAL,
    linkedin_quality    REAL,
    overall_score       REAL,
    passed              INTEGER,
    metric_details      TEXT,
    issues              TEXT,
    strengths           TEXT,
    judge_model         TEXT,
    evaluation_version  TEXT DEFAULT 'v2'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_score ON posts(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_posts_used ON posts(is_used);
CREATE INDEX IF NOT EXISTS idx_posts_fingerprint ON posts(content_fingerprint);
CREATE INDEX IF NOT EXISTS idx_eval_post ON evaluations(post_id);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
CREATE INDEX IF NOT EXISTS idx_sources_active ON sources(is_active);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_cache_fetched ON source_cache(fetched_at);
CREATE INDEX IF NOT EXISTS idx_feedback_post ON human_feedback(post_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON human_feedback(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_deteval_post ON detailed_evaluations(post_id);
CREATE INDEX IF NOT EXISTS idx_deteval_run ON detailed_evaluations(pipeline_run_id);
