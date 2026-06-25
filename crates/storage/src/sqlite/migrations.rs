use tracing::info;

use super::{SqliteError, SqlitePool};

impl SqlitePool {
    /// Run all schema migrations.
    pub(crate) fn migrate(&self) -> Result<(), SqliteError> {
        self.with_conn(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS _migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
                );",
            )?;

            let current_version: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(version), 0) FROM _migrations",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            let migrations = Self::migrations();

            for (version, name, sql) in &migrations {
                if *version > current_version {
                    info!("Running migration {version}: {name}");
                    conn.execute_batch(sql).map_err(|e| {
                        SqliteError::Migration(format!("Migration {version} ({name}) failed: {e}"))
                    })?;

                    conn.execute(
                        "INSERT INTO _migrations (version, name) VALUES (?1, ?2)",
                        rusqlite::params![version, name],
                    )?;
                }
            }

            if current_version < migrations.last().map_or(0, |m| m.0) {
                info!(
                    "Migrations complete (v{current_version} → v{})",
                    migrations.last().expect("BUG: migrations list is empty").0
                );
            }

            Ok(())
        })
    }

    /// All schema migrations in order.
    fn migrations() -> Vec<(i64, &'static str, &'static str)> {
        vec![
            (
                1,
                "create_sessions",
                "
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL DEFAULT (datetime('now')),
                    ended_at TEXT,
                    channel TEXT NOT NULL DEFAULT 'cli',
                    metadata TEXT
                );
            ",
            ),
            (
                2,
                "create_episodes",
                "
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    importance REAL NOT NULL DEFAULT 0.5,
                    decay_rate REAL NOT NULL DEFAULT 0.1,
                    reinforcement_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
                CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance DESC);
            ",
            ),
            (
                3,
                "create_episodes_fts",
                "
                CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                    content,
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                );
            ",
            ),
            (
                4,
                "create_semantic_facts",
                "
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source_episode_id TEXT REFERENCES episodes(id) ON DELETE SET NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    superseded_by TEXT REFERENCES semantic_facts(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_facts_category ON semantic_facts(category);
                CREATE INDEX IF NOT EXISTS idx_facts_subject ON semantic_facts(subject);
            ",
            ),
            (
                5,
                "create_user_profile",
                "
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    source TEXT,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            ",
            ),
            (
                6,
                "create_procedures",
                "
                CREATE TABLE IF NOT EXISTS procedures (
                    id              TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    steps_json      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    use_count       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                    ON procedures(trigger_pattern);
            ",
            ),
            (
                7,
                "create_audit_log",
                "
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                    action TEXT NOT NULL,
                    details TEXT,
                    prev_hash TEXT,
                    hash TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp DESC);
            ",
            ),
            (
                10,
                "add_namespace_to_semantic_facts",
                "
                ALTER TABLE semantic_facts ADD COLUMN namespace TEXT NOT NULL DEFAULT 'personal';
                CREATE INDEX IF NOT EXISTS idx_facts_namespace ON semantic_facts(namespace);
            ",
            ),
            (
                11,
                "add_namespace_to_episodes",
                "
                ALTER TABLE episodes ADD COLUMN namespace TEXT NOT NULL DEFAULT 'personal';
                CREATE INDEX IF NOT EXISTS idx_episodes_namespace ON episodes(namespace);
            ",
            ),
            (
                12,
                "rebuild_procedures_table",
                "
                DROP TABLE IF EXISTS procedures;
                CREATE TABLE IF NOT EXISTS procedures (
                    id              TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    steps_json      TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    use_count       INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_procedures_trigger
                    ON procedures(trigger_pattern);
            ",
            ),
            (
                13,
                "create_episode_promotions",
                "
                CREATE TABLE IF NOT EXISTS episode_promotions (
                    episode_id TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
                    fact_id TEXT NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
                    promoted_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            ",
            ),
            (
                14,
                "create_scheduled_intents",
                "
                CREATE TABLE IF NOT EXISTS scheduled_intents (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    cron TEXT,
                    namespace TEXT NOT NULL DEFAULT 'personal',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_intents_namespace
                    ON scheduled_intents(namespace);
                CREATE INDEX IF NOT EXISTS idx_scheduled_intents_status
                    ON scheduled_intents(status);
            ",
            ),
            (
                15,
                "create_notification_outbox",
                "
                CREATE TABLE IF NOT EXISTS notification_outbox (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 1,
                    triggered_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    delivered_at TEXT,
                    channel TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_pending
                    ON notification_outbox(delivered_at, priority, created_at)
                    WHERE delivered_at IS NULL;
            ",
            ),
            (
                16,
                "add_agent_column",
                "
                ALTER TABLE episodes ADD COLUMN agent TEXT;
                ALTER TABLE semantic_facts ADD COLUMN agent TEXT;
            ",
            ),
            (
                17,
                "fix_orphaned_facts",
                "
                -- Clear orphaned source_episode_id references
                UPDATE semantic_facts SET source_episode_id = NULL
                WHERE source_episode_id NOT IN (SELECT id FROM episodes);
                -- Clear orphaned superseded_by references
                UPDATE semantic_facts SET superseded_by = NULL
                WHERE superseded_by NOT IN (SELECT id FROM semantic_facts);
            ",
            ),
            (
                18,
                "add_performance_indexes",
                "
                -- Composite index for open-loop and habit detection
                -- (filters by role = 'user' AND timestamp >= ?)
                CREATE INDEX IF NOT EXISTS idx_episodes_role_timestamp
                    ON episodes(role, timestamp);

                -- Partial index for active (non-superseded) facts
                -- (count, list_all, list_by_namespace all filter superseded_by IS NULL)
                CREATE INDEX IF NOT EXISTS idx_facts_active
                    ON semantic_facts(superseded_by)
                    WHERE superseded_by IS NULL;
            ",
            ),
            (
                19,
                "create_dlq_entries",
                "
                CREATE TABLE IF NOT EXISTS dlq_entries (
                    id TEXT PRIMARY KEY,
                    tool_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    dlq_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_dlq_entries_tool
                    ON dlq_entries(tool_id, dlq_at DESC);
                CREATE INDEX IF NOT EXISTS idx_dlq_entries_recent
                    ON dlq_entries(dlq_at DESC);
            ",
            ),
            (
                20,
                "create_graph_nodes_edges",
                "
                -- Hippocampus graph memory. Nodes are typed entries in
                -- the episodic graph; edges link them with a typed
                -- relationship and a weight (drives both
                -- retrieval ranking and the compactor's half-life decay).
                -- Coexists with the legacy `episodes` / `semantic_facts`
                -- tables during v1.0; v1.1 deprecates the legacy store.
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT REFERENCES sessions(id),
                    namespace TEXT NOT NULL DEFAULT 'personal',
                    node_kind TEXT NOT NULL,
                    body_json TEXT NOT NULL,
                    vector_id TEXT,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                -- Primary read path: scoped by (session, namespace, kind).
                CREATE INDEX IF NOT EXISTS idx_nodes_session_ns_kind
                    ON nodes(session_id, namespace, node_kind);
                CREATE INDEX IF NOT EXISTS idx_nodes_namespace_kind
                    ON nodes(namespace, node_kind);
                CREATE INDEX IF NOT EXISTS idx_nodes_created
                    ON nodes(created_at DESC);

                CREATE TABLE IF NOT EXISTS edges (
                    src_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    dst_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    edge_kind TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (src_id, dst_id, edge_kind)
                );
                -- Traversals start from either endpoint.
                CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id, edge_kind);
                CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id, edge_kind);
            ",
            ),
            (
                21,
                "create_standing_approvals",
                "
                -- Standing approvals. A row authorizes a
                -- specific (agent_id, verb_ns, verb_action) triple to
                -- auto-approve through the ConfirmationEngine without
                -- prompting the user. A re-grant after revoke creates
                -- a new row rather than mutating the old one — keeps
                -- the audit trail intact. The partial index on
                -- (agent_id, verb_ns, verb_action) WHERE revoked_at
                -- IS NULL is the hot lookup path for `is_granted`.
                CREATE TABLE IF NOT EXISTS standing_approvals (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    verb_ns TEXT NOT NULL,
                    verb_action TEXT NOT NULL,
                    granted_at TEXT NOT NULL DEFAULT (datetime('now')),
                    revoked_at TEXT,
                    note TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_standing_approvals_lookup
                    ON standing_approvals(agent_id, verb_ns, verb_action)
                    WHERE revoked_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_standing_approvals_recent
                    ON standing_approvals(granted_at DESC);
            ",
            ),
            (
                22,
                "create_task_states",
                "
                -- Orchestrator state-machine history. One row
                -- per phase transition; the AUTOINCREMENT id doubles as
                -- a monotonic sequence so a task that re-enters a state
                -- (e.g. Executing after a replan) leaves a faithful
                -- audit trail. Replay = `ORDER BY id ASC WHERE task_id
                -- = ?`. Indexed on task_id so per-task lookups stay
                -- cheap as the table grows across many tasks.
                CREATE TABLE IF NOT EXISTS task_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    entered_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_task_states_task
                    ON task_states(task_id, id);
            ",
            ),
            (
                23,
                "create_nodes_fts",
                "
                -- Full-text index over graph node bodies so the episodic
                -- graph contributes a BM25 candidate list to recall (it was
                -- write-only w.r.t. retrieval before). Regular FTS5 index
                -- (stores its own content so hits are retrievable) mirroring
                -- `episodes_fts` (v3); the `text` column carries each node's
                -- raw `body_json`, which the porter tokenizer indexes
                -- term-wise (verbs, program names, args all searchable).
                CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                    text,
                    tokenize='porter unicode61'
                );

                -- Keep the index in sync via triggers so every writer
                -- (add_node, delete_node, the compactor) stays covered
                -- without touching Rust write paths. `nodes` has an implicit
                -- integer rowid we mirror as the FTS rowid.
                CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                    INSERT INTO nodes_fts(rowid, text) VALUES (new.rowid, new.body_json);
                END;
                CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                END;
                CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE OF body_json ON nodes BEGIN
                    DELETE FROM nodes_fts WHERE rowid = old.rowid;
                    INSERT INTO nodes_fts(rowid, text) VALUES (new.rowid, new.body_json);
                END;

                -- Backfill any nodes written before this migration.
                INSERT INTO nodes_fts(rowid, text)
                    SELECT rowid, body_json FROM nodes;
            ",
            ),
            (
                24,
                "create_capability_fitness",
                "
                -- Learned capability self-model: per-tool success/failure
                -- mass the kernel reinforces after each dispatch and decays
                -- under the forgetting curve (lazy, computed on read/write).
                -- One row per tool_id (`mcp:{server}:{tool}` or
                -- `native:{ns}.{action}`, mirroring ToolDescriptor.tool_id),
                -- so it joins directly against the live capability manifest.
                -- `*_mass` are decayed reinforcement counts (not raw tallies);
                -- `uses` is the undecayed lifetime invocation count.
                CREATE TABLE IF NOT EXISTS capability_fitness (
                    tool_id      TEXT PRIMARY KEY,
                    success_mass REAL    NOT NULL DEFAULT 0,
                    failure_mass REAL    NOT NULL DEFAULT 0,
                    uses         INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );
            ",
            ),
            (
                25,
                "create_memory_quarantine",
                "
                -- Memory-writer quarantine: a write attributed to an agent
                -- nobody vouched for (no trust entry, no key binding, no
                -- standing memory.write approval) gets a row here and is
                -- excluded from recall, search, and consolidation until the
                -- writer is approved. Approval deletes the agent's rows,
                -- releasing the memories. A side table (rather than a flag
                -- column) keeps the hot tables untouched and the membership
                -- check cheap when the table is empty — the zero-config case.
                CREATE TABLE IF NOT EXISTS memory_quarantine (
                    kind       TEXT NOT NULL CHECK(kind IN ('episode', 'fact')),
                    row_id     TEXT NOT NULL,
                    agent      TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (kind, row_id)
                );
                CREATE INDEX IF NOT EXISTS idx_quarantine_agent
                    ON memory_quarantine(agent);
            ",
            ),
            (
                26,
                "create_answer_fitness",
                "
                -- Learned answer-quality self-model: per (task-kind, model)
                -- success/failure mass scored from in-band conversational
                -- signals (a satisfied follow-up vs an immediate rephrase or
                -- correction) and decayed under the same forgetting curve as
                -- capability_fitness (lazy, computed on read/write). `model` is
                -- \"provider/model\" exactly as the L2 telemetry reports it, so a
                -- degraded model can be isolated per task kind. Feeds a bounded
                -- tier-selection bias and an honest digest line.
                CREATE TABLE IF NOT EXISTS answer_fitness (
                    kind         TEXT    NOT NULL,
                    model        TEXT    NOT NULL,
                    success_mass REAL    NOT NULL DEFAULT 0,
                    failure_mass REAL    NOT NULL DEFAULT 0,
                    uses         INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (kind, model)
                );
            ",
            ),
        ]
    }

    /// The highest schema version this build knows how to apply — the
    /// version a freshly migrated database lands on. Used by the open-time
    /// reconciliation gate to detect a future (downgrade) schema and by
    /// `doctor` to report binary-vs-disk skew.
    pub fn latest_schema_version() -> i64 {
        Self::migrations()
            .last()
            .map_or(0, |(version, _, _)| *version)
    }

    /// Get the current schema version.
    pub fn schema_version(&self) -> Result<i64, SqliteError> {
        self.with_conn(|conn| {
            let version: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(version), 0) FROM _migrations",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            Ok(version)
        })
    }
}
