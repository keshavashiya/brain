use super::*;

// Migration versions that must exist after a fresh `migrate()`. Gaps
// (8, 9) are intentional — they were removed during development. Adding a
// new migration means appending the version here AND regenerating the
// schema snapshot below.
const EXPECTED_MIGRATION_VERSIONS: &[i64] = &[
    1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
];

/// Committed snapshot of the post-migration `sqlite_master` schema.
/// Regenerate by running the workspace tests with `UPDATE_SCHEMA_SNAPSHOT=1`.
const SCHEMA_SNAPSHOT: &str = include_str!("schema.snapshot.txt");

#[test]
fn test_open_memory() {
    let pool = SqlitePool::open_memory().unwrap();
    let version = pool.schema_version().unwrap();
    assert_eq!(version, 25);
}

#[test]
fn open_connections_reports_pool_state() {
    let pool = SqlitePool::open_memory().unwrap();
    // The in-memory pool is built with max_size = 1; opening establishes at
    // least one connection, and the gauge never exceeds the configured cap.
    let n = pool.open_connections();
    assert!(n >= 1, "an opened pool must hold at least one connection");
    assert!(n <= 1, "in-memory pool is capped at max_size = 1");
}

#[test]
fn test_migrations_idempotent() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.migrate().unwrap();
    assert_eq!(pool.schema_version().unwrap(), 25);
}

#[test]
fn test_migrations_record_all_expected_versions() {
    let pool = SqlitePool::open_memory().unwrap();
    let recorded: Vec<i64> = pool
        .with_conn(|conn| {
            let mut stmt = conn.prepare("SELECT version FROM _migrations ORDER BY version")?;
            let rows: Result<Vec<i64>, _> =
                stmt.query_map([], |row| row.get::<_, i64>(0))?.collect();
            Ok(rows?)
        })
        .unwrap();
    assert_eq!(
        recorded, EXPECTED_MIGRATION_VERSIONS,
        "migration versions in _migrations table drifted from the declared list — \
         did someone add/remove a migration without updating EXPECTED_MIGRATION_VERSIONS?"
    );
}

#[test]
fn test_schema_snapshot_matches() {
    let pool = SqlitePool::open_memory().unwrap();
    let actual = dump_schema(&pool);

    if std::env::var("UPDATE_SCHEMA_SNAPSHOT").is_ok() {
        // Write to the source-tree snapshot. Manifest dir is the crate root.
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/sqlite/schema.snapshot.txt");
        std::fs::write(&path, &actual)
            .unwrap_or_else(|e| panic!("failed to update snapshot at {path:?}: {e}"));
        return;
    }

    if SCHEMA_SNAPSHOT != actual {
        panic!(
            "post-migration schema drifted from snapshot.\n\
             Run `UPDATE_SCHEMA_SNAPSHOT=1 cargo test -p brainos-storage \
             test_schema_snapshot_matches` to regenerate, then review the diff \
             before committing.\n\n--- expected (committed) ---\n{SCHEMA_SNAPSHOT}\n\
             --- actual (current) ---\n{actual}"
        );
    }
}

/// Deterministic dump of `sqlite_master` (tables, indexes, triggers, views).
/// Excludes auto-generated FTS shadow tables, autoindex rows, and the
/// migration-tracking metadata table itself.
fn dump_schema(pool: &SqlitePool) -> String {
    pool.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT type, name, sql FROM sqlite_master \
             WHERE sql IS NOT NULL \
               AND name NOT LIKE 'sqlite_%' \
               AND name NOT LIKE '%_fts_%' \
               AND name != '_migrations' \
             ORDER BY type, name",
        )?;
        let rows: Vec<(String, String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<_, _>>()?;
        let mut out = String::new();
        for (ty, name, sql) in rows {
            out.push_str(&format!("-- {ty}: {name}\n{}\n\n", sql.trim()));
        }
        Ok(out)
    })
    .unwrap()
}

#[test]
fn test_v20_graph_tables_created() {
    let pool = SqlitePool::open_memory().unwrap();
    let (nodes_n, edges_n) = pool
        .with_conn(|conn| {
            let n: i64 = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
            let e: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
            Ok((n, e))
        })
        .unwrap();
    assert_eq!(nodes_n, 0);
    assert_eq!(edges_n, 0);
}

#[test]
fn test_v20_edge_cascades_when_node_deleted() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute("PRAGMA foreign_keys = ON", [])?;
        conn.execute("INSERT INTO sessions(id) VALUES ('s1')", [])?;
        conn.execute(
            "INSERT INTO nodes(id, session_id, node_kind, body_json) VALUES ('a','s1','k','{}')",
            [],
        )?;
        conn.execute(
            "INSERT INTO nodes(id, session_id, node_kind, body_json) VALUES ('b','s1','k','{}')",
            [],
        )?;
        conn.execute(
            "INSERT INTO edges(src_id, dst_id, edge_kind) VALUES ('a','b','rel')",
            [],
        )?;
        conn.execute("DELETE FROM nodes WHERE id='a'", [])?;
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        assert_eq!(n, 0, "edge must cascade-delete with its src node");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_table_stats_empty() {
    let pool = SqlitePool::open_memory().unwrap();
    let stats = pool.table_stats().unwrap();
    assert_eq!(stats.len(), 9);
    for (_, count) in &stats {
        assert_eq!(*count, 0);
    }
}

#[test]
fn test_scheduled_intent_lifecycle() {
    let pool = SqlitePool::open_memory().unwrap();
    let id = pool
        .insert_scheduled_intent(
            "deploy release",
            Some("0 9 * * 1-5"),
            "work",
            Some(r#"{"source":"test"}"#),
        )
        .unwrap();

    let all = pool.list_scheduled_intents(None).unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].id, id);
    assert_eq!(all[0].namespace, "work");
    assert_eq!(all[0].status, "scheduled");

    let personal = pool.list_scheduled_intents(Some("personal")).unwrap();
    assert!(personal.is_empty());

    let work = pool.list_scheduled_intents(Some("work")).unwrap();
    assert_eq!(work.len(), 1);
    assert_eq!(work[0].description, "deploy release");
    assert_eq!(work[0].cron.as_deref(), Some("0 9 * * 1-5"));
    assert!(work[0].created_at.contains(':'));
    assert_eq!(work[0].metadata.as_deref(), Some(r#"{"source":"test"}"#));

    let updated = pool
        .update_scheduled_intent_status(&id, "cancelled")
        .unwrap();
    assert!(updated);

    let work_after = pool.list_scheduled_intents(Some("work")).unwrap();
    assert_eq!(work_after[0].status, "cancelled");
}

#[test]
fn test_insert_and_query_session() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute(
            "INSERT INTO sessions (id, channel) VALUES (?1, ?2)",
            rusqlite::params!["sess001", "cli"],
        )?;

        let channel: String = conn.query_row(
            "SELECT channel FROM sessions WHERE id = ?1",
            ["sess001"],
            |row| row.get(0),
        )?;
        assert_eq!(channel, "cli");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_insert_episode_with_fk() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute("INSERT INTO sessions (id) VALUES (?1)", ["sess001"])?;

        conn.execute(
            "INSERT INTO episodes (id, session_id, role, content)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params!["ep001", "sess001", "user", "Hello Brain!"],
        )?;

        let content: String = conn.query_row(
            "SELECT content FROM episodes WHERE id = ?1",
            ["ep001"],
            |row| row.get(0),
        )?;
        assert_eq!(content, "Hello Brain!");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_fk_constraint_enforced() {
    let pool = SqlitePool::open_memory().unwrap();
    let result = pool.with_conn(|conn| {
        conn.execute(
            "INSERT INTO episodes (id, session_id, role, content)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params!["ep001", "nonexistent", "user", "Hello"],
        )?;
        Ok(())
    });
    assert!(result.is_err());
}

#[test]
fn test_semantic_fact_insert() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute(
            "INSERT INTO semantic_facts (id, category, subject, predicate, object)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params!["fact001", "personal", "user", "name_is", "Keshav"],
        )?;

        let obj: String = conn.query_row(
            "SELECT object FROM semantic_facts WHERE subject = ?1 AND predicate = ?2",
            rusqlite::params!["user", "name_is"],
            |row| row.get(0),
        )?;
        assert_eq!(obj, "Keshav");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_namespace_column_on_semantic_facts() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute(
            "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params!["factw1", "work", "user", "role_is", "developer", "work"],
        )?;
        conn.execute(
            "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params!["factp1", "personal", "user", "name_is", "Keshav", "personal"],
        )?;

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM semantic_facts WHERE namespace = 'work'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(count, 1, "work namespace should have 1 fact");

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM semantic_facts WHERE namespace = 'personal'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(count, 1, "personal namespace should have 1 fact");

        let found: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM semantic_facts
                 WHERE namespace = 'work' AND predicate = 'name_is'",
                [],
                |row| row.get(0),
            )
            .unwrap_or(false);
        assert!(!found, "work namespace must not contain personal facts");

        Ok(())
    })
    .unwrap();
}

#[test]
fn test_namespace_default_is_personal() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        conn.execute(
            "INSERT INTO semantic_facts (id, category, subject, predicate, object)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params!["factdefault", "personal", "user", "likes", "Rust"],
        )?;

        let ns: String = conn.query_row(
            "SELECT namespace FROM semantic_facts WHERE id = 'factdefault'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(ns, "personal", "default namespace should be 'personal'");
        Ok(())
    })
    .unwrap();
}

#[test]
fn test_notification_outbox_lifecycle() {
    let pool = SqlitePool::open_memory().unwrap();

    let id1 = pool
        .insert_notification("Low priority nudge", 1, "habit:morning_review", None)
        .unwrap();
    let id2 = pool
        .insert_notification(
            "High priority reminder",
            3,
            "open_loop:todo",
            Some("chat-main"),
        )
        .unwrap();

    let pending = pool.pending_notifications(10).unwrap();
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].id, id2, "higher priority should come first");
    assert_eq!(pending[1].id, id1);
    assert!(pending[0].delivered_at.is_none());
    assert_eq!(pending[1].channel, None);
    assert_eq!(pending[0].channel.as_deref(), Some("chat-main"));

    assert!(pool.mark_notification_delivered(&id2).unwrap());
    let pending = pool.pending_notifications(10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].id, id1);

    assert!(!pool.mark_notification_delivered(&id2).unwrap());
}

#[test]
fn test_notification_prune() {
    let pool = SqlitePool::open_memory().unwrap();
    let id = pool.insert_notification("test", 1, "test", None).unwrap();
    pool.mark_notification_delivered(&id).unwrap();

    let pruned = pool.prune_notifications(365).unwrap();
    assert_eq!(pruned, 0, "recently delivered notifications should be kept");

    pool.with_conn(|conn| {
        conn.execute(
            "UPDATE notification_outbox SET created_at = datetime('now', '-400 days') WHERE id = ?1",
            [&id],
        )?;
        Ok(())
    })
    .unwrap();
    let pruned = pool.prune_notifications(365).unwrap();
    assert_eq!(pruned, 1, "old delivered notification should be pruned");
}

#[test]
fn test_list_namespaces_with_counts() {
    let pool = SqlitePool::open_memory().unwrap();
    pool.with_conn(|conn| {
        for i in 0..3 {
            conn.execute(
                "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
                 VALUES (?1, 'personal', 'user', 'fact', ?2, 'personal')",
                rusqlite::params![format!("p{i}"), format!("val{i}")],
            )?;
        }
        conn.execute(
            "INSERT INTO semantic_facts (id, category, subject, predicate, object, namespace)
             VALUES ('w1', 'work', 'user', 'role', 'dev', 'work')",
            [],
        )?;

        let mut stmt = conn.prepare(
            "SELECT namespace, COUNT(*) as cnt FROM semantic_facts
             WHERE superseded_by IS NULL
             GROUP BY namespace ORDER BY namespace",
        )?;
        let rows: Vec<(String, i64)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;

        assert_eq!(rows.len(), 2, "should have 2 namespaces");
        let personal = rows.iter().find(|(ns, _)| ns == "personal").unwrap();
        assert_eq!(personal.1, 3, "personal should have 3 facts");
        let work = rows.iter().find(|(ns, _)| ns == "work").unwrap();
        assert_eq!(work.1, 1, "work should have 1 fact");

        Ok(())
    })
    .unwrap();
}

// ── version reconciliation gate (downgrade guard + pre-migration backup) ──

/// `latest_schema_version()` tracks the declared migration list, so the
/// open-time gate and `doctor` agree with what `migrate()` actually applies.
#[test]
fn latest_schema_version_matches_declared_head() {
    let head = *EXPECTED_MIGRATION_VERSIONS.last().unwrap();
    assert_eq!(SqlitePool::latest_schema_version(), head);
    // A migrated database lands exactly on that version.
    let pool = SqlitePool::open_memory().unwrap();
    assert_eq!(pool.schema_version().unwrap(), head);
}

/// A database whose schema is newer than this build is refused with
/// `SchemaTooNew` — an old binary must never operate a future schema.
#[test]
fn open_refuses_future_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.db");

    // Stand up a normal DB, then forge a migration row one past the head to
    // simulate a file written by a newer build.
    let future = SqlitePool::latest_schema_version() + 1;
    {
        let pool = SqlitePool::open(&path).unwrap();
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO _migrations (version, name) VALUES (?1, 'from_the_future')",
                rusqlite::params![future],
            )?;
            Ok(())
        })
        .unwrap();
    }

    match SqlitePool::open(&path) {
        Err(SqliteError::SchemaTooNew { found, supported }) => {
            assert_eq!(found, future);
            assert_eq!(supported, SqlitePool::latest_schema_version());
        }
        Err(e) => panic!("expected SchemaTooNew, got a different error: {e}"),
        Ok(_) => panic!("expected SchemaTooNew, but open succeeded"),
    }

    // The downgrade override un-gates the open.
    assert!(SqlitePool::open_with(&path, true).is_ok());
}

/// A pending forward migration on an existing database writes a consistent
/// `*.bak-v<old>` snapshot before mutating the schema; a fresh open writes
/// none.
#[test]
fn forward_migration_snapshots_existing_db() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.db");

    // Fresh open: nothing to snapshot (no prior schema).
    SqlitePool::open(&path).unwrap();
    let bak_after_fresh: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".bak-v"))
        .collect();
    assert!(
        bak_after_fresh.is_empty(),
        "a fresh open must not produce a backup"
    );

    // Rewind the recorded version to simulate an older on-disk schema, so the
    // next open sees a pending forward migration against existing data.
    let head = SqlitePool::latest_schema_version();
    {
        let pool = SqlitePool::open(&path).unwrap();
        pool.with_conn(|conn| {
            conn.execute("DELETE FROM _migrations WHERE version >= ?1", [head])?;
            Ok(())
        })
        .unwrap();
    }

    SqlitePool::open(&path).unwrap();
    let backup = path.with_file_name(format!("brain.db.bak-v{}", head - 1));
    assert!(
        backup.exists(),
        "expected pre-migration snapshot at {}",
        backup.display()
    );
}
