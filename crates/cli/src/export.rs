//! Export / Import commands — backup and restore memory data.

use crate::encryption::resolve_llm_api_key;

/// JSON envelope written / read by `brain export` / `brain import`.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemoryExport {
    version: String,
    exported_at: String,
    facts: Vec<ExportFact>,
    episodes: Vec<ExportEpisode>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportFact {
    id: String,
    namespace: String,
    category: String,
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
    source_episode_id: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportEpisode {
    id: String,
    session_id: String,
    session_channel: String,
    #[serde(default = "default_export_namespace")]
    namespace: String,
    role: String,
    content: String,
    timestamp: String,
    importance: f64,
    reinforcement_count: i32,
}

fn default_export_namespace() -> String {
    "personal".to_string()
}

pub(crate) fn cmd_export(config: &brain_core::BrainConfig, output: Option<&str>) -> anyhow::Result<()> {
    let db = storage::SqlitePool::open(&config.sqlite_path())?;

    let facts: Vec<ExportFact> = db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, category, subject, predicate, object,
                    confidence, source_episode_id
             FROM semantic_facts
             ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ExportFact {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    category: row.get(2)?,
                    subject: row.get(3)?,
                    predicate: row.get(4)?,
                    object: row.get(5)?,
                    confidence: row.get(6)?,
                    source_episode_id: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    })?;

    let episodes: Vec<ExportEpisode> = db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT e.id, e.session_id, COALESCE(s.channel, 'cli'),
                    e.namespace, e.role, e.content, e.timestamp,
                    e.importance, e.reinforcement_count
             FROM episodes e
             LEFT JOIN sessions s ON s.id = e.session_id
             ORDER BY e.timestamp ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ExportEpisode {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    session_channel: row.get(2)?,
                    namespace: row.get(3)?,
                    role: row.get(4)?,
                    content: row.get(5)?,
                    timestamp: row.get(6)?,
                    importance: row.get(7)?,
                    reinforcement_count: row.get(8)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    })?;

    let n_facts = facts.len();
    let n_episodes = episodes.len();

    let export = MemoryExport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        facts,
        episodes,
    };

    let json = serde_json::to_string_pretty(&export)?;

    match output {
        Some(path) => {
            std::fs::write(path, &json)?;
            println!("Exported {n_facts} facts and {n_episodes} episodes to {path}");
        }
        None => {
            println!("{}", json);
        }
    }

    Ok(())
}

pub(crate) async fn cmd_import(
    config: &brain_core::BrainConfig,
    file: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let raw =
        std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("Cannot read {file}: {e}"))?;
    let export: MemoryExport =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("Invalid export file: {e}"))?;

    println!(
        "Import preview: {} facts, {} episodes (exported at {})",
        export.facts.len(),
        export.episodes.len(),
        export.exported_at,
    );

    if dry_run {
        println!("Dry-run: no changes written.");
        return Ok(());
    }

    let db = storage::SqlitePool::open(&config.sqlite_path())?;

    let mut sessions: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for ep in &export.episodes {
        sessions
            .entry(ep.session_id.clone())
            .or_insert_with(|| ep.session_channel.clone());
    }

    let mut facts_imported = 0usize;
    let mut episodes_imported = 0usize;
    let mut new_fact_ids: Vec<usize> = Vec::new();

    db.with_conn(|conn| {
        for (sid, channel) in &sessions {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES (?1, ?2)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![sid, channel],
            )?;
        }

        for (idx, f) in export.facts.iter().enumerate() {
            let n = conn.execute(
                "INSERT INTO semantic_facts
                    (id, namespace, category, subject, predicate, object,
                     confidence, source_episode_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![
                    f.id,
                    f.namespace,
                    f.category,
                    f.subject,
                    f.predicate,
                    f.object,
                    f.confidence,
                    f.source_episode_id
                ],
            )?;
            if n > 0 {
                new_fact_ids.push(idx);
            }
            facts_imported += n;
        }

        for e in &export.episodes {
            let n = conn.execute(
                "INSERT INTO episodes
                    (id, session_id, namespace, role, content, timestamp,
                     importance, reinforcement_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![
                    e.id,
                    e.session_id,
                    e.namespace,
                    e.role,
                    e.content,
                    e.timestamp,
                    e.importance,
                    e.reinforcement_count
                ],
            )?;
            episodes_imported += n;
        }

        Ok(())
    })?;

    println!(
        "Imported: {} new facts, {} new episodes ({} facts and {} episodes already existed).",
        facts_imported,
        episodes_imported,
        export.facts.len() - facts_imported,
        export.episodes.len() - episodes_imported,
    );

    // Re-embed newly imported facts into RuVector
    if !new_fact_ids.is_empty() {
        let embedding_dim = config.embedding.dimensions as usize;
        let ruv_result = storage::RuVectorStore::open(&config.ruvector_path(), embedding_dim).await;

        match ruv_result {
            Ok(ruv) => {
                ruv.ensure_tables().await.ok();

                let llm_api_key = resolve_llm_api_key(config);
                let embedder = match config.llm.provider.as_str() {
                    "openai" => hippocampus::Embedder::for_openai(
                        &config.llm.base_url,
                        &config.embedding.model,
                        &llm_api_key,
                    ),
                    _ => hippocampus::Embedder::for_ollama(
                        &config.llm.base_url,
                        &config.embedding.model,
                    ),
                };

                let mut embedded = 0usize;
                let mut failed = 0usize;

                for &idx in &new_fact_ids {
                    let f = &export.facts[idx];
                    let text = format!("{} {} {}", f.subject, f.predicate, f.object);

                    match embedder.embed(&text).await {
                        Ok(vector) => {
                            let now = chrono::Utc::now().to_rfc3339();
                            if let Err(e) = ruv
                                .add_vectors(
                                    "facts_vec",
                                    vec![f.id.clone()],
                                    vec![text],
                                    vec![vector],
                                    vec![now],
                                    "semantic",
                                )
                                .await
                            {
                                tracing::warn!("RuVector insert failed for fact {}: {e}", f.id);
                                failed += 1;
                            } else {
                                embedded += 1;
                            }
                        }
                        Err(e) => {
                            if embedded == 0 && failed == 0 {
                                println!(
                                    "Warning: Embedding unavailable ({e}). \
                                     Imported facts will not appear in vector search until re-embedded."
                                );
                                break;
                            }
                            failed += 1;
                        }
                    }
                }

                if embedded > 0 {
                    println!("Re-embedded {embedded} facts into vector index.");
                }
                if failed > 0 {
                    println!("Warning: {failed} facts failed to embed.");
                }
            }
            Err(e) => {
                println!(
                    "Warning: RuVector unavailable ({e}). \
                     Imported facts visible in SQLite but not vector search."
                );
            }
        }
    }

    Ok(())
}
