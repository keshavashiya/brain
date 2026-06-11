//! Export / import commands — backup and restore memory data.
//!
//! All operations go through the running daemon's HTTP API to ensure
//! a single shared SignalProcessor (no RuVector lock contention).

use std::time::Duration;

/// JSON envelope written / read by `brain export` / `brain import`.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemoryExport {
    version: String,
    exported_at: String,
    /// Namespaces whose residency policy was `local_only` at export
    /// time. Facts/episodes in them (or their sub-namespaces) have
    /// never been sent off-machine — handle this file accordingly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    local_only_namespaces: Vec<String>,
    facts: Vec<signal::ExportedFact>,
    episodes: Vec<signal::ExportedEpisode>,
}

pub(crate) async fn cmd_export(
    config: &brain::BrainConfig,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;

    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;

    let resp = client
        .get(format!("{daemon_url}/v1/memory/export"))
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let facts: Vec<signal::ExportedFact> = serde_json::from_value(resp["facts"].clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse facts: {e}"))?;
    let episodes: Vec<signal::ExportedEpisode> =
        serde_json::from_value(resp["episodes"].clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse episodes: {e}"))?;

    let n_facts = facts.len();
    let n_episodes = episodes.len();

    let export = MemoryExport {
        version: resp["version"].as_str().unwrap_or("unknown").to_string(),
        exported_at: resp["exported_at"].as_str().unwrap_or("").to_string(),
        local_only_namespaces: config
            .memory
            .local_only_namespaces()
            .into_iter()
            .map(String::from)
            .collect(),
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
    config: &brain::BrainConfig,
    file: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let daemon_url = crate::bootstrap::require_daemon(config).await?;

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

    let api_key = config
        .access
        .api_keys
        .first()
        .map(|k| k.key.clone())
        .unwrap_or_default();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let import_body = serde_json::json!({
        "facts": export.facts,
        "episodes": export.episodes,
        "dry_run": false,
    });

    let resp = client
        .post(format!("{daemon_url}/v1/memory/import"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&import_body)
        .send()
        .await?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let facts_imported = resp["facts_imported"].as_u64().unwrap_or(0) as usize;
    let episodes_imported = resp["episodes_imported"].as_u64().unwrap_or(0) as usize;
    let facts_existed = resp["facts_already_existed"].as_u64().unwrap_or(0) as usize;
    let episodes_existed = resp["episodes_already_existed"].as_u64().unwrap_or(0) as usize;
    let embedded = resp["embedded"].as_u64().unwrap_or(0) as usize;
    let embed_failed = resp["embed_failed"].as_u64().unwrap_or(0) as usize;

    println!(
        "Imported: {} new facts, {} new episodes ({} facts and {} episodes already existed).",
        facts_imported, episodes_imported, facts_existed, episodes_existed,
    );

    if embedded > 0 {
        println!("Re-embedded {embedded} facts into vector index.");
    }
    if embed_failed > 0 {
        println!("Warning: {embed_failed} facts failed to embed.");
    }

    Ok(())
}
