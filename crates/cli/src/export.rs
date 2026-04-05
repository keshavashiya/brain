//! Export / Import commands — backup and restore memory data.
//!
//! All data access goes through SignalProcessor (via bootstrap) to ensure
//! consistency with the rest of the pipeline.

/// JSON envelope written / read by `brain export` / `brain import`.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemoryExport {
    version: String,
    exported_at: String,
    facts: Vec<signal::ExportedFact>,
    episodes: Vec<signal::ExportedEpisode>,
}

pub(crate) async fn cmd_export(
    config: &brain_core::BrainConfig,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let processor = crate::bootstrap::build_processor(config).await?;

    let facts = processor.export_facts()?;
    let episodes = processor.export_episodes()?;

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

    let processor = crate::bootstrap::build_processor(config).await?;

    // Import facts and episodes via the processor
    let (facts_imported, new_fact_indices) = processor.import_facts(&export.facts)?;
    let episodes_imported = processor.import_episodes(&export.episodes)?;

    println!(
        "Imported: {} new facts, {} new episodes ({} facts and {} episodes already existed).",
        facts_imported,
        episodes_imported,
        export.facts.len() - facts_imported,
        export.episodes.len() - episodes_imported,
    );

    // Re-embed newly imported facts into the vector index
    if !new_fact_indices.is_empty() {
        let new_facts: Vec<signal::ExportedFact> = new_fact_indices
            .iter()
            .map(|&idx| export.facts[idx].clone())
            .collect();

        let (embedded, failed) = processor.reembed_facts(&new_facts).await;

        if embedded > 0 {
            println!("Re-embedded {embedded} facts into vector index.");
        }
        if failed > 0 {
            println!("Warning: {failed} facts failed to embed.");
        }
        if embedded == 0 && failed == 0 {
            println!(
                "Warning: Embedding unavailable. \
                 Imported facts will not appear in vector search until re-embedded."
            );
        }
    }

    Ok(())
}
