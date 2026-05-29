//! Health checks for the local environment — `brain doctor` plus the
//! lightweight Ollama probe shown at the end of `brain init`.

use brain::{BrainConfig, ProviderEntry};

/// Strip a `:tag` suffix so callers can match against Ollama's `name` field
/// regardless of whether the user wrote `nomic-embed-text` or
/// `nomic-embed-text:latest` in their config.
fn base_name(model: &str) -> &str {
    model.split(':').next().unwrap_or(model)
}

/// Fetch the list of installed Ollama models from `{base_url}/api/tags`.
/// Returns `None` when the daemon is unreachable so callers can render a
/// distinct "Ollama not running" message instead of a stack trace.
async fn list_ollama_models(base_url: &str) -> Option<Vec<String>> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .ok()?;
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    let models = json.get("models")?.as_array()?;
    Some(
        models
            .iter()
            .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
            .collect(),
    )
}

fn model_present(installed: &[String], wanted: &str) -> bool {
    let wanted_base = base_name(wanted);
    installed
        .iter()
        .any(|m| m == wanted || base_name(m) == wanted_base)
}

/// Probe an OpenAI-compatible `/models` endpoint. Returns `Some(ids)` when
/// the request succeeds (even with an empty list — auth worked, the server
/// just chose not to enumerate). Returns `None` on transport failure or a
/// non-2xx response so callers can distinguish "unreachable / unauthorized"
/// from "reachable but model absent".
async fn list_openai_compat_models(base_url: &str, api_key: &str) -> Option<Vec<String>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(4))
        .build()
        .ok()?;
    let mut req = client.get(&url);
    if !api_key.trim().is_empty() {
        req = req.bearer_auth(api_key.trim());
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let json: serde_json::Value = resp.json().await.ok()?;
    let data = json.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| m.get("id").and_then(|n| n.as_str()).map(String::from))
            .collect(),
    )
}

/// Classify a provider entry as either Ollama-style or OpenAI-compatible,
/// resolving the effective `base_url` from the preset table when the entry
/// didn't supply one explicitly.
enum ProviderKind {
    Ollama { base_url: String },
    OpenAiCompat { base_url: String },
    Unknown,
}

fn classify(entry: &ProviderEntry) -> ProviderKind {
    let kind = entry.kind.trim().to_ascii_lowercase();
    if kind == "ollama" {
        let base = if entry.base_url.trim().is_empty() {
            "http://localhost:11434".to_string()
        } else {
            entry.base_url.clone()
        };
        return ProviderKind::Ollama { base_url: base };
    }
    if kind == "openai_compat" {
        if entry.base_url.trim().is_empty() {
            return ProviderKind::Unknown;
        }
        return ProviderKind::OpenAiCompat {
            base_url: entry.base_url.clone(),
        };
    }
    if let Some(preset) = cortex::presets::resolve(&kind) {
        let base = if entry.base_url.trim().is_empty() {
            preset.base_url.to_string()
        } else {
            entry.base_url.clone()
        };
        return ProviderKind::OpenAiCompat { base_url: base };
    }
    ProviderKind::Unknown
}

/// Find a usable Ollama base URL for the embedding probe. Prefers the first
/// `kind: ollama` entry in `providers[]`, then falls back to the legacy
/// single-shape `llm.base_url`. Returns `None` only when no Ollama-shaped
/// endpoint is configured at all.
fn ollama_base_for_embedding(config: &BrainConfig) -> Option<String> {
    for p in &config.llm.providers {
        if let ProviderKind::Ollama { base_url } = classify(p) {
            return Some(base_url);
        }
    }
    #[allow(deprecated)]
    let legacy = config.llm.base_url.trim();
    if !legacy.is_empty() {
        Some(legacy.to_string())
    } else {
        None
    }
}

/// Light probe used at the end of `brain init`. Prints a one-liner per
/// provider/model. Never fails — this is a hint, not a gate.
pub(crate) async fn check_ollama_models(config: &BrainConfig) {
    println!();
    if config.llm.providers.is_empty() {
        // Legacy single-shape config — same shortcut path as before.
        #[allow(deprecated)]
        let base = config.llm.base_url.clone();
        #[allow(deprecated)]
        let llm_model = config.llm.model.clone();
        match list_ollama_models(&base).await {
            Some(installed) => print_model_status("Cortex LLM", &llm_model, &installed),
            None => {
                println!("  Ollama:    unreachable at {base}");
                println!("            install from https://ollama.com, then run:");
                println!("            ollama pull {llm_model}");
            }
        }
    } else {
        for entry in &config.llm.providers {
            print_provider_hint(entry).await;
        }
    }

    let embed_model = config.embedding.model.clone();
    match ollama_base_for_embedding(config) {
        Some(base) => match list_ollama_models(&base).await {
            Some(installed) => print_model_status("Embedding ", &embed_model, &installed),
            None => println!(
                "  Embedding: {embed_model} (Ollama at {base} unreachable — run `ollama serve`)"
            ),
        },
        None => println!(
            "  Embedding: {embed_model} (no Ollama provider configured — embeddings need Ollama)"
        ),
    }
}

async fn print_provider_hint(entry: &ProviderEntry) {
    let label = format!("Provider {:8}", entry.name);
    match classify(entry) {
        ProviderKind::Ollama { base_url } => match list_ollama_models(&base_url).await {
            Some(installed) => print_model_status(&label, &entry.model, &installed),
            None => println!(
                "  {label}: {} (Ollama at {base_url} unreachable — run `ollama serve`)",
                entry.model
            ),
        },
        ProviderKind::OpenAiCompat { base_url } => {
            match list_openai_compat_models(&base_url, &entry.api_key).await {
                Some(ids) if model_present(&ids, &entry.model) => {
                    println!("  {label}: {} (ready @ {base_url})", entry.model)
                }
                Some(_) => println!(
                    "  {label}: {} (reachable @ {base_url} — model not listed by provider; will be checked on first call)",
                    entry.model
                ),
                None => println!(
                    "  {label}: {} (unreachable or unauthorized @ {base_url})",
                    entry.model
                ),
            }
        }
        ProviderKind::Unknown => println!(
            "  {label}: kind `{}` not recognised — set kind to ollama, openai_compat, or a known preset",
            entry.kind
        ),
    }
}

fn print_model_status(label: &str, model: &str, installed: &[String]) {
    if model_present(installed, model) {
        println!("  {label}: {model} (ready)");
    } else {
        println!("  {label}: {model} (missing — run `ollama pull {model}`)");
    }
}

/// `brain doctor` — full environment check (plus store-level probes with `--deep`).
pub(crate) async fn cmd_doctor(config: &BrainConfig, deep: bool) -> anyhow::Result<()> {
    let mut failures = 0u32;
    println!("Brain doctor — environment check");

    // ── data dir writable ────────────────────────────────────────────────
    let data_dir = config.data_dir();
    let writable = if data_dir.exists() {
        let probe = data_dir.join(".brain_doctor_probe");
        let ok = std::fs::write(&probe, b"").is_ok();
        let _ = std::fs::remove_file(&probe);
        ok
    } else {
        false
    };
    if writable {
        println!("  [ok]   data dir writable      {}", data_dir.display());
    } else {
        failures += 1;
        println!(
            "  [fail] data dir not writable  {} (run `brain init`)",
            data_dir.display()
        );
    }

    // ── LLM providers ────────────────────────────────────────────────────
    // Check each entry in `providers[]` per its `kind`. Ollama entries get
    // an `/api/tags` probe; openai-compatible entries get a `/models` probe
    // with their bearer token. If `providers[]` is empty, fall back to the
    // legacy single-shape `llm.{base_url,model}` fields against Ollama.
    if config.llm.providers.is_empty() {
        #[allow(deprecated)]
        let base = config.llm.base_url.clone();
        #[allow(deprecated)]
        let model = config.llm.model.clone();
        match list_ollama_models(&base).await {
            Some(installed) => {
                println!("  [ok]   ollama reachable       {}", base);
                if model_present(&installed, &model) {
                    println!("  [ok]   LLM model installed    {}", model);
                } else {
                    failures += 1;
                    println!(
                        "  [fail] LLM model missing      {} (run `ollama pull {}`)",
                        model, model
                    );
                }
            }
            None => {
                failures += 1;
                println!(
                    "  [fail] ollama unreachable     {} (install from https://ollama.com)",
                    base
                );
            }
        }
    } else {
        let mut any_reachable = false;
        for entry in &config.llm.providers {
            if check_provider(entry).await {
                any_reachable = true;
            }
        }
        if !any_reachable {
            failures += 1;
            println!("  [fail] no LLM provider reachable — fix at least one entry above");
        }
    }

    // ── embedding (always Ollama) ────────────────────────────────────────
    let embed_model = config.embedding.model.clone();
    match ollama_base_for_embedding(config) {
        Some(base) => match list_ollama_models(&base).await {
            Some(installed) => {
                if model_present(&installed, &embed_model) {
                    println!("  [ok]   embedding installed    {}", embed_model);
                } else {
                    failures += 1;
                    println!(
                        "  [fail] embedding missing      {} (run `ollama pull {}`)",
                        embed_model, embed_model
                    );
                }
            }
            None => {
                failures += 1;
                println!(
                    "  [fail] embedding unreachable  Ollama at {} not responding (run `ollama serve`)",
                    base
                );
            }
        },
        None => {
            failures += 1;
            println!(
                "  [fail] no Ollama provider configured — embeddings require Ollama (add a `kind: ollama` entry to `llm.providers`)"
            );
        }
    }

    // ── adapter ports ────────────────────────────────────────────────────
    let daemon_url = crate::bootstrap::detect_running_daemon(config).await;
    let host = config.adapters.http.host.clone();
    let ports = [
        (
            "HTTP ",
            config.adapters.http.port,
            config.adapters.http.enabled,
        ),
        ("WS   ", config.adapters.ws.port, config.adapters.ws.enabled),
        (
            "MCP  ",
            config.adapters.mcp.port,
            config.adapters.mcp.enabled,
        ),
        (
            "gRPC ",
            config.adapters.grpc.port,
            config.adapters.grpc.enabled,
        ),
    ];
    for (name, port, enabled) in ports {
        if !enabled {
            println!("  [--]   {} port {} (disabled in config)", name, port);
            continue;
        }
        let listening = port_in_use(&host, port).await;
        match (listening, daemon_url.is_some()) {
            (true, true) => println!("  [ok]   {} port {} (Brain listening)", name, port),
            (true, false) => {
                failures += 1;
                println!(
                    "  [fail] {} port {} (in use by another process)",
                    name, port
                );
            }
            (false, _) => println!("  [ok]   {} port {} (free)", name, port),
        }
    }

    // ── api key configured ───────────────────────────────────────────────
    if config.access.api_keys.is_empty() {
        failures += 1;
        println!("  [fail] no API keys configured (run `brain init`)");
    } else {
        println!(
            "  [ok]   API keys configured    ({})",
            config.access.api_keys.len()
        );
    }

    // ── deep store-level probes (--deep) ─────────────────────────────────
    if deep {
        let daemon_up = daemon_url.is_some();
        run_deep_checks(config, daemon_up, &mut failures).await;
    }

    println!();
    if failures == 0 {
        println!("All checks passed. Run `brain start` to wake Brain.");
        Ok(())
    } else {
        anyhow::bail!(
            "{} check(s) failed — fix above and re-run `brain doctor`",
            failures
        )
    }
}

/// Deep probes that open the actual data stores. Each prints `[ok]`/`[fail]`
/// and folds into the shared `failures` tally. The vector-store open is
/// skipped when a daemon is running, since it holds the RuVector file lock.
async fn run_deep_checks(config: &BrainConfig, daemon_up: bool, failures: &mut u32) {
    println!();
    println!("Deep checks (--deep)");

    // ── SQLite: open runs migrations; success == schema current ──────────
    // The opened pool is reused for the audit-chain + counts probes below.
    let pool = match storage::SqlitePool::open(&config.sqlite_path()) {
        Ok(pool) => match pool.schema_version() {
            Ok(version) => {
                println!("  [ok]   sqlite schema          v{version} (migrations applied)");
                Some(pool)
            }
            Err(e) => {
                *failures += 1;
                println!("  [fail] sqlite schema version  {e}");
                Some(pool)
            }
        },
        Err(e) => {
            *failures += 1;
            println!(
                "  [fail] sqlite open            {} ({e})",
                config.sqlite_path().display()
            );
            None
        }
    };

    // ── audit hash-chain linkage + row counts (reuse the pool) ───────────
    if let Some(pool) = &pool {
        match check_audit_chain(pool) {
            Ok(rows) => println!("  [ok]   audit chain            {rows} row(s), linkage intact"),
            Err(e) => {
                *failures += 1;
                println!("  [fail] audit chain            {e}");
            }
        }
        match memory_counts(pool) {
            Ok((episodes, facts, nodes)) => println!(
                "  [ok]   memory counts          {episodes} episode(s), {facts} fact(s), {nodes} graph node(s)"
            ),
            Err(e) => {
                *failures += 1;
                println!("  [fail] memory counts          {e}");
            }
        }
    }

    // ── vector store (skipped while the daemon holds the lock) ───────────
    let dim = config.embedding.dimensions as usize;
    if daemon_up {
        println!("  [--]   vector store           skipped (daemon running — holds the lock)");
    } else {
        match storage::RuVectorStore::open(&config.ruvector_path(), dim).await {
            Ok(ruv) => {
                if let Err(e) = ruv.ensure_tables().await {
                    *failures += 1;
                    println!("  [fail] vector store tables    {e}");
                } else {
                    let mut parts = Vec::new();
                    for name in ["facts_vec", "episodes_vec", "graph_vec"] {
                        let n = ruv.table_count(name).await.unwrap_or(0);
                        parts.push(format!("{name}={n}"));
                    }
                    println!(
                        "  [ok]   vector store           dim {dim}, {}",
                        parts.join(" ")
                    );
                }
            }
            Err(e) => {
                *failures += 1;
                println!(
                    "  [fail] vector store           {} ({e})",
                    config.ruvector_path().display()
                );
            }
        }
    }

    // ── embedder round-trip (dimension match) ────────────────────────────
    #[allow(deprecated)]
    let provider = config.llm.provider.clone();
    #[allow(deprecated)]
    let base = config.llm.base_url.clone();
    match hippocampus::Embedder::from_config(&provider, &base, &config.embedding.model, "") {
        Ok(Some(embedder)) => match embedder.embed("brain doctor probe").await {
            Ok(vec) if vec.len() == dim => {
                println!("  [ok]   embedder round-trip    {} dims", vec.len())
            }
            Ok(vec) => {
                *failures += 1;
                println!(
                    "  [fail] embedder round-trip    got {} dims, config expects {dim}",
                    vec.len()
                );
            }
            Err(e) => {
                *failures += 1;
                println!("  [fail] embedder round-trip    {e}");
            }
        },
        Ok(None) => println!("  [--]   embedder               not configured"),
        Err(e) => {
            *failures += 1;
            println!("  [fail] embedder init          {e}");
        }
    }
}

/// Verify the audit log's `prev_hash` linkage is contiguous: each row's
/// `prev_hash` must equal the previous row's `hash` (ordered by id). Detects
/// deletions / reordering without recomputing hashes. Returns the row count.
fn check_audit_chain(pool: &storage::SqlitePool) -> anyhow::Result<i64> {
    pool.with_conn(|conn| {
        let mut stmt = conn.prepare("SELECT prev_hash, hash FROM audit_log ORDER BY id ASC")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, Option<String>>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut prev: Option<String> = None;
        for (i, (prev_hash, hash)) in rows.iter().enumerate() {
            if let Some(expected) = &prev {
                if prev_hash.as_deref() != Some(expected.as_str()) {
                    return Err(storage::sqlite::SqliteError::Migration(format!(
                        "broken linkage at row {i}: prev_hash does not match prior row's hash"
                    )));
                }
            }
            prev = Some(hash.clone());
        }
        Ok(rows.len() as i64)
    })
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Episode / fact / graph-node counts straight from SQLite.
fn memory_counts(pool: &storage::SqlitePool) -> anyhow::Result<(i64, i64, i64)> {
    pool.with_conn(|conn| {
        let episodes: i64 = conn.query_row("SELECT COUNT(*) FROM episodes", [], |r| r.get(0))?;
        let facts: i64 = conn.query_row("SELECT COUNT(*) FROM semantic_facts", [], |r| r.get(0))?;
        let nodes: i64 = conn.query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get(0))?;
        Ok((episodes, facts, nodes))
    })
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Check one provider entry. Returns `true` when the entry is fully healthy
/// (reachable and, where applicable, the named model is available).
async fn check_provider(entry: &ProviderEntry) -> bool {
    let tag = &entry.name;
    match classify(entry) {
        ProviderKind::Ollama { base_url } => match list_ollama_models(&base_url).await {
            Some(installed) => {
                if model_present(&installed, &entry.model) {
                    println!(
                        "  [ok]   provider `{tag}` ollama @ {base_url} — model `{}` installed",
                        entry.model
                    );
                    true
                } else {
                    println!(
                        "  [warn] provider `{tag}` ollama @ {base_url} reachable, but model `{}` not pulled (run `ollama pull {}`)",
                        entry.model, entry.model
                    );
                    false
                }
            }
            None => {
                println!(
                    "  [warn] provider `{tag}` ollama @ {base_url} unreachable (run `ollama serve` or remove this entry)"
                );
                false
            }
        },
        ProviderKind::OpenAiCompat { base_url } => {
            match list_openai_compat_models(&base_url, &entry.api_key).await {
                Some(ids) if model_present(&ids, &entry.model) => {
                    println!(
                        "  [ok]   provider `{tag}` ({}) @ {base_url} — model `{}` available",
                        entry.kind, entry.model
                    );
                    true
                }
                Some(_) => {
                    println!(
                        "  [ok]   provider `{tag}` ({}) @ {base_url} reachable — model `{}` not in /models listing; will be tried on first call",
                        entry.kind, entry.model
                    );
                    true
                }
                None => {
                    let hint = if entry.api_key.trim().is_empty() {
                        " (no api_key set)"
                    } else {
                        ""
                    };
                    println!(
                        "  [warn] provider `{tag}` ({}) @ {base_url} unreachable or unauthorized{hint}",
                        entry.kind
                    );
                    false
                }
            }
        }
        ProviderKind::Unknown => {
            println!(
                "  [warn] provider `{tag}` has unknown kind `{}` (use ollama, openai_compat, or a preset: openai/openrouter/groq/deepseek/together/gemini-compat/nvidia)",
                entry.kind
            );
            false
        }
    }
}

async fn port_in_use(host: &str, port: u16) -> bool {
    tokio::net::TcpStream::connect((host, port)).await.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> storage::SqlitePool {
        storage::SqlitePool::open_memory().expect("memory pool")
    }

    fn insert_audit(pool: &storage::SqlitePool, prev: Option<&str>, hash: &str) {
        pool.with_conn(|conn| {
            conn.execute(
                "INSERT INTO audit_log (action, prev_hash, hash) VALUES ('test', ?1, ?2)",
                rusqlite::params![prev, hash],
            )?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn audit_chain_passes_on_contiguous_linkage() {
        let p = pool();
        insert_audit(&p, None, "h1");
        insert_audit(&p, Some("h1"), "h2");
        insert_audit(&p, Some("h2"), "h3");
        assert_eq!(check_audit_chain(&p).unwrap(), 3);
    }

    #[test]
    fn audit_chain_fails_on_broken_linkage() {
        let p = pool();
        insert_audit(&p, None, "h1");
        // prev_hash should be "h1" but points at a wrong value.
        insert_audit(&p, Some("WRONG"), "h2");
        assert!(check_audit_chain(&p).is_err());
    }

    #[test]
    fn audit_chain_ok_on_empty_log() {
        assert_eq!(check_audit_chain(&pool()).unwrap(), 0);
    }

    #[test]
    fn memory_counts_zero_on_fresh_db() {
        assert_eq!(memory_counts(&pool()).unwrap(), (0, 0, 0));
    }
}
