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

/// `brain doctor` — full environment check.
pub(crate) async fn cmd_doctor(config: &BrainConfig) -> anyhow::Result<()> {
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
