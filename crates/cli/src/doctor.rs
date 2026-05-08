//! Health checks for the local environment — `brain doctor` plus the
//! lightweight Ollama probe shown at the end of `brain init`.

use brain_core::BrainConfig;

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

/// Resolve the LLM endpoint actually used at runtime — falls back to the
/// legacy `llm.base_url` only when `providers[]` is empty.
fn effective_llm_base_url(config: &BrainConfig) -> String {
    if let Some(p) = config.llm.providers.first() {
        if !p.base_url.trim().is_empty() {
            return p.base_url.clone();
        }
    }
    config.llm.base_url.clone()
}

fn effective_llm_model(config: &BrainConfig) -> String {
    if let Some(p) = config.llm.providers.first() {
        if !p.model.trim().is_empty() {
            return p.model.clone();
        }
    }
    config.llm.model.clone()
}

/// Light probe used at the end of `brain init`. Prints a one-liner per model.
/// Never fails — this is a hint, not a gate.
pub(crate) async fn check_ollama_models(config: &BrainConfig) {
    let base = effective_llm_base_url(config);
    let llm_model = effective_llm_model(config);
    let embed_model = config.embedding.model.clone();

    println!();
    match list_ollama_models(&base).await {
        Some(installed) => {
            print_model_status("Cortex LLM", &llm_model, &installed);
            print_model_status("Sensory   ", &embed_model, &installed);
        }
        None => {
            println!("  Ollama:    unreachable at {base}");
            println!("            install from https://ollama.com, then run:");
            println!("            ollama pull {llm_model}");
            println!("            ollama pull {embed_model}");
        }
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

    // ── ollama reachable + models present ────────────────────────────────
    let base = effective_llm_base_url(config);
    match list_ollama_models(&base).await {
        Some(installed) => {
            println!("  [ok]   ollama reachable       {}", base);
            let llm_model = effective_llm_model(config);
            if model_present(&installed, &llm_model) {
                println!("  [ok]   LLM model installed    {}", llm_model);
            } else {
                failures += 1;
                println!(
                    "  [fail] LLM model missing      {} (run `ollama pull {}`)",
                    llm_model, llm_model
                );
            }
            if model_present(&installed, &config.embedding.model) {
                println!("  [ok]   embedding installed    {}", config.embedding.model);
            } else {
                failures += 1;
                println!(
                    "  [fail] embedding missing      {} (run `ollama pull {}`)",
                    config.embedding.model, config.embedding.model
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

async fn port_in_use(host: &str, port: u16) -> bool {
    tokio::net::TcpStream::connect((host, port)).await.is_ok()
}
