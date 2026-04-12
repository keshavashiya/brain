//! Status command — system health and diagnostics display.

use crate::bootstrap;
use crate::daemon::{is_process_running, read_pid};
use crate::encryption::resolve_llm_api_key;

pub(crate) async fn show_status(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Scan");
    println!("  DNA:          v{}", env!("CARGO_PKG_VERSION"));
    println!("  Cortex:       {}", config.data_dir().display());

    // Daemon state via HTTP health probe — the single source of truth.
    let daemon_running = bootstrap::detect_running_daemon(config).await;
    match (daemon_running.as_ref(), read_pid(config)) {
        (Some(_), _) => {
            let pid = read_pid(config).map(|p| format!(" (PID {p})")).unwrap_or_default();
            println!("  State:        awake{}", pid);
        }
        (None, Some(pid)) if is_process_running(pid) => {
            // Process running but HTTP not responding — zombie state
            println!("  State:        zombie (PID {pid}, not responding)");
        }
        (None, Some(_)) => {
            println!("  State:        asleep (stale PID file)");
        }
        (None, None) => {
            println!("  State:        asleep (run `brain start` to wake)");
        }
    }

    println!(
        "  Cortex LLM:   {} ({})",
        config.llm.model, config.llm.provider
    );
    println!(
        "  Sensory:      {} ({}d)",
        config.embedding.model, config.embedding.dimensions
    );
    println!(
        "  Barrier:      {}",
        if config.encryption.enabled {
            "sealed"
        } else {
            "open"
        }
    );
    println!("  Hippocampus:  {}", config.sqlite_path().display());
    println!("  Neural mesh:  {}", config.ruvector_path().display());
    println!(
        "  Genome:       {}",
        brain_core::BrainConfig::user_config_path().display()
    );

    println!("\n  Synapses:");
    let h = &config.adapters.http;
    println!(
        "    HTTP      : port {} ({})",
        h.port,
        if h.enabled { "active" } else { "dormant" }
    );
    let w = &config.adapters.ws;
    println!(
        "    WebSocket : port {} ({})",
        w.port,
        if w.enabled { "active" } else { "dormant" }
    );
    let m = &config.adapters.mcp;
    println!(
        "    MCP       : port {} ({})",
        m.port,
        if m.enabled { "active" } else { "dormant" }
    );
    let g = &config.adapters.grpc;
    println!(
        "    gRPC      : port {} ({})",
        g.port,
        if g.enabled { "active" } else { "dormant" }
    );

    // LLM health
    let llm_api_key = resolve_llm_api_key(config);
    let llm_cfg = cortex::llm::ProviderConfig {
        provider: config.llm.provider.clone(),
        base_url: config.llm.base_url.clone(),
        api_key: if llm_api_key.is_empty() {
            None
        } else {
            Some(llm_api_key)
        },
        model: config.llm.model.clone(),
        temperature: config.llm.temperature,
        max_tokens: config.llm.max_tokens as i32,
    };
    let llm_healthy = match cortex::llm::create_provider(&llm_cfg) {
        Ok(provider) => provider.health_check().await,
        Err(e) => {
            tracing::warn!("Failed to create LLM provider for health check: {e}");
            false
        }
    };
    println!(
        "  Cortex:       {}",
        if llm_healthy {
            "responsive"
        } else {
            "unresponsive"
        }
    );

    // Database stats — only query via HTTP when daemon is running.
    // Opening SQLite directly causes RuVector lock contention.
    if let Some(base_url) = daemon_running {
        let client = reqwest::Client::builder()
            .timeout(brain_core::timeouts::STATUS_CHECK)
            .build()
            .unwrap_or_default();
        let api_key = config
            .access
            .api_keys
            .first()
            .map(|k| k.key.clone())
            .unwrap_or_default();

        // Fetch memory stats via HTTP API
        let stats_url = format!("{}/v1/memory/namespaces", base_url);
        let req = client.get(&stats_url).header("Authorization", format!("Bearer {}", api_key));
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(namespaces) = json.get("namespaces") {
                        println!("\n  Memory Regions:");
                        for (ns, info) in namespaces.as_object().unwrap_or(&serde_json::Map::new()) {
                            if let Some(episodes) = info.get("episodes").and_then(|v| v.as_u64()) {
                                println!("    {}: {} episodes", ns, episodes);
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Daemon not running — report from config only, don't open files
        println!("\n  Memory Regions:");
        println!("    (daemon not running — start with `brain start` to view)");
    }

    // External service health
    let searxng_ep = config
        .actions
        .web_search
        .endpoint
        .trim()
        .trim_end_matches('/');
    if !searxng_ep.is_empty() {
        println!("\n  External Services:");
        let client = reqwest::Client::builder()
            .timeout(brain_core::timeouts::STATUS_CHECK)
            .build()
            .unwrap_or_default();
        let health_url = format!("{}/healthz", searxng_ep);
        let healthy = client
            .get(&health_url)
            .send()
            .await
            .is_ok_and(|r| r.status().is_success());
        println!(
            "    {:<10}: {} ({})",
            "SearXNG",
            if healthy { "running" } else { "stopped" },
            searxng_ep
        );
    }

    Ok(())
}
