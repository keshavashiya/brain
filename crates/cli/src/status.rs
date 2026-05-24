//! Status command — system health and diagnostics display.

use crate::bootstrap;
use crate::daemon::{is_process_running, read_pid};
use crate::encryption::resolve_llm_api_key;

pub(crate) async fn show_status(config: &brain::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Scan");
    println!("  DNA:          v{}", env!("CARGO_PKG_VERSION"));
    println!("  Cortex:       {}", config.data_dir().display());

    // Daemon state via HTTP health probe — the single source of truth.
    let daemon_running = bootstrap::detect_running_daemon(config).await;
    match (daemon_running.as_ref(), read_pid(config)) {
        (Some(_), _) => {
            let pid = read_pid(config)
                .map(|p| format!(" (PID {p})"))
                .unwrap_or_default();
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

    // Resolve the actual provider once (same logic as daemon boot) so both
    // the display label and the health check reflect the selected entry.
    let llm_api_key = resolve_llm_api_key(config)?;
    let mut llm_cfg = config.llm.clone();
    if llm_cfg.providers.is_empty() {
        llm_cfg.api_key = llm_api_key;
    }
    let resolved_provider = cortex::llm::select_provider(&llm_cfg).await.ok();
    // Fallback display when no live provider is reachable. `llm.provider`
    // is #[deprecated] (Issue 40) but still the legitimate single-shape
    // source when `providers[]` is empty.
    #[allow(deprecated)]
    let legacy_provider = config.llm.provider.clone();
    let (display_model, display_kind) = resolved_provider
        .as_ref()
        .map(|p| (p.model().to_string(), p.name().to_string()))
        .unwrap_or_else(|| (config.llm.model.clone(), legacy_provider));
    println!("  Cortex LLM:   {} ({})", display_model, display_kind);
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
        brain::BrainConfig::user_config_path().display()
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

    let llm_healthy = match resolved_provider.as_ref() {
        Some(provider) => provider.health_check().await,
        None => false,
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
            .timeout(brain::timeouts::STATUS_CHECK)
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
        let req = client
            .get(&stats_url)
            .header("Authorization", format!("Bearer {}", api_key));
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(namespaces) = json.get("namespaces") {
                        println!("\n  Memory Regions:");
                        for (ns, info) in namespaces.as_object().unwrap_or(&serde_json::Map::new())
                        {
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

    // External service health — only probe SearXNG when the user has
    // explicitly selected it as the provider. The default DuckDuckGo
    // backend has no external service to probe, so we'd otherwise report
    // "SearXNG: stopped" on every default install and create the
    // impression of a missing dependency.
    let searxng_ep = config
        .actions
        .web_search
        .endpoint
        .trim()
        .trim_end_matches('/');
    let probes_searxng = matches!(
        config.actions.web_search.provider,
        brain::config::WebSearchProvider::Searxng
    );
    if probes_searxng && !searxng_ep.is_empty() {
        println!("\n  External Services:");
        let client = reqwest::Client::builder()
            .timeout(brain::timeouts::STATUS_CHECK)
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
