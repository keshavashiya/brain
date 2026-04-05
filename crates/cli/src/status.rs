//! Status command — system health and diagnostics display.

use crate::daemon::{is_process_running, read_pid};
use crate::encryption::resolve_llm_api_key;

pub(crate) async fn show_status(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Scan");
    println!("  DNA:          v{}", env!("CARGO_PKG_VERSION"));
    println!("  Cortex:       {}", config.data_dir().display());

    // Daemon state
    match read_pid(config) {
        Some(pid) if is_process_running(pid) => {
            println!("  State:        awake (PID {})", pid);
        }
        Some(_) => {
            println!("  State:        asleep (stale PID file)");
        }
        None => {
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
    let provider = cortex::llm::create_provider(&llm_cfg);
    let llm_healthy = provider.health_check().await;
    println!(
        "  Cortex:       {}",
        if llm_healthy {
            "responsive"
        } else {
            "unresponsive"
        }
    );

    // Database stats
    match storage::SqlitePool::open(&config.sqlite_path()) {
        Ok(pool) => match pool.table_stats() {
            Ok(stats) => {
                println!("\n  Memory Regions:");
                for (table, count) in stats {
                    println!("    {}: {} rows", table, count);
                }
            }
            Err(e) => println!("\n  Hippocampus: error reading stats — {}", e),
        },
        Err(e) => println!("\n  Hippocampus: error opening — {}", e),
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
