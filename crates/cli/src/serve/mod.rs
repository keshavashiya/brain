//! `brain serve` orchestration.
//!
//! `cmd_serve` is the foreground server entry point: it builds the wired
//! [`signal::SignalProcessor`], binds the requested adapter ports, wires
//! preset transports + channel relays, and parks every background
//! maintenance loop on a shared [`tokio::task::JoinSet`]. A SIGTERM /
//! Ctrl+C / fatal adapter exit all trigger graceful shutdown via
//! `set.abort_all()` + `processor.shutdown()`.
//!
//! Sibling modules own one responsibility each (issue 109, Wave C
//! split, mirrors the `signal::pipeline/` cut from issue 108):
//!
//! - [`adapters`] — per-adapter `try_bind_*` helpers (HTTP / WS / gRPC /
//!   MCP / Terminal)
//! - [`background`] — long-running maintenance loops (proactivity,
//!   open-loop detector, scheduled-intent poller, consolidator, graph
//!   compactor) + `promote_candidates`
//! - [`dlq`] — DLQ drain task + private batch helper
//! - [`reflex`] — reactive signal sources (FS / cron / sys)
//! - [`transports`] — preset transport wiring + channel relays

mod adapters;
mod background;
mod dlq;
mod reflex;
mod transports;

use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_serve(
    config: &brain::BrainConfig,
    http: bool,
    ws: bool,
    grpc: bool,
    mcp: bool,
    terminal: bool,
    host: String,
) -> anyhow::Result<()> {
    // Validate config before starting
    match config.validate() {
        Err(hard_err) => anyhow::bail!("Configuration error: {}", hard_err),
        Ok(warnings) => {
            for w in &warnings {
                tracing::warn!(warning = %w, "config warning");
                eprintln!("WARNING: {w}");
            }
        }
    }

    let run_all = !http && !ws && !grpc && !mcp && !terminal;

    // Issue 38: `adapters.<x>.enabled` is a hard gate applied AFTER the
    // CLI flag. `brain serve --http` with `adapters.http.enabled: false`
    // in config skips HTTP (with a loud warning) rather than starting
    // it anyway. `run_all` honours the same gate silently — operators
    // who disabled an adapter in YAML want it off, not on by default.
    let want_http = (run_all || http) && config.adapters.http.enabled;
    let want_ws = (run_all || ws) && config.adapters.ws.enabled;
    #[cfg(feature = "grpc")]
    let want_grpc = (run_all || grpc) && config.adapters.grpc.enabled;
    let want_mcp = (run_all || mcp) && config.adapters.mcp.enabled;
    for (flag, enabled, name) in [
        (http, config.adapters.http.enabled, "http"),
        (ws, config.adapters.ws.enabled, "ws"),
        (grpc, config.adapters.grpc.enabled, "grpc"),
        (mcp, config.adapters.mcp.enabled, "mcp"),
    ] {
        if flag && !enabled {
            tracing::warn!(
                adapter = name,
                "--{name} requested but adapters.{name}.enabled = false in config — skipping"
            );
            eprintln!(
                "WARNING: --{name} requested but adapters.{name}.enabled = false in config — skipping"
            );
        }
    }

    // Build the fully-wired processor via the shared bootstrap path.
    let mut processor = crate::bootstrap::build_processor(config).await?;

    // Wire the notification router (serve-specific — needed for proactive delivery)
    {
        let db = processor.episodic().pool().clone();
        let delivery_config = config.proactivity.delivery.clone();
        let mut router = signal::notification::NotificationRouter::new(db, delivery_config);

        if config.actions.messaging.enabled && !config.actions.messaging.channels.is_empty() {
            let res = &config.actions.resilience;
            match backends::WebhookMessageBackend::new_with_metrics(
                &config.actions.messaging.channels,
                config.actions.messaging.timeout_ms,
                res,
                Some(processor.metrics().clone()),
            ) {
                Ok(sender) => {
                    router = router.with_webhook_sender(Box::new(sender));
                    tracing::info!("Notification webhook sender attached");
                }
                Err(e) => {
                    tracing::warn!("Failed to init notification webhook sender: {e}");
                }
            }
        }

        processor = processor.with_notification_router(router);
    }

    let processor = Arc::new(processor);

    println!("Waking Brain OS...");

    let mut set = tokio::task::JoinSet::new();

    // ── Reactive signal sources ──────────────────────────────────────
    // Subscribe to every configured reflex source and bridge each
    // firing into the pipeline via `signal::reflex_runner::spawn_reflex`. All reflex
    // tasks live in the same JoinSet as the adapters so cmd_serve
    // shutdown cleans them up together. Reflex-origin signals carry
    // `Provenance::Reflex { trigger, … }` so audit / recall can
    // distinguish them from user-typed input.
    reflex::wire_reflex_sources(&config.reflex, &processor, &mut set).await;
    let mut bound_adapters = Vec::new();
    let mut failed_adapters = Vec::new();

    // ── Preset-driven transports ──────────────────────────────────────
    // Hold shutdown senders alive for the lifetime of cmd_serve so the
    // per-transport run loops don't see an immediate shutdown on drop.
    let (_transport_shutdown_guards, webhook_handlers) = if config.channel.transports.is_empty() {
        (Vec::new(), std::collections::HashMap::new())
    } else {
        let dispatcher = processor.channel_dispatcher().cloned();
        let correlator = processor.confirmation_correlator().cloned();
        match (dispatcher, correlator) {
            (Some(dispatcher), Some(correlator)) => {
                transports::wire_preset_transports(
                    &config.channel.transports,
                    processor.clone(),
                    dispatcher,
                    correlator,
                    &mut set,
                )
                .await
            }
            _ => {
                tracing::warn!(
                    "channel.transports configured but channel intelligence is not wired — skipping"
                );
                (Vec::new(), std::collections::HashMap::new())
            }
        }
    };

    // Bind adapters sequentially — HTTP is critical (health check endpoint).
    // If HTTP fails, abort entirely. Other adapters fail gracefully.
    if want_http {
        let port = config.adapters.http.port;
        match adapters::try_bind_http(
            processor.clone(),
            &host,
            port,
            webhook_handlers.clone(),
            &mut set,
        )
        .await
        {
            Ok(()) => bound_adapters.push("HTTP"),
            Err(e) => {
                failed_adapters.push(e);
                anyhow::bail!(
                    "Cannot bind HTTP synapse at {host}:{port} — another Brain daemon may already be running. \
                     This port is critical for health checks and inter-process communication. \
                     Try `brain stop` first."
                );
            }
        }
    }

    if want_ws {
        let port = config.adapters.ws.port;
        match adapters::try_bind_ws(processor.clone(), &host, port, &mut set).await {
            Ok(()) => bound_adapters.push("WebSocket"),
            Err(e) => {
                tracing::warn!(port, "WebSocket synapse bind failed");
                failed_adapters.push(e);
            }
        }
    }

    #[cfg(feature = "grpc")]
    if want_grpc {
        let port = config.adapters.grpc.port;
        match adapters::try_bind_grpc(processor.clone(), &host, port, &mut set).await {
            Ok(()) => bound_adapters.push("gRPC"),
            Err(e) => {
                tracing::warn!(port, "gRPC synapse bind failed");
                failed_adapters.push(e);
            }
        }
    }
    #[cfg(not(feature = "grpc"))]
    if grpc {
        eprintln!("WARNING: brain was built without the `grpc` feature — gRPC synapse disabled");
    }

    if want_mcp {
        let port = config.adapters.mcp.port;
        match adapters::try_bind_mcp(processor.clone(), &host, port, &mut set).await {
            Ok(()) => bound_adapters.push("MCP"),
            Err(e) => {
                tracing::warn!(port, "MCP synapse bind failed");
                failed_adapters.push(e);
            }
        }
    }

    // Terminal Bridge gRPC server. Requires `adapters.terminal.enabled`
    // (so config can disable it globally even with `--terminal`), a
    // wired bridge, and a wired identity store on the processor.
    if (run_all || terminal) && config.adapters.terminal.enabled {
        match (
            processor.terminal_bridge().cloned(),
            processor.identity_store().cloned(),
        ) {
            (Some(bridge), Some(identity_store)) => {
                let port = config.adapters.terminal.port;
                match adapters::try_bind_terminal(
                    bridge,
                    identity_store,
                    config.access.api_keys.clone(),
                    &host,
                    port,
                    &mut set,
                )
                .await
                {
                    Ok(()) => bound_adapters.push("Terminal"),
                    Err(e) => {
                        tracing::warn!(port, "Terminal synapse bind failed");
                        failed_adapters.push(e);
                    }
                }
            }
            (None, _) => {
                failed_adapters.push(
                    "Terminal: bridge not wired in bootstrap (build_processor regression)".into(),
                );
            }
            (Some(_), None) => {
                failed_adapters.push(
                    "Terminal: identity store not wired — refusing to expose unauthenticated bridge".into(),
                );
            }
        }
    }

    // Report partial startup failures
    if !failed_adapters.is_empty() {
        tracing::warn!(
            adapters = ?failed_adapters,
            "Some adapters failed to bind — continuing with partial setup"
        );
    }

    // ── Proactivity first-run notice ────────────────────────────────
    if config.proactivity.enabled {
        // Show a notice if user hasn't explicitly configured proactivity
        let user_config_path = brain::BrainConfig::user_config_path();
        let user_explicitly_set = std::fs::read_to_string(&user_config_path)
            .map(|s| s.contains("proactivity") && s.contains("enabled"))
            .unwrap_or(false);
        if !user_explicitly_set {
            tracing::info!(
                "Proactive notifications are enabled by default (max 2/day, quiet hours 20:00-10:00). \
                 Disable with: brain proactivity off"
            );
        }
    }

    // ── Background tasks ────────────────────────────────────────────
    #[cfg(feature = "ganglia")]
    if config.proactivity.enabled {
        background::spawn_habit_engine(processor.clone(), &config.proactivity, &mut set);
    }
    #[cfg(feature = "ganglia")]
    if config.proactivity.enabled && config.proactivity.open_loop.enabled {
        background::spawn_open_loop_detector(
            processor.clone(),
            config.proactivity.open_loop.clone(),
            &mut set,
        );
    }
    // Scheduled intents *fire* exclusively through the cron reflex now
    // (the historical direct-execution poller was retired in favour of
    // the reflex pipeline — see `reflex::CronReflex`). `actions.scheduling`
    // remains the *write* axis (create/persist intents); `reflex.cron`
    // is the *fire* axis. With scheduling enabled but the cron reflex
    // off, intents persist but never fire — warn so that's not silent.
    if config.actions.scheduling.enabled && !config.reflex.cron.enabled {
        tracing::warn!(
            "actions.scheduling is enabled but reflex.cron is disabled — scheduled \
             intents will persist but never fire. Set reflex.cron.enabled = true to \
             fire them through the pipeline."
        );
    }
    if config.memory.consolidation.enabled {
        background::spawn_consolidator(
            processor.clone(),
            config.memory.consolidation.interval_hours,
            config.memory.consolidation.forgetting_threshold,
            &mut set,
        );
    }
    background::spawn_graph_compactor(processor.clone(), &mut set);
    dlq::spawn_dlq_drain(processor.clone(), &mut set);

    // ── Channel relay adapters ────────────────────────────────────────
    if !config.channel.relays.is_empty() {
        let router = processor.channel_router().cloned();
        let correlator = processor.confirmation_correlator().cloned();
        let prefs = processor.channel_preferences().cloned();
        match (router, correlator, prefs) {
            (Some(router), Some(correlator), Some(prefs)) => {
                transports::wire_channel_relays(
                    &config.channel.relays,
                    &processor,
                    router,
                    correlator,
                    prefs,
                    &mut set,
                )
                .await;
            }
            _ => {
                tracing::warn!(
                    "channel.relays configured but channel intelligence is not wired — skipping"
                );
            }
        }
    }

    let _ = bound_adapters; // currently informational only

    println!("\nBrain is conscious. Press Ctrl+C to sleep.\n");

    // ── Graceful shutdown ─────────────────────────────────────────────
    #[cfg(unix)]
    let mut sigterm_listener = {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to register SIGTERM handler")
    };
    let sigterm_fut = async {
        #[cfg(unix)]
        {
            sigterm_listener.recv().await;
        }
        #[cfg(not(unix))]
        {
            std::future::pending::<()>().await;
        }
    };

    tokio::select! {
        result = set.join_next() => {
            if let Some(r) = result {
                match r {
                    Ok(Err(e)) => eprintln!("Adapter error: {e}"),
                    Err(e) => eprintln!("Task panicked: {e}"),
                    Ok(Ok(())) => {}
                }
            }
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received Ctrl+C — shutting down");
        }
        _ = sigterm_fut => {
            tracing::info!("Received SIGTERM — shutting down");
        }
    }

    set.abort_all();
    processor.shutdown();
    tracing::info!("Brain OS is asleep");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue 54 regression: `cmd_serve` must refuse to boot when
    /// `access.api_keys` is empty.
    #[tokio::test]
    async fn cmd_serve_refuses_empty_api_keys() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.access.api_keys.clear();

        let err = cmd_serve(
            &config,
            false,
            false,
            false,
            false,
            false,
            "127.0.0.1".to_string(),
        )
        .await
        .expect_err("serve should bail when api_keys is empty");

        let msg = err.to_string();
        assert!(
            msg.contains("No API keys configured"),
            "expected the explicit error message, got: {msg}"
        );
    }
}
