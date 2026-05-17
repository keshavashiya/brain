//! Serve command — run all services in the foreground.

use std::sync::Arc;

use async_trait::async_trait;
use backends::*;
use bridge::BridgeMessage;
use channel::relay::SignalHandler;

/// Bridge-facing handler that forwards non-correlation messages into the
/// main signal pipeline. Used by each configured relay adapter so replies
/// arriving on any external transport hit the same pipeline as HTTP/WS
/// traffic.
struct RelayPipelineHandler {
    processor: Arc<signal::SignalProcessor>,
    channel_id: String,
    namespace: String,
}

#[async_trait]
impl SignalHandler for RelayPipelineHandler {
    async fn handle(&self, msg: &BridgeMessage) -> String {
        let sender = msg.source.clone().unwrap_or_else(|| "relay".to_string());
        let sig = signal::Signal::new(
            signal::SignalSource::WebSocket,
            &self.channel_id,
            sender,
            &msg.content,
        )
        .with_namespace(&self.namespace);
        match self.processor.process(sig).await {
            Ok(resp) => signal::response_to_text(&resp.response),
            Err(e) => format!("error: {e}"),
        }
    }
}

pub(crate) async fn cmd_serve(
    config: &brain_core::BrainConfig,
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

    // Build the fully-wired processor via the shared bootstrap path.
    let mut processor = crate::bootstrap::build_processor(config).await?;

    // Wire the notification router (serve-specific — needed for proactive delivery)
    {
        let db = processor.episodic().pool().clone();
        let delivery_config = config.proactivity.delivery.clone();
        let mut router = signal::notification::NotificationRouter::new(db, delivery_config);

        if config.actions.messaging.enabled && !config.actions.messaging.channels.is_empty() {
            let res = &config.actions.resilience;
            match WebhookMessageBackend::new_with_metrics(
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
    wire_reflex_sources(&config.reflex, &processor, &mut set).await;
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
                wire_preset_transports(
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
    if run_all || http {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.http.port;
        let handlers = webhook_handlers.clone();
        match tokio::net::TcpListener::bind(format!("{h}:{port}")).await {
            Ok(_listener) => {
                println!("  Synapse HTTP  → http://{}:{}", h, port);
                let p = p.clone();
                let h = h.clone();
                set.spawn(async move { httpadapter::serve(p, handlers, &h, port).await });
                bound_adapters.push("HTTP");
            }
            Err(e) => {
                failed_adapters.push(format!("HTTP ({h}:{port}): {e}"));
                anyhow::bail!(
                    "Cannot bind HTTP synapse at {h}:{port} — {e}\n\
                     This port is critical for health checks and inter-process communication.\n\
                     Another Brain daemon may already be running. Try `brain stop` first."
                );
            }
        }
    }

    if run_all || ws {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.ws.port;
        match tokio::net::TcpListener::bind(format!("{h}:{port}")).await {
            Ok(_listener) => {
                println!("  Synapse WS    → ws://{}:{}", h, port);
                let p = p.clone();
                let h = h.clone();
                set.spawn(async move { wsadapter::serve(p, &h, port).await });
                bound_adapters.push("WebSocket");
            }
            Err(e) => {
                failed_adapters.push(format!("WS ({h}:{port}): {e}"));
                tracing::warn!(port, "WebSocket synapse bind failed: {e}");
            }
        }
    }

    #[cfg(feature = "grpc")]
    if run_all || grpc {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.grpc.port;
        match tokio::net::TcpListener::bind(format!("{h}:{port}")).await {
            Ok(_listener) => {
                println!("  Synapse gRPC  → {}:{}", h, port);
                let p = p.clone();
                let h = h.clone();
                set.spawn(async move { grpcadapter::serve(p, &h, port).await });
                bound_adapters.push("gRPC");
            }
            Err(e) => {
                failed_adapters.push(format!("gRPC ({h}:{port}): {e}"));
                tracing::warn!(port, "gRPC synapse bind failed: {e}");
            }
        }
    }
    #[cfg(not(feature = "grpc"))]
    if grpc {
        eprintln!("WARNING: brain was built without the `grpc` feature — gRPC synapse disabled");
    }

    if run_all || mcp {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.mcp.port;
        match tokio::net::TcpListener::bind(format!("{h}:{port}")).await {
            Ok(_listener) => {
                println!("  Synapse MCP   → http://{}:{}", h, port);
                let p = p.clone();
                let h = h.clone();
                set.spawn(async move { mcp::serve_http(p, &h, port).await });
                bound_adapters.push("MCP");
            }
            Err(e) => {
                failed_adapters.push(format!("MCP ({h}:{port}): {e}"));
                tracing::warn!(port, "MCP synapse bind failed: {e}");
            }
        }
    }

    // Terminal Bridge gRPC server — exposes the wired
    // `processor.terminal_bridge()` over the wire. Requires
    // `adapters.terminal.enabled` (so config can disable it globally
    // even with `--terminal`) and a wired bridge on the processor.
    if (run_all || terminal) && config.adapters.terminal.enabled {
        match (
            processor.terminal_bridge().cloned(),
            processor.identity_store().cloned(),
        ) {
            (Some(bridge), Some(identity_store)) => {
                let h = host.clone();
                let port = config.adapters.terminal.port;
                match tokio::net::TcpListener::bind(format!("{h}:{port}")).await {
                    Ok(_listener) => {
                        println!("  Synapse Term  → {}:{}", h, port);
                        // Pair the wired identity store with this adapter's
                        // api-key clone at spawn time — same shape as the
                        // HTTP/WS/gRPC/MCP serve sites.
                        let auth = terminal::TerminalAuth::new(
                            identity_store,
                            config.access.api_keys.clone(),
                        );
                        let bridge_for_spawn = bridge.as_ref().clone().with_auth(auth);
                        let h = h.clone();
                        set.spawn(async move {
                            let addr = match format!("{h}:{port}").parse() {
                                Ok(a) => a,
                                Err(e) => {
                                    return Err(anyhow::anyhow!(
                                        "Terminal Bridge: invalid bind addr {h}:{port}: {e}"
                                    ));
                                }
                            };
                            tonic::transport::Server::builder()
                                .add_service(bridge_for_spawn.into_server())
                                .serve(addr)
                                .await
                                .map_err(|e| anyhow::anyhow!("Terminal Bridge serve: {e}"))
                        });
                        bound_adapters.push("Terminal");
                    }
                    Err(e) => {
                        failed_adapters.push(format!("Terminal ({h}:{port}): {e}"));
                        tracing::warn!(port, "Terminal synapse bind failed: {e}");
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
        let user_config_path = brain_core::BrainConfig::user_config_path();
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

    // ── Proactivity / habit engine background task ────────────────────
    #[cfg(feature = "ganglia")]
    if config.proactivity.enabled {
        let p = processor.clone();
        let habit_cfg = ganglia::HabitConfig {
            max_per_day: config.proactivity.max_per_day,
            min_interval_minutes: config.proactivity.min_interval_minutes,
            quiet_start: config.proactivity.quiet_hours.start.clone(),
            quiet_end: config.proactivity.quiet_hours.end.clone(),
            ..Default::default()
        };
        set.spawn(async move {
            let engine = ganglia::HabitEngine::new(p.episodic().pool().clone(), habit_cfg.clone());
            if let Err(e) = engine.ensure_tables() {
                tracing::warn!("HabitEngine table init failed: {e}");
                return Ok(());
            }
            let check_interval =
                tokio::time::Duration::from_secs(habit_cfg.min_interval_minutes as u64 * 60);
            let mut ticker = tokio::time::interval(check_interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                match engine.generate_proactive() {
                    Ok(Some(msg)) => {
                        tracing::info!(
                            triggered_by = %msg.triggered_by,
                            "Proactive: {}",
                            msg.content
                        );
                        if let Some(router) = p.notification_router() {
                            router.deliver(msg.into()).await;
                        }
                    }
                    Ok(None) => {}
                    Err(e) => tracing::warn!("HabitEngine error: {e}"),
                }
            }
        });
        tracing::info!(
            interval_minutes = config.proactivity.min_interval_minutes,
            "Proactivity engine scheduled"
        );
    }

    // ── Open-loop detection background task ───────────────────────────
    #[cfg(feature = "ganglia")]
    if config.proactivity.enabled && config.proactivity.open_loop.enabled {
        let p = processor.clone();
        let ol_cfg = config.proactivity.open_loop.clone();
        set.spawn(async move {
            let detector = ganglia::OpenLoopDetector::with_llm(
                p.episodic().pool().clone(),
                ganglia::OpenLoopConfig {
                    scan_window_hours: ol_cfg.scan_window_hours,
                    resolution_window_hours: ol_cfg.resolution_window_hours,
                    max_reminders: 3,
                },
                p.llm().clone(),
            );
            let check_interval =
                tokio::time::Duration::from_secs(ol_cfg.check_interval_minutes as u64 * 60);
            let mut ticker = tokio::time::interval(check_interval);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                match detector.generate_reminders_async().await {
                    Ok(reminders) if !reminders.is_empty() => {
                        if let Some(router) = p.notification_router() {
                            for msg in reminders {
                                tracing::info!(
                                    triggered_by = %msg.triggered_by,
                                    "Open loop: {}",
                                    msg.content
                                );
                                router.deliver(msg.into()).await;
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!("OpenLoopDetector error: {e}"),
                }
            }
        });
        tracing::info!(
            interval_minutes = config.proactivity.open_loop.check_interval_minutes,
            "Open-loop detector scheduled"
        );
    }

    // ── Scheduled intent poller ────────────────────────────────────────
    if config.actions.scheduling.enabled {
        let p = processor.clone();
        set.spawn(async move {
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(60));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let db = p.episodic().pool();
                let due = match db.due_scheduled_intents() {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("Scheduled intent poll failed: {e}");
                        continue;
                    }
                };
                for intent in due {
                    tracing::info!(
                        id = %intent.id,
                        description = %intent.description,
                        "Firing scheduled intent"
                    );
                    if let Some(router) = p.notification_router() {
                        let notif = signal::notification::ProactiveNotification {
                            content: format!("[scheduled] {}", intent.description),
                            triggered_by: "scheduler".to_string(),
                            priority: 1,
                            agent: None,
                        };
                        router.deliver(notif).await;
                    }
                    let _ = db.update_scheduled_intent_status(&intent.id, "fired");
                }
            }
        });
        tracing::info!("Scheduled intent poller started (every 60s)");
    }

    // ── Memory consolidation background task ──────────────────────────
    if config.memory.consolidation.enabled {
        let p = processor.clone();
        let interval_hours = config.memory.consolidation.interval_hours;
        let prune_threshold = config.memory.consolidation.forgetting_threshold;
        set.spawn(async move {
            let consolidator = hippocampus::Consolidator::new(hippocampus::ConsolidationConfig {
                prune_threshold,
                ..Default::default()
            });
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(
                interval_hours as u64 * 3600,
            ));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                match consolidator.consolidate(p.episodic()) {
                    Ok(r) => {
                        let promoted_now =
                            promote_candidates(p.as_ref(), &r.promotion_candidates).await;

                        if let Some(router) = p.notification_router() {
                            router.prune();
                        }

                        let metrics = p.metrics();
                        metrics.inc_consolidation_run();
                        metrics.add_consolidation_pruned(r.episodes_pruned as u64);
                        metrics.add_consolidation_promoted(promoted_now as u64);

                        tracing::info!(
                            pruned = r.episodes_pruned,
                            promotion_candidates = r.episodes_promoted,
                            promoted = promoted_now,
                            remaining = r.episodes_remaining,
                            "Memory consolidation complete"
                        );
                    }
                    Err(e) => tracing::warn!("Memory consolidation error: {e}"),
                }
            }
        });
        tracing::info!(interval_hours, "Memory consolidation scheduled");
    }

    // ── Episodic graph compactor ──────────────────────────────────────
    // 24h reflection cycle: half-life decays every node's weight and
    // prunes the ones that fall below the eviction cutoff. The graph
    // is opened lazily per tick against the episodic pool so we don't
    // hold a `SqliteGraph` across ticks — the pool is the shared
    // resource, the wrapper is cheap.
    {
        use hippocampus::Compactor as _;
        let p = processor.clone();
        set.spawn(async move {
            let compactor =
                hippocampus::DefaultCompactor::new(hippocampus::CompactConfig::default());
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(24 * 3600));
            // Skip the immediate first tick — `interval` fires once at
            // start; we want the first compaction 24h after boot.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let graph = hippocampus::SqliteGraph::new(p.episodic().pool().clone());
                match compactor.compact(&graph).await {
                    Ok(stats) => tracing::info!(
                        scanned = stats.scanned,
                        decayed = stats.decayed,
                        evicted = stats.evicted,
                        "Graph compactor cycle complete"
                    ),
                    Err(e) => tracing::warn!(error = %e, "Graph compactor cycle failed"),
                }
            }
        });
        tracing::info!("Graph compactor scheduled (every 24h, default half-life 7d)");
    }

    // ── Channel relay adapters ────────────────────────────────────────
    if !config.channel.relays.is_empty() {
        let router = processor.channel_router().cloned();
        let correlator = processor.confirmation_correlator().cloned();
        let prefs = processor.channel_preferences().cloned();
        match (router, correlator, prefs) {
            (Some(router), Some(correlator), Some(prefs)) => {
                for entry in &config.channel.relays {
                    let bridge_cfg = bridge::BridgeConfig {
                        initial_backoff_ms: entry.initial_backoff_ms,
                        max_backoff_ms: entry.max_backoff_ms,
                        max_reconnect_attempts: None,
                    };
                    let relay_cfg = channel::RelayConfig::new(&entry.id, &entry.label, &entry.url)
                        .with_namespace(&entry.namespace)
                        .with_bridge(bridge_cfg);
                    let fallback: Arc<dyn SignalHandler> = Arc::new(RelayPipelineHandler {
                        processor: processor.clone(),
                        channel_id: entry.id.clone(),
                        namespace: entry.namespace.clone(),
                    });
                    let adapter = Arc::new(channel::RelayAdapter::new(
                        relay_cfg,
                        router.clone(),
                        correlator.clone(),
                        prefs.clone(),
                        fallback,
                    ));
                    if let Err(e) = adapter.register_channel().await {
                        tracing::warn!(
                            channel = %entry.id,
                            "Relay channel registration failed (non-fatal): {e}"
                        );
                    } else {
                        println!("  Relay        → {} ({})", entry.label, entry.url);
                    }
                    let a = adapter.clone();
                    set.spawn(async move {
                        if let Err(e) = a.run().await {
                            tracing::warn!("Relay adapter exited: {e}");
                        }
                        Ok(())
                    });
                }
            }
            _ => {
                tracing::warn!(
                    "channel.relays configured but channel intelligence is not wired — skipping"
                );
            }
        }
    }

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

pub(crate) async fn promote_candidates(
    processor: &signal::SignalProcessor,
    candidates: &[hippocampus::PromotionCandidate],
) -> usize {
    let mut promoted_now = 0usize;

    for candidate in candidates {
        let already_promoted = processor
            .episodic()
            .pool()
            .with_conn(|conn| {
                let exists: i64 = conn.query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM episode_promotions
                        WHERE episode_id = ?1
                    )",
                    [&candidate.episode_id],
                    |row| row.get(0),
                )?;
                Ok(exists > 0)
            })
            .unwrap_or(false);

        if already_promoted {
            continue;
        }

        let subject = "user";
        let predicate = "said";
        let object = &candidate.content;

        if object.trim().is_empty() {
            continue;
        }

        match processor
            .store_fact_direct(
                &candidate.namespace,
                "consolidated",
                subject,
                predicate,
                object,
                None,
            )
            .await
        {
            Ok(fact_id) => {
                if let Err(e) = processor.episodic().pool().with_conn(|conn| {
                    conn.execute(
                        "INSERT INTO episode_promotions (episode_id, fact_id)
                         VALUES (?1, ?2)
                         ON CONFLICT(episode_id) DO NOTHING",
                        rusqlite::params![&candidate.episode_id, fact_id],
                    )?;
                    Ok(())
                }) {
                    tracing::warn!(
                        episode_id = %candidate.episode_id,
                        "Failed to persist promotion marker: {e}"
                    );
                } else {
                    promoted_now += 1;
                }
            }
            Err(e) => tracing::warn!(
                episode_id = %candidate.episode_id,
                "Failed to promote episode: {e}"
            ),
        }
    }

    promoted_now
}

/// Pipe an inbound transport message into the main signal pipeline with the
/// same shape the relay handler uses, then route the pipeline's response
/// back to the originating channel via the dispatcher. Without the
/// round-trip, replies to inbound Telegram/etc. messages would be silently
/// dropped (the request side has no live WS adapter to write back to).
async fn forward_inbound_to_signal(
    msg: channel::InboundMessage,
    processor: Arc<signal::SignalProcessor>,
    dispatcher: Arc<channel::ChannelDispatcher>,
    channel_id: String,
    namespace: String,
) {
    let sender = msg.user_ref.clone().unwrap_or_else(|| "transport".into());
    let reply_to = msg.reply_to.clone();
    let sig = signal::Signal::new(
        signal::SignalSource::WebSocket,
        &channel_id,
        sender,
        &msg.content,
    )
    .with_namespace(&namespace);
    let resp = match processor.process(sig).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(channel = %channel_id, error = %e, "Transport inbound pipeline error");
            return;
        }
    };

    let body = signal::response_to_text(&resp.response);
    if body.trim().is_empty() {
        return;
    }
    let mut intent = channel::DeliveryIntent::new(
        body,
        channel::DeliveryCategory::Response,
        channel::UrgencyLevel::Normal,
    )
    .with_namespace(&namespace)
    .with_preferred(&channel_id)
    .with_initiation(&channel_id);
    if let Some(rt) = reply_to {
        intent = intent.with_metadata("reply_to", rt);
    }
    if let Err(e) = dispatcher.dispatch(intent).await {
        tracing::warn!(channel = %channel_id, error = %e, "Failed to route inbound response back to source");
    }
}

/// Construct every configured `reflex::ReflexSource`, hook it into the
/// pipeline via `signal::reflex_runner::spawn_reflex`, and park the resulting task on
/// the shared join set. Each firing builds a `Signal` stamped with
/// `Provenance::Reflex { trigger, … }` so the pipeline can distinguish
/// reflex-driven activity from user-typed input without inspecting the
/// trigger string itself.
///
/// Failures during subscribe (e.g. an FS watcher can't bind to a
/// non-existent path) are logged at `warn` and do not abort serve —
/// other reflexes still spawn, mirroring how adapter bind failures
/// degrade gracefully.
async fn wire_reflex_sources(
    cfg: &brain_core::config::ReflexConfig,
    processor: &Arc<signal::SignalProcessor>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    use reflex::ReflexSource;

    fn reflex_signal(name: &str, ev: reflex::ReflexEvent) -> signal::Signal {
        let raw_input = serde_json::to_string(&ev.payload).ok();
        signal::Signal::new(
            signal::SignalSource::Cli,
            format!("reflex:{name}"),
            name.to_string(),
            ev.trigger.clone(),
        )
        .with_provenance(intent::Provenance::Reflex {
            trigger: ev.trigger,
            raw_input,
            ts: ev.ts,
        })
    }

    fn expand_tilde_path(p: &str) -> std::path::PathBuf {
        if let Some(rest) = p.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return std::path::PathBuf::from(home).join(rest);
            }
        }
        std::path::PathBuf::from(p)
    }

    // FS watchers — one spawn per entry so a bad path on one entry
    // doesn't take down the rest.
    for entry in &cfg.fs {
        let paths: Vec<std::path::PathBuf> =
            entry.paths.iter().map(|p| expand_tilde_path(p)).collect();
        let fs_cfg = reflex::FsReflexConfig::new(paths)
            .recursive(entry.recursive)
            .debounce(std::time::Duration::from_millis(entry.debounce_ms));
        let source: Arc<dyn ReflexSource> =
            Arc::new(reflex::FsReflex::new(entry.name.clone(), fs_cfg));
        let name_for_log = entry.name.clone();
        let name_for_builder = entry.name.clone();
        match signal::reflex_runner::spawn_reflex(
            entry.name.clone(),
            source,
            processor.clone(),
            move |ev| reflex_signal(&name_for_builder, ev),
        )
        .await
        {
            Ok(handle) => {
                tracing::info!(reflex = %name_for_log, "FS reflex spawned");
                set.spawn(async move {
                    let _ = handle.await;
                    Ok(())
                });
            }
            Err(e) => tracing::warn!(
                reflex = %name_for_log,
                error = %e,
                "FS reflex subscribe failed; skipping"
            ),
        }
    }

    // Cron — single reflex that polls scheduled_intents via the
    // episodic pool. Disabled by default; turning it on simply moves
    // the historical 60s scheduler ticker into the reflex stream.
    if cfg.cron.enabled {
        let mut cron_cfg = reflex::CronReflexConfig::new(std::time::Duration::from_secs(
            cfg.cron.poll_interval_seconds,
        ));
        if let Some(ns) = &cfg.cron.namespace_filter {
            cron_cfg = cron_cfg.namespace(ns.clone());
        }
        let pool = processor.episodic().pool().clone();
        let source: Arc<dyn ReflexSource> =
            Arc::new(reflex::CronReflex::new("cron", pool, cron_cfg));
        match signal::reflex_runner::spawn_reflex("cron", source, processor.clone(), move |ev| {
            reflex_signal("cron", ev)
        })
        .await
        {
            Ok(handle) => {
                tracing::info!("Cron reflex spawned");
                set.spawn(async move {
                    let _ = handle.await;
                    Ok(())
                });
            }
            Err(e) => tracing::warn!(error = %e, "cron reflex subscribe failed; skipping"),
        }
    }

    // SysState — uses a `NoopSampler` until per-platform sampler
    // implementations are wired. The reflex still spawns when
    // `enabled = true` so config can be validated end-to-end and
    // rule transitions surface as soon as a real sampler lands.
    if cfg.sys.enabled && !cfg.sys.rules.is_empty() {
        let rules: Vec<reflex::SysStateRule> = cfg
            .sys
            .rules
            .iter()
            .map(|r| match r {
                brain_core::config::SysReflexRuleEntry::BatteryBelow { threshold } => {
                    reflex::SysStateRule::BatteryBelow(*threshold)
                }
                brain_core::config::SysReflexRuleEntry::OnAcChanged => {
                    reflex::SysStateRule::OnAcChanged
                }
                brain_core::config::SysReflexRuleEntry::NetworkChanged => {
                    reflex::SysStateRule::NetworkChanged
                }
                brain_core::config::SysReflexRuleEntry::LockChanged => {
                    reflex::SysStateRule::LockChanged
                }
            })
            .collect();
        let sys_cfg = reflex::SysStateReflexConfig::new(std::time::Duration::from_secs(
            cfg.sys.poll_interval_seconds,
        ))
        .with_rules(rules);
        let sampler: Arc<dyn reflex::SysStateSampler> = Arc::new(reflex::NoopSampler);
        let source: Arc<dyn ReflexSource> =
            Arc::new(reflex::SysStateReflex::new("sys", sampler, sys_cfg));
        match signal::reflex_runner::spawn_reflex("sys", source, processor.clone(), move |ev| {
            reflex_signal("sys", ev)
        })
        .await
        {
            Ok(handle) => {
                tracing::info!("SysState reflex spawned (using NoopSampler)");
                set.spawn(async move {
                    let _ = handle.await;
                    Ok(())
                });
            }
            Err(e) => tracing::warn!(error = %e, "sys reflex subscribe failed; skipping"),
        }
    }
}

/// Build preset-driven transports, register each with the dispatcher (so
/// it can actually deliver), spawn polling loops for HttpPolled kinds,
/// and feed inbound messages into the signal pipeline (after the
/// correlator has had a chance to claim them).
async fn wire_preset_transports(
    entries: &[brain_core::config::TransportEntry],
    processor: Arc<signal::SignalProcessor>,
    dispatcher: Arc<channel::ChannelDispatcher>,
    correlator: Arc<channel::ConfirmationCorrelator>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> (
    Vec<tokio::sync::oneshot::Sender<()>>,
    std::collections::HashMap<String, Arc<channel::transport::inbound::WebhookInboundTransport>>,
) {
    use channel::transport::inbound::{WebhookInboundConfig, WebhookInboundTransport};
    use channel::transport::outbound::{WebhookOutboundConfig, WebhookOutboundTransport};
    use channel::transport::polled::{HttpPolledConfig, HttpPolledTransport};
    use channel::transport::preset::{self, PresetKind};
    use channel::ChannelTransport;

    let mut shutdown_guards = Vec::new();
    let mut webhook_handlers = std::collections::HashMap::new();

    for entry in entries {
        let preset = match preset::load(&entry.preset) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    transport = %entry.id,
                    preset = %entry.preset,
                    error = %e,
                    "Preset load failed — skipping transport"
                );
                continue;
            }
        };

        match preset.kind {
            PresetKind::HttpPolled => {
                let cfg = HttpPolledConfig::new(
                    &entry.id,
                    &entry.label,
                    preset.clone(),
                    &entry.credential,
                );
                let transport = match HttpPolledTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "HttpPolled init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, polled)",
                    entry.label, entry.preset
                );

                let rx = transport.inbound();
                let proc_clone = processor.clone();
                let disp_clone = dispatcher.clone();
                let chan_id = entry.id.clone();
                let ns = entry.namespace.clone();
                let corr = correlator.clone();
                set.spawn(async move {
                    let mut rx = rx;
                    while let Ok(msg) = rx.recv().await {
                        match corr.process(&msg.content).await {
                            Ok(channel::CorrelationOutcome::NoMatch) => {}
                            Ok(_) => continue,
                            Err(e) => {
                                tracing::debug!(error = %e, "correlator error — forwarding raw");
                            }
                        }
                        forward_inbound_to_signal(
                            msg,
                            proc_clone.clone(),
                            disp_clone.clone(),
                            chan_id.clone(),
                            ns.clone(),
                        )
                        .await;
                    }
                    Ok(())
                });

                let run_transport = transport.clone();
                let (sd_tx, sd_rx) = tokio::sync::oneshot::channel::<()>();
                shutdown_guards.push(sd_tx);
                set.spawn(async move {
                    run_transport.run(sd_rx).await;
                    Ok(())
                });
            }
            PresetKind::WebhookOutbound => {
                let cfg = WebhookOutboundConfig::new(&entry.id, &entry.label, preset.clone())
                    .with_credential(&entry.credential);
                let transport = match WebhookOutboundTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "WebhookOutbound init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, outbound-only)",
                    entry.label, entry.preset
                );
            }
            PresetKind::WebhookInbound => {
                let mut cfg = WebhookInboundConfig::new(&entry.id, &entry.label, preset.clone());
                if !entry.credential.is_empty() {
                    cfg = cfg.with_credential(&entry.credential);
                }
                if let Some(secret) = &entry.signing_secret {
                    cfg = cfg.with_signing_secret(secret);
                }
                let transport = match WebhookInboundTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "WebhookInbound init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, webhook-inbound)",
                    entry.label, entry.preset
                );
                webhook_handlers.insert(entry.id.clone(), transport);
            }
        }
    }

    (shutdown_guards, webhook_handlers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_promotion_idempotency_guard() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();

        let session_id = processor.episodic().create_session("test").unwrap();
        let episode_id = processor
            .episodic()
            .store_episode(
                &session_id,
                "user",
                "project uses bun",
                0.9,
                Some("work"),
                None,
            )
            .unwrap();

        let candidates = vec![hippocampus::PromotionCandidate {
            episode_id,
            namespace: "work".to_string(),
            content: "project uses bun".to_string(),
            importance: 0.9,
            reinforcement_count: 3,
        }];

        let first = promote_candidates(&processor, &candidates).await;
        let second = promote_candidates(&processor, &candidates).await;

        assert_eq!(first, 1, "first promotion should persist");
        assert_eq!(second, 0, "second promotion should be skipped");
        assert_eq!(processor.list_facts(Some("work")).len(), 1);
    }

    // Default `BrainConfig::default()` has every reflex disabled / empty —
    // `wire_reflex_sources` should be a no-op in that case so a fresh
    // install spawns zero reflex tasks.
    #[tokio::test]
    async fn wire_reflex_sources_noop_on_default_config() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = Arc::new(signal::SignalProcessor::new(config.clone()).await.unwrap());
        let mut set: tokio::task::JoinSet<anyhow::Result<()>> = tokio::task::JoinSet::new();
        wire_reflex_sources(&config.reflex, &processor, &mut set).await;
        assert!(
            set.is_empty(),
            "no reflex tasks should spawn on default config"
        );
    }

    // Toggling `reflex.cron.enabled = true` should produce exactly one
    // spawned task. The CronReflex polls the episodic scheduler pool and
    // emits `Provenance::Reflex { trigger, .. }` per due intent; here we
    // just confirm the wiring shows up on the JoinSet — the cron logic
    // itself is covered by `reflex` crate tests.
    #[tokio::test]
    async fn wire_reflex_sources_spawns_cron_when_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        config.reflex.cron.enabled = true;
        config.reflex.cron.poll_interval_seconds = 60;
        let processor = Arc::new(signal::SignalProcessor::new(config.clone()).await.unwrap());
        let mut set: tokio::task::JoinSet<anyhow::Result<()>> = tokio::task::JoinSet::new();
        wire_reflex_sources(&config.reflex, &processor, &mut set).await;
        assert_eq!(
            set.len(),
            1,
            "cron reflex should add one task to the join set"
        );
        set.abort_all();
    }
}
