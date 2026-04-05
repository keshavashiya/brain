//! Serve command — run all services in the foreground.

use std::sync::Arc;

use backends::*;

pub(crate) async fn cmd_serve(
    config: &brain_core::BrainConfig,
    http: bool,
    ws: bool,
    grpc: bool,
    mcp: bool,
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

    let run_all = !http && !ws && !grpc && !mcp;

    // Build the fully-wired processor via the shared bootstrap path.
    let mut processor = crate::bootstrap::build_processor(config).await?;

    // Wire the notification router (serve-specific — needed for proactive delivery)
    {
        let db = processor.episodic().pool().clone();
        let delivery_config = config.proactivity.delivery.clone();
        let mut router = signal::notification::NotificationRouter::new(db, delivery_config);

        if config.actions.messaging.enabled && !config.actions.messaging.channels.is_empty() {
            let res = &config.actions.resilience;
            match WebhookMessageBackend::new(
                &config.actions.messaging.channels,
                config.actions.messaging.timeout_ms,
                res,
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

    if run_all || http {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.http.port;
        println!("  Synapse HTTP  → http://{}:{}", h, port);
        set.spawn(async move { httpadapter::serve(p, &h, port).await });
    }

    if run_all || ws {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.ws.port;
        println!("  Synapse WS    → ws://{}:{}", h, port);
        set.spawn(async move { wsadapter::serve(p, &h, port).await });
    }

    if run_all || grpc {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.grpc.port;
        println!("  Synapse gRPC  → {}:{}", h, port);
        set.spawn(async move { grpcadapter::serve(p, &h, port).await });
    }

    if run_all || mcp {
        let p = processor.clone();
        let h = host.clone();
        let port = config.adapters.mcp.port;
        println!("  Synapse MCP   → http://{}:{}", h, port);
        set.spawn(async move { mcp::serve_http(p, &h, port).await });
    }

    // ── Proactivity / habit engine background task ────────────────────
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
}
