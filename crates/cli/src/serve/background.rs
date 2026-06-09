//! Background-task spawners for `cmd_serve`.
//!
//! Each `spawn_*` helper attaches one long-running maintenance loop to
//! the shared join set. Helpers are no-ops at the caller-site when the
//! relevant config is disabled — `cmd_serve` decides whether to invoke
//! them based on `config.proactivity` / `config.memory.consolidation`.
//! The graph compactor always runs.
//!
//! Scheduled-intent *firing* no longer lives here: it was migrated off
//! the direct-execution poller onto `reflex::CronReflex`, which routes
//! every due intent through the full pipeline (identity, confirmation,
//! breakers, audit) instead of delivering a bare notification.
//!
//! `promote_candidates` (consolidation companion) also lives here
//! because the consolidation loop is its only caller.

use std::sync::Arc;

#[cfg(feature = "ganglia")]
pub(super) fn spawn_habit_engine(
    processor: Arc<signal::SignalProcessor>,
    proactivity: &brain::config::ProactivityConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let habit_cfg = ganglia::HabitConfig {
        max_per_day: proactivity.max_per_day,
        min_interval_minutes: proactivity.min_interval_minutes,
        quiet_start: proactivity.quiet_hours.start.clone(),
        quiet_end: proactivity.quiet_hours.end.clone(),
        ..Default::default()
    };
    let runtime_toggle = processor.proactivity_enabled();
    let p = processor.clone();
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
            if !runtime_toggle.load(std::sync::atomic::Ordering::SeqCst) {
                continue;
            }
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
        interval_minutes = proactivity.min_interval_minutes,
        "Proactivity engine scheduled"
    );
}

#[cfg(feature = "ganglia")]
pub(super) fn spawn_open_loop_detector(
    processor: Arc<signal::SignalProcessor>,
    ol_cfg: brain::config::OpenLoopDetectionConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let runtime_toggle = processor.proactivity_enabled();
    let p = processor.clone();
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
            if !runtime_toggle.load(std::sync::atomic::Ordering::SeqCst) {
                continue;
            }
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
        interval_minutes = ol_cfg.check_interval_minutes,
        "Open-loop detector scheduled"
    );
}

pub(super) fn spawn_consolidator(
    processor: Arc<signal::SignalProcessor>,
    interval_hours: u32,
    prune_threshold: f64,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let p = processor.clone();
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

/// 24h reflection cycle: half-life decays every node's weight and prunes
/// the ones that fall below the eviction cutoff. The graph is opened
/// lazily per tick against the episodic pool so we don't hold a
/// `SqliteGraph` across ticks — the pool is the shared resource, the
/// wrapper is cheap.
pub(super) fn spawn_graph_compactor(
    processor: Arc<signal::SignalProcessor>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    use hippocampus::Compactor as _;
    let p = processor.clone();
    set.spawn(async move {
        let compactor = hippocampus::DefaultCompactor::new(hippocampus::CompactConfig::default());
        let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(24 * 3600));
        // Skip the immediate first tick — `interval` fires once at start;
        // we want the first compaction 24h after boot.
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

/// Resource sampler: one bounded task that gauges process RSS, CPU, open
/// SQLite connections, and `~/.brain` disk usage every `sample_secs`, writing
/// the readings into the shared [`metrics::ResourceMetrics`] store.
///
/// The probe is built once and reused across ticks so `sysinfo` can compute CPU
/// as a delta since the previous sample — the first tick therefore reports `0%`
/// CPU (no baseline) but populates every other gauge immediately. This loop is
/// itself the source of the resource gauges, so it satisfies the "no background
/// loop without a metric" invariant by construction.
pub(super) fn spawn_resource_sampler(
    processor: Arc<signal::SignalProcessor>,
    resource_metrics: Arc<metrics::ResourceMetrics>,
    data_dir: std::path::PathBuf,
    sample_secs: u64,
    thresholds: brain::config::ResourceThresholds,
    log_sample_1_in_n: u32,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let p = processor.clone();
    set.spawn(async move {
        // The per-tick "Resource sample" line is a heartbeat: high-volume and
        // low-information once gauges are also on `/metrics` and the bus. Throttle
        // it to 1-in-N so a fast sample cadence doesn't flood the log; pressure
        // crossings below are never sampled — they always log.
        let heartbeat_sampler = observe::LogSampler::one_in(log_sample_1_in_n);
        let mut probe = super::resource::ResourceProbe::new(data_dir);
        let mut tracker = super::resource::PressureTracker::default();
        // No leading `tick()` skip here (unlike consolidation): we want gauges
        // populated as soon as the daemon comes up, not one interval later.
        let mut ticker =
            tokio::time::interval(tokio::time::Duration::from_secs(sample_secs.max(1)));
        loop {
            ticker.tick().await;
            let connections = u64::from(p.episodic().pool().open_connections());
            let snap = probe.sample(Some(connections));

            resource_metrics.set_rss_bytes(snap.rss_bytes);
            resource_metrics.set_cpu_pct(snap.cpu_pct);
            resource_metrics.set_open_connections(snap.open_connections);
            resource_metrics.set_open_fds(snap.open_fds);
            resource_metrics.set_disk_bytes(snap.disk_bytes);

            if heartbeat_sampler.should_emit() {
                tracing::debug!(
                    rss_mb = snap.rss_bytes.map(|b| b / (1024 * 1024)),
                    cpu_pct = snap.cpu_pct,
                    open_connections = snap.open_connections,
                    disk_mb = snap.disk_bytes.map(|b| b / (1024 * 1024)),
                    "Resource sample"
                );
            }

            // Edge-triggered: act on a fresh crossing only, so neither the bus
            // nor the user is spammed while a gauge stays over its ceiling. Each
            // crossing both publishes a `ResourcePressure` event (for the Live
            // tab / metrics) and delivers a proactive, actionable notification
            // (Issue 136) — the latter is what reaches the user out-of-band.
            for c in tracker.evaluate(&snap, &thresholds) {
                tracing::warn!(
                    gauge = c.gauge,
                    value = c.value,
                    threshold = c.threshold,
                    severity = c.severity,
                    "Resource pressure: {} over ceiling",
                    c.gauge
                );
                if let Some(observer) = p.observer() {
                    let ev = observe::BrainEvent::ResourcePressure {
                        id: uuid::Uuid::new_v4(),
                        gauge: c.gauge.to_string(),
                        value: c.value,
                        threshold: c.threshold,
                        severity: c.severity.to_string(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
                if let Some(router) = p.notification_router() {
                    // Operational health alert — delivered regardless of the
                    // proactivity toggle (that gates habit-style nudges, not
                    // self-health warnings). Priority 2 outranks habit nudges (1).
                    router
                        .deliver(signal::notification::ProactiveNotification {
                            content: c.advisory(),
                            triggered_by: format!("resource_pressure:{}", c.gauge),
                            priority: 2,
                            agent: None,
                        })
                        .await;
                }
            }
        }
    });
    tracing::info!(sample_secs, "Resource sampler scheduled");
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
        let mut config = brain::BrainConfig::default();
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
