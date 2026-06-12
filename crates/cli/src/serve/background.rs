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
            // Background nudge generation is fast-tier work — on a
            // configured local fast lane it never leaves the machine.
            p.llm_tier(cortex::llm::TaskTier::Fast),
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
    defer_on_battery: bool,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let p = processor.clone();
    let power = processor.power();
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
            // Battery etiquette: a due consolidation holds (not skips) until
            // external power returns, so it still runs — just not on battery.
            super::power::hold_while_on_battery(&power, defer_on_battery, "memory consolidation")
                .await;
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
    defer_on_battery: bool,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    use hippocampus::Compactor as _;
    let p = processor.clone();
    let power = processor.power();
    set.spawn(async move {
        let compactor = hippocampus::DefaultCompactor::new(hippocampus::CompactConfig::default());
        let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(24 * 3600));
        // Skip the immediate first tick — `interval` fires once at start;
        // we want the first compaction 24h after boot.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Battery etiquette: a due sweep holds until external power
            // returns rather than running on battery.
            super::power::hold_while_on_battery(&power, defer_on_battery, "graph compaction").await;
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

/// Spawn one bounded health-probe loop for a single configured service
/// (Issue 135). The loop probes `svc.target` every `svc.interval_secs`, and on
/// an up↔down *transition* both publishes a `ServiceHealthChanged` event (for
/// the Live tab / metrics) and delivers a proactive, actionable notification
/// through the router — the same two surfaces the resource sampler feeds.
///
/// One task per service keeps each probe on its own cadence with a private
/// edge state, so there is no shared map to lock. The probe itself is the only
/// I/O, so the loop satisfies the bounded-task invariant by construction.
pub(super) fn spawn_service_monitor(
    processor: Arc<signal::SignalProcessor>,
    svc: brain::config::ServiceCheck,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let name = svc.name.clone();
    let p = processor.clone();
    set.spawn(async move {
        // One client per loop, carrying the per-probe timeout. `build` only
        // fails on a TLS/backend misconfiguration; fall back to a default
        // client so a single bad timeout can't silence the monitor.
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(svc.timeout_secs.max(1)))
            .build()
            .unwrap_or_default();
        let mut edge = super::health::HealthEdge::default();
        let mut ticker =
            tokio::time::interval(std::time::Duration::from_secs(svc.interval_secs.max(1)));
        // No leading-tick skip: probe as soon as the daemon comes up so an
        // already-down service is surfaced immediately, not one interval later.
        loop {
            ticker.tick().await;
            let (healthy, detail) = match super::health::probe(&client, &svc).await {
                Ok(()) => (true, String::new()),
                Err(reason) => (false, reason),
            };
            // Edge-triggered: act only when reachability flips, so neither the
            // bus nor the user is spammed while a service holds one state.
            let Some(now_healthy) = edge.evaluate(healthy) else {
                continue;
            };
            if now_healthy {
                tracing::info!(service = %svc.name, target = %svc.target, "Service recovered");
            } else {
                tracing::warn!(
                    service = %svc.name,
                    target = %svc.target,
                    detail = %detail,
                    "Service unreachable"
                );
            }
            if let Some(observer) = p.observer() {
                let ev = observe::BrainEvent::ServiceHealthChanged {
                    id: uuid::Uuid::new_v4(),
                    service: svc.name.clone(),
                    target: svc.target.clone(),
                    healthy: now_healthy,
                    detail: detail.clone(),
                    ts: chrono::Utc::now(),
                };
                let _ = observer.publish(ev).await;
            }
            if let Some(router) = p.notification_router() {
                // Operational health alert — delivered regardless of the
                // proactivity toggle (that gates habit-style nudges, not
                // self-health warnings). Priority 2 outranks habit nudges (1),
                // matching the resource-pressure path.
                router
                    .deliver(signal::notification::ProactiveNotification {
                        content: super::health::advisory(&svc, now_healthy, &detail),
                        triggered_by: format!("service_health:{}", svc.name),
                        priority: 2,
                        agent: None,
                    })
                    .await;
            }
        }
    });
    tracing::info!(service = %name, "Service health monitor scheduled");
}

/// Spawn the connectivity probe — the single writer behind the processor's
/// [`brain::Connectivity`] handle. Each round TCP-connects the derived target
/// set (see `connectivity::probe_targets`; already-configured remote provider
/// endpoints only, so no new egress) and folds the tally into
/// `Online / Degraded / Offline`. On a *transition* it publishes a
/// `ConnectivityChanged` event and delivers a proactive notification — the
/// same two surfaces and edge discipline as the resource sampler and the
/// service monitors. The caller skips spawning entirely when probing is
/// disabled or the target set is empty, which pins the state to `Online`.
pub(super) fn spawn_connectivity_probe(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::ConnectivityProbeConfig,
    targets: Vec<String>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let handle = processor.connectivity();
    let p = processor;
    let target_count = targets.len();
    set.spawn(async move {
        let timeout = std::time::Duration::from_secs(cfg.timeout_secs.max(1));
        let mut ticker =
            tokio::time::interval(std::time::Duration::from_secs(cfg.interval_secs.max(1)));
        // No leading-tick skip: a daemon that boots offline should know it
        // within one probe round, not one interval later.
        loop {
            ticker.tick().await;
            let reachable = super::connectivity::probe_round(&targets, timeout).await;
            let state = super::connectivity::state_for(reachable, target_count);
            // Edge-triggered: `set` reports the previous state only on a
            // transition, so neither the bus nor the user is spammed while
            // the network holds one state.
            let Some(previous) = handle.set(state) else {
                continue;
            };
            let detail = super::connectivity::detail_for(reachable, target_count);
            if state == brain::ConnectivityState::Online {
                tracing::info!(previous = %previous, "Connectivity restored");
            } else {
                tracing::warn!(
                    previous = %previous,
                    state = %state,
                    detail = %detail,
                    "Connectivity changed"
                );
            }
            if let Some(observer) = p.observer() {
                let ev = observe::BrainEvent::ConnectivityChanged {
                    id: uuid::Uuid::new_v4(),
                    state: state.as_str().to_string(),
                    previous: previous.as_str().to_string(),
                    detail: detail.clone(),
                    ts: chrono::Utc::now(),
                };
                let _ = observer.publish(ev).await;
            }
            if let Some(router) = p.notification_router() {
                // Operational health alert — delivered regardless of the
                // proactivity toggle, priority 2, matching the resource and
                // service-health paths. An offline transition still reaches
                // local sinks (CLI tail / terminal) even though outbound
                // channels are down.
                router
                    .deliver(signal::notification::ProactiveNotification {
                        content: super::connectivity::advisory(state, &detail),
                        triggered_by: "connectivity".to_string(),
                        priority: 2,
                        agent: None,
                    })
                    .await;
            }
        }
    });
    tracing::info!(
        targets = target_count,
        interval_secs = cfg.interval_secs,
        "Connectivity probe scheduled"
    );
}

/// Spawn the power probe — the single writer behind the processor's
/// [`brain::Power`] handle. Each round asks the platform for the power
/// source (see `power::probe`; `pmset` / sysfs, no network) and folds it
/// into `External / Battery`. On a *transition* it publishes a
/// `PowerStateChanged` event — but, unlike the connectivity and service
/// monitors, no proactive notification: plugging and unplugging a laptop is
/// routine and user-initiated, not news; the consequences surface in the
/// maintenance loops' defer/resume log lines and the capability digest.
/// If the very first probe reports the source undetectable (non-mac/linux
/// platform, or a desktop with nothing to report), the task exits and the
/// state stays pinned `External`.
pub(super) fn spawn_power_probe(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::PowerProbeConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    let handle = processor.power();
    let p = processor;
    set.spawn(async move {
        // Support check: one up-front probe decides whether this platform
        // can answer at all, so an unsupported host doesn't keep a no-op
        // loop alive forever.
        if super::power::probe().await.is_none() {
            tracing::info!(
                "Power probe: source undetectable on this platform — state pinned to external"
            );
            return Ok(());
        }
        let mut ticker =
            tokio::time::interval(std::time::Duration::from_secs(cfg.interval_secs.max(1)));
        // No leading-tick skip: a daemon booted on battery should know
        // within one round, not one interval later.
        loop {
            ticker.tick().await;
            // A transient mid-flight read failure holds the last state
            // rather than flapping.
            let Some((state, detail)) = super::power::probe().await else {
                continue;
            };
            // Edge-triggered, same as every other monitor here.
            if let Some(previous) = handle.set(state) {
                if state == brain::PowerState::Battery {
                    tracing::info!(detail = %detail, "Power: now on battery");
                } else {
                    tracing::info!(previous = %previous, "Power: external power restored");
                }
                if let Some(observer) = p.observer() {
                    let ev = observe::BrainEvent::PowerStateChanged {
                        id: uuid::Uuid::new_v4(),
                        state: state.as_str().to_string(),
                        previous: previous.as_str().to_string(),
                        detail: detail.clone(),
                        ts: chrono::Utc::now(),
                    };
                    let _ = observer.publish(ev).await;
                }
            }
        }
    });
    tracing::info!(interval_secs = cfg.interval_secs, "Power probe scheduled");
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
