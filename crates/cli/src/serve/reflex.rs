//! Reflex source wiring for `cmd_serve`.
//!
//! Construct every configured `reflex::ReflexSource`, hook it into the
//! pipeline via `signal::reflex_runner::spawn_reflex`, and park the
//! resulting task on the shared join set. Each firing builds a `Signal`
//! stamped with `Provenance::Reflex { trigger, … }` so the pipeline can
//! distinguish reflex-driven activity from user-typed input without
//! inspecting the trigger string itself.
//!
//! Failures during subscribe (e.g. an FS watcher can't bind to a
//! non-existent path) are logged at `warn` and do not abort serve —
//! other reflexes still spawn, mirroring how adapter bind failures
//! degrade gracefully.

use std::sync::Arc;

pub(super) async fn wire_reflex_sources(
    cfg: &brain::config::ReflexConfig,
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

    // SysState — backed by `sys_sampler::SysSampler`, which composes real
    // kernel signals: `battery_below` / `on_ac_changed` from the power probe
    // (pmset/sysfs), `network_changed` from the shared connectivity handle, and
    // `lock_changed` from systemd-logind (Linux) / CoreGraphics (macOS). Any
    // dimension the platform can't report stays `None`, so its rule never fires
    // spuriously.
    if cfg.sys.enabled && !cfg.sys.rules.is_empty() {
        let rules: Vec<reflex::SysStateRule> = cfg
            .sys
            .rules
            .iter()
            .map(|r| match r {
                brain::config::SysReflexRuleEntry::BatteryBelow { threshold } => {
                    reflex::SysStateRule::BatteryBelow(*threshold)
                }
                brain::config::SysReflexRuleEntry::OnAcChanged => reflex::SysStateRule::OnAcChanged,
                brain::config::SysReflexRuleEntry::NetworkChanged => {
                    reflex::SysStateRule::NetworkChanged
                }
                brain::config::SysReflexRuleEntry::LockChanged => reflex::SysStateRule::LockChanged,
            })
            .collect();
        let sys_cfg = reflex::SysStateReflexConfig::new(std::time::Duration::from_secs(
            cfg.sys.poll_interval_seconds,
        ))
        .with_rules(rules);
        let sampler: Arc<dyn reflex::SysStateSampler> = Arc::new(
            super::sys_sampler::SysSampler::new(processor.connectivity()),
        );
        let source: Arc<dyn ReflexSource> =
            Arc::new(reflex::SysStateReflex::new("sys", sampler, sys_cfg));
        match signal::reflex_runner::spawn_reflex("sys", source, processor.clone(), move |ev| {
            reflex_signal("sys", ev)
        })
        .await
        {
            Ok(handle) => {
                tracing::info!("SysState reflex spawned (battery + AC + network + lock)");
                set.spawn(async move {
                    let _ = handle.await;
                    Ok(())
                });
            }
            Err(e) => tracing::warn!(error = %e, "sys reflex subscribe failed; skipping"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Default `BrainConfig::default()` has every reflex disabled / empty —
    // `wire_reflex_sources` should be a no-op in that case so a fresh
    // install spawns zero reflex tasks.
    #[tokio::test]
    async fn wire_reflex_sources_noop_on_default_config() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain::BrainConfig::default();
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
        let mut config = brain::BrainConfig::default();
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
