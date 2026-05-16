//! System-state reflex — polls a pluggable sampler and emits a
//! [`ReflexEvent`] when one of the configured [`SysStateRule`]s
//! observes a transition.
//!
//! ## Why a sampler abstraction
//!
//! Real battery / network / lock-state APIs are platform-specific
//! (macOS IOKit, Linux `/sys`, Windows WMI) and most published Rust
//! crates that wrap them either pull heavy native deps or behave
//! flakily under CI. Rather than pin one set, this slice ships the
//! [`SysStateSampler`] trait + [`NoopSampler`] default. Consumers (or
//! a later PR-5d.x slice) can plug a real per-OS sampler in without
//! changing the reflex surface.
//!
//! ## Edge-triggered semantics
//!
//! Every rule fires on a **transition**, not on a level. If the
//! battery starts at 15% and the rule is `BatteryBelow(20)`, the
//! reflex stays silent — it only fires when the battery *crosses*
//! the threshold from above. Same for network online↔offline, AC
//! connect/disconnect, and lock/unlock. This keeps the bus clean
//! when the reflex starts in an already-degraded state and avoids
//! re-emitting the same event on every tick.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{ReflexError, ReflexEvent, ReflexSource, ReflexStream};

/// One sampled view of the relevant OS-level state. Every field is
/// `Option` so partial samplers (e.g. one that only knows the
/// battery) round-trip without lying about the rest of the world.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SysSnapshot {
    /// Battery charge percentage, 0-100. `None` when no battery
    /// present or the sampler can't read it.
    pub battery_percent: Option<u8>,
    /// Whether the machine is on AC power. `None` when unknown.
    pub on_ac: Option<bool>,
    /// Coarse network reachability. `None` when unknown.
    pub network: Option<NetworkState>,
    /// Whether the session is locked (screen lock, sleep). `None`
    /// when unknown.
    pub locked: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NetworkState {
    Online,
    Offline,
}

impl NetworkState {
    fn as_str(self) -> &'static str {
        match self {
            NetworkState::Online => "online",
            NetworkState::Offline => "offline",
        }
    }
}

/// Pluggable sampler. Implementations should return promptly — the
/// reflex calls this on every poll tick.
#[async_trait]
pub trait SysStateSampler: Send + Sync {
    async fn sample(&self) -> SysSnapshot;
}

/// Sampler that always returns an empty snapshot. Useful as a
/// default and as the per-OS placeholder until real implementations
/// land.
pub struct NoopSampler;

#[async_trait]
impl SysStateSampler for NoopSampler {
    async fn sample(&self) -> SysSnapshot {
        SysSnapshot::default()
    }
}

/// What to watch for. Each rule is edge-triggered: it fires when the
/// transition predicate becomes true between consecutive snapshots,
/// not while the predicate stays true.
#[derive(Debug, Clone, Copy)]
pub enum SysStateRule {
    /// Battery percentage crossed below `threshold` from at-or-above.
    BatteryBelow(u8),
    /// `on_ac` flipped (either direction). Fires once per flip.
    OnAcChanged,
    /// `network` flipped between Online and Offline.
    NetworkChanged,
    /// `locked` flipped between false and true.
    LockChanged,
}

#[derive(Debug, Clone)]
pub struct SysStateReflexConfig {
    /// Interval between sampler calls. Default 30s — fast enough to
    /// catch a power-loss within a minute, slow enough not to wake
    /// the CPU constantly.
    pub poll_interval: Duration,
    /// Rules to evaluate on each transition. Empty list means the
    /// reflex never fires (useful as a smoke test).
    pub rules: Vec<SysStateRule>,
}

impl Default for SysStateReflexConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(30),
            rules: Vec::new(),
        }
    }
}

impl SysStateReflexConfig {
    pub fn new(poll_interval: Duration) -> Self {
        Self {
            poll_interval,
            rules: Vec::new(),
        }
    }

    pub fn with_rule(mut self, rule: SysStateRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_rules(mut self, rules: impl IntoIterator<Item = SysStateRule>) -> Self {
        self.rules.extend(rules);
        self
    }
}

pub struct SysStateReflex {
    name: String,
    sampler: Arc<dyn SysStateSampler>,
    config: SysStateReflexConfig,
}

impl SysStateReflex {
    pub fn new(
        name: impl Into<String>,
        sampler: Arc<dyn SysStateSampler>,
        config: SysStateReflexConfig,
    ) -> Self {
        Self {
            name: name.into(),
            sampler,
            config,
        }
    }

    pub fn config(&self) -> &SysStateReflexConfig {
        &self.config
    }
}

#[async_trait]
impl ReflexSource for SysStateReflex {
    fn name(&self) -> &str {
        &self.name
    }

    async fn subscribe(self: Arc<Self>) -> Result<ReflexStream, ReflexError> {
        let (out_tx, out_rx) = mpsc::channel::<ReflexEvent>(64);

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(self.config.poll_interval);
            // Skip the immediate auto-fire so the first event follows
            // a real poll cadence, mirroring CronReflex.
            ticker.tick().await;

            // Seed with one sample so we have a baseline to diff
            // against — without this, every rule would either fire
            // spuriously on first compare or never fire.
            let mut prev = self.sampler.sample().await;

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        if out_tx.is_closed() {
                            return;
                        }
                        let current = self.sampler.sample().await;
                        for rule in &self.config.rules {
                            if let Some(event) = evaluate_rule(*rule, &prev, &current) {
                                if out_tx.send(event).await.is_err() {
                                    return;
                                }
                            }
                        }
                        prev = current;
                    }
                    _ = out_tx.closed() => return,
                }
            }
        });

        Ok(ReceiverStream::new(out_rx).boxed())
    }
}

/// Pure rule-evaluation function — pulled out so tests can pin
/// transition semantics without standing up the polling loop.
pub fn evaluate_rule(
    rule: SysStateRule,
    prev: &SysSnapshot,
    current: &SysSnapshot,
) -> Option<ReflexEvent> {
    match rule {
        SysStateRule::BatteryBelow(threshold) => {
            let now = current.battery_percent?;
            // Was at-or-above (or unknown) before, now below.
            let was_above = match prev.battery_percent {
                Some(p) => p >= threshold,
                None => true,
            };
            if was_above && now < threshold {
                Some(ReflexEvent::new(
                    format!("sys:battery_below:{threshold}"),
                    serde_json::json!({
                        "dimension": "battery",
                        "threshold": threshold,
                        "before": prev.battery_percent,
                        "after": now,
                    }),
                ))
            } else {
                None
            }
        }
        SysStateRule::OnAcChanged => {
            let before = prev.on_ac?;
            let after = current.on_ac?;
            if before == after {
                return None;
            }
            let direction = if after { "connected" } else { "disconnected" };
            Some(ReflexEvent::new(
                format!("sys:ac:{direction}"),
                serde_json::json!({
                    "dimension": "ac",
                    "before": before,
                    "after": after,
                }),
            ))
        }
        SysStateRule::NetworkChanged => {
            let before = prev.network?;
            let after = current.network?;
            if before == after {
                return None;
            }
            Some(ReflexEvent::new(
                format!("sys:network:{}", after.as_str()),
                serde_json::json!({
                    "dimension": "network",
                    "before": before,
                    "after": after,
                }),
            ))
        }
        SysStateRule::LockChanged => {
            let before = prev.locked?;
            let after = current.locked?;
            if before == after {
                return None;
            }
            let label = if after { "locked" } else { "unlocked" };
            Some(ReflexEvent::new(
                format!("sys:lock:{label}"),
                serde_json::json!({
                    "dimension": "lock",
                    "before": before,
                    "after": after,
                }),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use std::collections::VecDeque;
    use tokio::sync::Mutex;

    /// Sampler that returns a scripted sequence of snapshots; once
    /// exhausted, it returns the last snapshot forever (so the
    /// reflex's poll loop stays alive without panicking).
    struct ScriptedSampler {
        script: Mutex<VecDeque<SysSnapshot>>,
        last: Mutex<SysSnapshot>,
    }

    impl ScriptedSampler {
        fn new(samples: Vec<SysSnapshot>) -> Self {
            Self {
                script: Mutex::new(samples.into()),
                last: Mutex::new(SysSnapshot::default()),
            }
        }
    }

    #[async_trait]
    impl SysStateSampler for ScriptedSampler {
        async fn sample(&self) -> SysSnapshot {
            let mut script = self.script.lock().await;
            if let Some(s) = script.pop_front() {
                *self.last.lock().await = s.clone();
                s
            } else {
                self.last.lock().await.clone()
            }
        }
    }

    #[tokio::test]
    async fn sys_reflex_with_noop_sampler_emits_nothing() {
        let reflex = Arc::new(SysStateReflex::new(
            "sys-noop",
            Arc::new(NoopSampler),
            SysStateReflexConfig::new(Duration::from_millis(20))
                .with_rule(SysStateRule::BatteryBelow(20)),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");
        let res = tokio::time::timeout(Duration::from_millis(80), stream.next()).await;
        assert!(res.is_err(), "noop sampler must never fire any rule");
    }

    #[tokio::test]
    async fn sys_reflex_fires_on_battery_threshold_crossing() {
        let sampler = Arc::new(ScriptedSampler::new(vec![
            SysSnapshot {
                battery_percent: Some(50),
                ..Default::default()
            },
            SysSnapshot {
                battery_percent: Some(15),
                ..Default::default()
            },
        ]));
        let reflex = Arc::new(SysStateReflex::new(
            "sys-batt",
            sampler,
            SysStateReflexConfig::new(Duration::from_millis(20))
                .with_rule(SysStateRule::BatteryBelow(20)),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");
        assert_eq!(event.trigger, "sys:battery_below:20");
        assert_eq!(
            event.payload.get("after").and_then(|v| v.as_u64()),
            Some(15)
        );
        assert_eq!(
            event.payload.get("before").and_then(|v| v.as_u64()),
            Some(50)
        );
    }

    #[tokio::test]
    async fn sys_reflex_fires_on_network_flip() {
        let sampler = Arc::new(ScriptedSampler::new(vec![
            SysSnapshot {
                network: Some(NetworkState::Online),
                ..Default::default()
            },
            SysSnapshot {
                network: Some(NetworkState::Offline),
                ..Default::default()
            },
        ]));
        let reflex = Arc::new(SysStateReflex::new(
            "sys-net",
            sampler,
            SysStateReflexConfig::new(Duration::from_millis(20))
                .with_rule(SysStateRule::NetworkChanged),
        ));
        let mut stream = reflex.subscribe().await.expect("subscribe");
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("event within timeout")
            .expect("stream still open");
        assert_eq!(event.trigger, "sys:network:offline");
    }

    #[test]
    fn battery_rule_is_edge_triggered_not_level() {
        // Start already below threshold — no event should fire on
        // first observation because there's no prior to compare.
        let prev = SysSnapshot {
            battery_percent: Some(10),
            ..Default::default()
        };
        let curr = SysSnapshot {
            battery_percent: Some(8),
            ..Default::default()
        };
        assert!(
            evaluate_rule(SysStateRule::BatteryBelow(20), &prev, &curr).is_none(),
            "stays-below must not re-fire"
        );
    }

    #[test]
    fn battery_rule_skips_when_current_unknown() {
        let prev = SysSnapshot {
            battery_percent: Some(50),
            ..Default::default()
        };
        let curr = SysSnapshot::default();
        assert!(evaluate_rule(SysStateRule::BatteryBelow(20), &prev, &curr).is_none());
    }

    #[test]
    fn lock_rule_emits_correct_direction() {
        let prev = SysSnapshot {
            locked: Some(true),
            ..Default::default()
        };
        let curr = SysSnapshot {
            locked: Some(false),
            ..Default::default()
        };
        let ev = evaluate_rule(SysStateRule::LockChanged, &prev, &curr).expect("event");
        assert_eq!(ev.trigger, "sys:lock:unlocked");
    }

    #[test]
    fn ac_rule_skips_when_unchanged() {
        let prev = SysSnapshot {
            on_ac: Some(true),
            ..Default::default()
        };
        let curr = SysSnapshot {
            on_ac: Some(true),
            ..Default::default()
        };
        assert!(evaluate_rule(SysStateRule::OnAcChanged, &prev, &curr).is_none());
    }

    #[test]
    fn ac_rule_emits_disconnected_label() {
        let prev = SysSnapshot {
            on_ac: Some(true),
            ..Default::default()
        };
        let curr = SysSnapshot {
            on_ac: Some(false),
            ..Default::default()
        };
        let ev = evaluate_rule(SysStateRule::OnAcChanged, &prev, &curr).expect("event");
        assert_eq!(ev.trigger, "sys:ac:disconnected");
    }
}
