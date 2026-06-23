//! Capability discovery nudge.
//!
//! On a slow cadence the daemon asks the signal layer for capabilities the user
//! has never used (authored, user-facing faculties with no recorded fitness),
//! and surfaces one as a gentle "did you know Brain can…" proactive nudge — so a
//! capability the user never knew about doesn't stay invisible. Each capability
//! is suggested at most once per process; the cadence and bookkeeping live here,
//! the manifest×fitness query lives in `signal::discovery`.
//!
//! Gated like every other nudge by the runtime proactivity toggle, and delivered
//! at priority 1 (a habit-style suggestion, not a health alert) through the same
//! notification router.

use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Spawn the discovery loop. A no-op when discovery is disabled. The first scan
/// lands one interval after boot, not at startup, so a fresh daemon doesn't
/// greet the user with a suggestion before they've done anything.
pub(super) fn spawn_capability_discovery(
    processor: Arc<signal::SignalProcessor>,
    cfg: brain::config::DiscoveryConfig,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    if !cfg.enabled {
        return;
    }
    let runtime_toggle = processor.proactivity_enabled();
    let p = processor.clone();
    set.spawn(async move {
        // Suggested capabilities, so each is surfaced at most once this run.
        let mut suggested: HashSet<String> = HashSet::new();
        let period = tokio::time::Duration::from_secs(cfg.interval_hours.max(1) * 3600);
        let mut ticker = tokio::time::interval(period);
        // Skip the immediate first tick — the first nudge lands one interval in.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            // Respect the live proactivity toggle, the same gate the habit and
            // open-loop nudges honour.
            if !runtime_toggle.load(Ordering::SeqCst) {
                continue;
            }
            // Surface the first untried capability we haven't already suggested.
            let Some(cap) = p
                .untried_capabilities()
                .await
                .into_iter()
                .find(|c| !suggested.contains(&c.tool_id))
            else {
                continue;
            };
            suggested.insert(cap.tool_id.clone());
            tracing::info!(
                tool_id = %cap.tool_id,
                "Capability discovery: suggesting an unused capability"
            );
            if let Some(router) = p.notification_router() {
                router
                    .deliver(signal::notification::ProactiveNotification {
                        content: discovery_message(&cap),
                        triggered_by: format!("capability_discovery:{}", cap.tool_id),
                        priority: 1,
                        agent: None,
                    })
                    .await;
            }
        }
    });
    tracing::info!(
        interval_hours = cfg.interval_hours,
        "Capability discovery scheduled"
    );
}

/// The nudge body — names the capability, why it's useful, and (when authored)
/// how to invoke it.
fn discovery_message(cap: &signal::discovery::UntriedCapability) -> String {
    let mut msg = format!(
        "💡 You have a capability you haven't used yet — `{}`: {}.",
        cap.verb, cap.when_to_use
    );
    if let Some(example) = &cap.example {
        msg.push_str(&format!(" For example: {example}"));
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    use signal::discovery::UntriedCapability;

    #[test]
    fn message_names_capability_and_reason() {
        let cap = UntriedCapability {
            tool_id: "net.check".into(),
            verb: "net.check".into(),
            when_to_use: "test whether a host is reachable".into(),
            example: Some("is github.com reachable?".into()),
        };
        let msg = discovery_message(&cap);
        assert!(msg.contains("net.check"), "{msg}");
        assert!(msg.contains("test whether a host is reachable"), "{msg}");
        assert!(msg.contains("is github.com reachable?"), "{msg}");
    }

    #[test]
    fn message_without_example_omits_the_example_clause() {
        let cap = UntriedCapability {
            tool_id: "x.y".into(),
            verb: "x.y".into(),
            when_to_use: "do a thing".into(),
            example: None,
        };
        let msg = discovery_message(&cap);
        assert!(msg.contains("x.y") && msg.contains("do a thing"));
        assert!(!msg.contains("For example"), "{msg}");
    }
}
