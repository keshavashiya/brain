//! Live capability health as kernel state.
//!
//! A `ToolDescriptor` is *registration-time* truth: it says a
//! capability exists and what it needs. It says nothing about whether that
//! capability would work *right now* — the embedding model could be down, a
//! circuit breaker could be open, a precondition could have lapsed. [`ManifestHealth`]
//! closes that gap: a cheap, cloneable handle holding the kernel's current
//! per-capability health, keyed by `tool_id`.
//!
//! The serve loop's manifest-health sweep is the writer (it probes the
//! subsystems each capability depends on and `replace`s the map); the signal
//! pipeline is the reader (the capability digest and `tools/list` annotate
//! degraded/breaker-open tools so the reasoner doesn't promise a faculty that
//! is currently dead).
//!
//! The default is empty: a processor without a sweep loop (CLI one-shots,
//! tests) reports [`CapabilityHealth::Verified`] for everything via
//! [`ManifestHealth::get`], so behaviour is unchanged from before this state
//! existed.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// The kernel's runtime view of one capability's health.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityHealth {
    /// Probed (or assumed, absent any sweep) to be working.
    Verified,
    /// A dependency the capability needs is currently unreachable or failing.
    /// `reason` is a short, user-facing phrase (e.g. "local embedding model
    /// unreachable").
    Degraded { reason: String },
    /// The capability's circuit breaker is open — recent failures tripped it
    /// and it is in cooldown, so calls fail fast until it half-opens.
    BreakerOpen,
    /// A declared precondition is not met (e.g. an allowlist is empty, a
    /// required path is missing). `reason` is a short, user-facing phrase.
    PreconditionFailed { reason: String },
}

impl CapabilityHealth {
    /// Lowercase wire/display label for the variant kind (no `reason`).
    pub fn as_str(&self) -> &'static str {
        match self {
            CapabilityHealth::Verified => "verified",
            CapabilityHealth::Degraded { .. } => "degraded",
            CapabilityHealth::BreakerOpen => "breaker-open",
            CapabilityHealth::PreconditionFailed { .. } => "precondition-failed",
        }
    }

    /// Whether this capability is usable right now. Only [`Verified`] is
    /// healthy; every other variant means a call would fail or be refused.
    pub fn is_healthy(&self) -> bool {
        matches!(self, CapabilityHealth::Verified)
    }

    /// A one-line human reason for an unhealthy state, suitable for a digest
    /// or `tools/list` annotation. `None` when healthy.
    pub fn reason(&self) -> Option<String> {
        match self {
            CapabilityHealth::Verified => None,
            CapabilityHealth::Degraded { reason }
            | CapabilityHealth::PreconditionFailed { reason } => Some(reason.clone()),
            CapabilityHealth::BreakerOpen => {
                Some("circuit breaker open after recent failures".to_string())
            }
        }
    }
}

/// One health transition surfaced by [`ManifestHealth::replace`]: a capability
/// whose health *changed* between sweeps. `previous` is `None` for a capability
/// the kernel had no prior reading for (first sweep, or newly registered).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthTransition {
    pub tool_id: String,
    pub previous: Option<CapabilityHealth>,
    pub current: CapabilityHealth,
}

/// Shared, cloneable manifest-health handle. Clones observe the same map.
#[derive(Debug, Clone, Default)]
pub struct ManifestHealth {
    inner: Arc<RwLock<HashMap<String, CapabilityHealth>>>,
}

impl ManifestHealth {
    /// Health for one capability. Absent from the map (no sweep yet, or not
    /// covered) reads as [`CapabilityHealth::Verified`] — the closed-world
    /// default that keeps probe-less processors unchanged.
    pub fn get(&self, tool_id: &str) -> CapabilityHealth {
        self.inner
            .read()
            .ok()
            .and_then(|m| m.get(tool_id).cloned())
            .unwrap_or(CapabilityHealth::Verified)
    }

    /// A copy of the whole current map. Empty before the first sweep.
    pub fn snapshot(&self) -> HashMap<String, CapabilityHealth> {
        self.inner.read().map(|m| m.clone()).unwrap_or_default()
    }

    /// Whether any capability is currently unhealthy.
    pub fn any_unhealthy(&self) -> bool {
        self.inner
            .read()
            .map(|m| m.values().any(|h| !h.is_healthy()))
            .unwrap_or(false)
    }

    /// Install a freshly-computed health map, returning the capabilities whose
    /// health *changed* (edges worth surfacing on the bus). A capability that
    /// dropped out of `next` entirely (e.g. its tool was deregistered) is not
    /// reported — only capabilities present in the new map are compared.
    pub fn replace(&self, next: HashMap<String, CapabilityHealth>) -> Vec<HealthTransition> {
        let mut guard = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut transitions = Vec::new();
        for (tool_id, current) in &next {
            let previous = guard.get(tool_id).cloned();
            if previous.as_ref() != Some(current) {
                transitions.push(HealthTransition {
                    tool_id: tool_id.clone(),
                    previous,
                    current: current.clone(),
                });
            }
        }
        *guard = next;
        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(r: &str) -> CapabilityHealth {
        CapabilityHealth::Degraded {
            reason: r.to_string(),
        }
    }

    #[test]
    fn unknown_tool_reads_as_verified() {
        let h = ManifestHealth::default();
        assert_eq!(h.get("memory.search"), CapabilityHealth::Verified);
        assert!(!h.any_unhealthy());
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(CapabilityHealth::Verified.as_str(), "verified");
        assert_eq!(deg("x").as_str(), "degraded");
        assert_eq!(CapabilityHealth::BreakerOpen.as_str(), "breaker-open");
        assert_eq!(
            CapabilityHealth::PreconditionFailed { reason: "y".into() }.as_str(),
            "precondition-failed"
        );
    }

    #[test]
    fn replace_reports_only_changed_capabilities() {
        let h = ManifestHealth::default();
        // First sweep: every entry is a transition (no prior reading).
        let mut m = HashMap::new();
        m.insert("memory.search".to_string(), CapabilityHealth::Verified);
        m.insert("web.search".to_string(), deg("network unreachable"));
        let t = h.replace(m.clone());
        assert_eq!(t.len(), 2);
        assert!(t.iter().all(|x| x.previous.is_none()));

        // Second sweep, identical → no transitions.
        assert!(h.replace(m).is_empty());

        // Third sweep: memory.search degrades, web.search recovers.
        let mut m2 = HashMap::new();
        m2.insert("memory.search".to_string(), deg("embedder down"));
        m2.insert("web.search".to_string(), CapabilityHealth::Verified);
        let mut t2 = h.replace(m2);
        t2.sort_by(|a, b| a.tool_id.cmp(&b.tool_id));
        assert_eq!(t2.len(), 2);
        assert_eq!(t2[0].tool_id, "memory.search");
        assert_eq!(t2[0].previous, Some(CapabilityHealth::Verified));
        assert_eq!(t2[0].current, deg("embedder down"));
        assert_eq!(t2[1].tool_id, "web.search");
        assert_eq!(t2[1].current, CapabilityHealth::Verified);
    }

    #[test]
    fn clones_share_one_map() {
        let a = ManifestHealth::default();
        let b = a.clone();
        let mut m = HashMap::new();
        m.insert("x".to_string(), deg("down"));
        a.replace(m);
        assert_eq!(b.get("x"), deg("down"));
        assert!(b.any_unhealthy());
    }

    #[test]
    fn reason_is_present_only_when_unhealthy() {
        assert_eq!(CapabilityHealth::Verified.reason(), None);
        assert_eq!(deg("nope").reason().as_deref(), Some("nope"));
        assert!(CapabilityHealth::BreakerOpen.reason().is_some());
    }
}
