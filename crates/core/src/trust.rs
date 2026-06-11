//! Per-agent memory-trust policy.
//!
//! Every stored fact and episode carries its originating agent
//! (`None` = the user's own input through a local surface). Recall
//! scoring multiplies each memory's score by the trust weight of the
//! agent that wrote it, so a memory written by a low-trust agent
//! cannot dominate context assembly for a query no matter how its
//! content or claimed importance is crafted. The decision record for
//! this model lives in the decisions journal (memory-trust entry,
//! 2026-06-11).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// `memory.trust` config block: trust weights in `[0, 1]` for
/// agent-attributed memories. User-origin memories (no agent) are
/// always weighted 1.0 and are not configurable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrustConfig {
    /// Weight applied to memories from agents with no entry in
    /// [`agents`](Self::agents). Default 1.0 — provenance weighting is
    /// opt-in and zero-config behavior is unchanged.
    #[serde(default = "default_agent_trust")]
    pub default_agent_trust: f64,
    /// Per-agent overrides, keyed by the agent id stored on the memory.
    #[serde(default)]
    pub agents: HashMap<String, f64>,
}

fn default_agent_trust() -> f64 {
    1.0
}

impl Default for MemoryTrustConfig {
    fn default() -> Self {
        Self {
            default_agent_trust: 1.0,
            agents: HashMap::new(),
        }
    }
}

impl MemoryTrustConfig {
    /// Compile into a config-free [`AgentTrustPolicy`] for subsystems
    /// that never see `BrainConfig`. Weights are clamped to `[0, 1]`.
    pub fn policy(&self) -> AgentTrustPolicy {
        AgentTrustPolicy {
            default_trust: self.default_agent_trust.clamp(0.0, 1.0),
            agents: self
                .agents
                .iter()
                .map(|(a, t)| (a.clone(), t.clamp(0.0, 1.0)))
                .collect(),
        }
    }
}

/// Compiled trust weights, consumed by the recall engine.
#[derive(Debug, Clone)]
pub struct AgentTrustPolicy {
    default_trust: f64,
    agents: HashMap<String, f64>,
}

impl Default for AgentTrustPolicy {
    /// The no-op policy: every memory weighs 1.0.
    fn default() -> Self {
        Self {
            default_trust: 1.0,
            agents: HashMap::new(),
        }
    }
}

impl AgentTrustPolicy {
    /// Trust weight for a memory written by `agent`. `None` (the user's
    /// own input) is pinned at 1.0; a configured agent reads its entry;
    /// an unknown agent reads the configured default.
    pub fn trust_of(&self, agent: Option<&str>) -> f64 {
        match agent {
            None => 1.0,
            Some(a) => self.agents.get(a).copied().unwrap_or(self.default_trust),
        }
    }

    /// True when the policy cannot change any score — the zero-config
    /// fast path: every weight is 1.0.
    pub fn is_noop(&self) -> bool {
        self.default_trust >= 1.0 && self.agents.values().all(|t| *t >= 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_origin_is_pinned_at_full_trust() {
        let policy = MemoryTrustConfig {
            default_agent_trust: 0.2,
            agents: HashMap::new(),
        }
        .policy();
        assert_eq!(policy.trust_of(None), 1.0);
    }

    #[test]
    fn configured_agent_overrides_default_and_unknown_reads_default() {
        let policy = MemoryTrustConfig {
            default_agent_trust: 0.5,
            agents: [("vetted".to_string(), 0.9)].into(),
        }
        .policy();
        assert_eq!(policy.trust_of(Some("vetted")), 0.9);
        assert_eq!(policy.trust_of(Some("stranger")), 0.5);
    }

    #[test]
    fn weights_are_clamped_to_unit_interval() {
        let policy = MemoryTrustConfig {
            default_agent_trust: 7.0,
            agents: [("neg".to_string(), -3.0)].into(),
        }
        .policy();
        assert_eq!(policy.trust_of(Some("anyone")), 1.0);
        assert_eq!(policy.trust_of(Some("neg")), 0.0);
    }

    #[test]
    fn noop_detection() {
        assert!(AgentTrustPolicy::default().is_noop());
        assert!(MemoryTrustConfig::default().policy().is_noop());
        assert!(!MemoryTrustConfig {
            default_agent_trust: 0.6,
            agents: HashMap::new(),
        }
        .policy()
        .is_noop());
        assert!(!MemoryTrustConfig {
            default_agent_trust: 1.0,
            agents: [("x".to_string(), 0.4)].into(),
        }
        .policy()
        .is_noop());
    }

    #[test]
    fn config_defaults_are_the_identity() {
        let cfg: MemoryTrustConfig = serde_yaml::from_str("{}").unwrap();
        assert_eq!(cfg.default_agent_trust, 1.0);
        assert!(cfg.policy().is_noop());
    }
}
