//! Network connectivity as kernel state.
//!
//! [`Connectivity`] is a cheap, cloneable handle holding the kernel's current
//! view of the network: [`Online`](ConnectivityState::Online),
//! [`Degraded`](ConnectivityState::Degraded), or
//! [`Offline`](ConnectivityState::Offline). The serve loop's probe task is the
//! writer (it reaches the *already-configured* remote provider endpoints — no
//! new egress destinations); the signal pipeline is the reader (offline turns
//! route to a local model tier, web search degrades honestly, the capability
//! digest tells the reasoner where it stands).
//!
//! The default is `Online`: a processor without a probe loop (CLI one-shots,
//! tests, fully-local deployments with nothing remote to probe) behaves
//! exactly as before this state existed.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// The kernel's view of network reachability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectivityState {
    /// Every probed endpoint is reachable (or nothing remote is configured).
    Online,
    /// Some probed endpoints are reachable, others are not.
    Degraded,
    /// No probed endpoint is reachable.
    Offline,
}

impl ConnectivityState {
    /// Lowercase wire/display label (`"online" | "degraded" | "offline"`).
    pub fn as_str(self) -> &'static str {
        match self {
            ConnectivityState::Online => "online",
            ConnectivityState::Degraded => "degraded",
            ConnectivityState::Offline => "offline",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            2 => ConnectivityState::Offline,
            1 => ConnectivityState::Degraded,
            _ => ConnectivityState::Online,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            ConnectivityState::Online => 0,
            ConnectivityState::Degraded => 1,
            ConnectivityState::Offline => 2,
        }
    }
}

impl std::fmt::Display for ConnectivityState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shared, lock-free connectivity handle. Clones observe the same state.
#[derive(Debug, Clone)]
pub struct Connectivity {
    state: Arc<AtomicU8>,
}

impl Default for Connectivity {
    fn default() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(ConnectivityState::Online.as_u8())),
        }
    }
}

impl Connectivity {
    /// Current state.
    pub fn state(&self) -> ConnectivityState {
        ConnectivityState::from_u8(self.state.load(Ordering::SeqCst))
    }

    /// Whether the kernel currently believes nothing remote is reachable.
    pub fn is_offline(&self) -> bool {
        self.state() == ConnectivityState::Offline
    }

    /// Record a fresh observation. Returns the previous state when this
    /// observation *changed* the state (an edge worth surfacing), `None`
    /// while the state holds — the same edge discipline as the health and
    /// pressure trackers.
    pub fn set(&self, next: ConnectivityState) -> Option<ConnectivityState> {
        let prev = ConnectivityState::from_u8(self.state.swap(next.as_u8(), Ordering::SeqCst));
        (prev != next).then_some(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_online() {
        let c = Connectivity::default();
        assert_eq!(c.state(), ConnectivityState::Online);
        assert!(!c.is_offline());
    }

    #[test]
    fn set_returns_the_previous_state_only_on_an_edge() {
        let c = Connectivity::default();
        assert_eq!(c.set(ConnectivityState::Online), None, "no edge — silent");
        assert_eq!(
            c.set(ConnectivityState::Offline),
            Some(ConnectivityState::Online)
        );
        assert_eq!(c.set(ConnectivityState::Offline), None, "holding — silent");
        assert_eq!(
            c.set(ConnectivityState::Degraded),
            Some(ConnectivityState::Offline)
        );
        assert_eq!(c.state(), ConnectivityState::Degraded);
    }

    #[test]
    fn clones_share_one_state() {
        let a = Connectivity::default();
        let b = a.clone();
        a.set(ConnectivityState::Offline);
        assert!(b.is_offline());
    }

    #[test]
    fn labels_are_lowercase() {
        assert_eq!(ConnectivityState::Online.as_str(), "online");
        assert_eq!(ConnectivityState::Degraded.as_str(), "degraded");
        assert_eq!(ConnectivityState::Offline.to_string(), "offline");
    }
}
