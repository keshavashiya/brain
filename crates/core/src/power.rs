//! Power source as kernel state.
//!
//! [`Power`] is a cheap, cloneable handle holding the kernel's current view
//! of the power source: [`External`](PowerState::External) or
//! [`Battery`](PowerState::Battery). The serve loop's probe task is the
//! writer (a platform query — `pmset` on macOS, `/sys/class/power_supply`
//! on Linux — no new dependency); heavy maintenance loops are the readers:
//! consolidation and graph sweeps hold while on battery (config-overridable)
//! and resume when external power returns, and the capability digest tells
//! the reasoner the machine is running on battery.
//!
//! The default is `External`: a processor without a probe loop (CLI
//! one-shots, tests, desktops with no battery to probe) behaves exactly as
//! before this state existed.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// The kernel's view of the machine's power source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Wall power — mains/AC online, or no battery hardware at all.
    External,
    /// Running on battery.
    Battery,
}

impl PowerState {
    /// Lowercase wire/display label (`"external" | "battery"`).
    pub fn as_str(self) -> &'static str {
        match self {
            PowerState::External => "external",
            PowerState::Battery => "battery",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            1 => PowerState::Battery,
            _ => PowerState::External,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            PowerState::External => 0,
            PowerState::Battery => 1,
        }
    }
}

impl std::fmt::Display for PowerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shared, lock-free power-source handle. Clones observe the same state.
#[derive(Debug, Clone)]
pub struct Power {
    state: Arc<AtomicU8>,
}

impl Default for Power {
    fn default() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(PowerState::External.as_u8())),
        }
    }
}

impl Power {
    /// Current state.
    pub fn state(&self) -> PowerState {
        PowerState::from_u8(self.state.load(Ordering::SeqCst))
    }

    /// Whether the kernel currently believes it is running on battery.
    pub fn is_battery(&self) -> bool {
        self.state() == PowerState::Battery
    }

    /// Record a fresh observation. Returns the previous state when this
    /// observation *changed* the state (an edge worth surfacing), `None`
    /// while the state holds — the same edge discipline as
    /// [`Connectivity`](crate::Connectivity) and the health trackers.
    pub fn set(&self, next: PowerState) -> Option<PowerState> {
        let prev = PowerState::from_u8(self.state.swap(next.as_u8(), Ordering::SeqCst));
        (prev != next).then_some(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_external() {
        let p = Power::default();
        assert_eq!(p.state(), PowerState::External);
        assert!(!p.is_battery());
    }

    #[test]
    fn set_returns_the_previous_state_only_on_an_edge() {
        let p = Power::default();
        assert_eq!(p.set(PowerState::External), None, "no edge — silent");
        assert_eq!(p.set(PowerState::Battery), Some(PowerState::External));
        assert_eq!(p.set(PowerState::Battery), None, "holding — silent");
        assert_eq!(p.set(PowerState::External), Some(PowerState::Battery));
    }

    #[test]
    fn clones_share_one_state() {
        let a = Power::default();
        let b = a.clone();
        a.set(PowerState::Battery);
        assert!(b.is_battery());
    }

    #[test]
    fn labels_are_lowercase() {
        assert_eq!(PowerState::External.as_str(), "external");
        assert_eq!(PowerState::Battery.to_string(), "battery");
    }
}
