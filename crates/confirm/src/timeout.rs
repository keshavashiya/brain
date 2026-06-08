//! Timeout and escalation policy.

use serde::{Deserialize, Serialize};

/// What to do when an approval request times out.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EscalationPolicy {
    /// Cancel the action, log the timeout.
    Abort,
    /// Alert user, then abort.
    NotifyAndAbort,
    /// Put the action in pending queue, retry later.
    Defer,
    /// Auto-approve — only for tier-lowered actions the user routinely approves.
    AutoApprove,
}

impl std::fmt::Display for EscalationPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EscalationPolicy::Abort => write!(f, "abort"),
            EscalationPolicy::NotifyAndAbort => write!(f, "notify_and_abort"),
            EscalationPolicy::Defer => write!(f, "defer"),
            EscalationPolicy::AutoApprove => write!(f, "auto_approve"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn any_policy() -> impl Strategy<Value = EscalationPolicy> {
        prop_oneof![
            Just(EscalationPolicy::Abort),
            Just(EscalationPolicy::NotifyAndAbort),
            Just(EscalationPolicy::Defer),
            Just(EscalationPolicy::AutoApprove),
        ]
    }

    // ── Property tests ────────────────────────────────────────────────
    //
    // `EscalationPolicy` carries two hand-maintained string representations —
    // the serde `rename_all = "snake_case"` discriminant (persisted in the
    // `escalation` column) and the `Display` label (shown to users). These pin
    // that the two never drift apart and that the persisted form round-trips.
    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

        /// The serialized form survives a JSON round-trip unchanged — the
        /// on-disk escalation token always parses back to the same variant.
        #[test]
        fn round_trips_through_json(p in any_policy()) {
            let json = serde_json::to_string(&p).unwrap();
            let back: EscalationPolicy = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(p, back);
        }

        /// The `Display` label agrees with the serde discriminant: the shown
        /// string is exactly the persisted token, so the two representations
        /// can never silently diverge.
        #[test]
        fn display_matches_serialized_token(p in any_policy()) {
            let json = serde_json::to_string(&p).unwrap();
            prop_assert_eq!(json, format!("\"{p}\""));
        }
    }
}
