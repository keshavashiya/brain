//! Budget warning levels and notification helpers.

use serde::{Deserialize, Serialize};

/// Warning level for budget consumption.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WarnLevel {
    /// Under 50% — no warning needed.
    Ok,
    /// Over 50% — soft warning, log it.
    Warning50,
    /// Over 80% — stronger warning, notify user.
    Warning80,
    /// At or over 100% — hard stop, requires re-approval.
    Exceeded,
}

impl WarnLevel {
    /// Determine warning level from consumption percentage.
    pub fn from_pct(pct: f32) -> Self {
        if pct >= 100.0 {
            WarnLevel::Exceeded
        } else if pct >= 80.0 {
            WarnLevel::Warning80
        } else if pct >= 50.0 {
            WarnLevel::Warning50
        } else {
            WarnLevel::Ok
        }
    }

    /// Whether this level requires user notification.
    pub fn requires_notification(self) -> bool {
        matches!(self, WarnLevel::Warning80 | WarnLevel::Exceeded)
    }

    /// Whether this level requires hard stop.
    pub fn requires_hard_stop(self) -> bool {
        matches!(self, WarnLevel::Exceeded)
    }
}

/// Budget warning event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetWarning {
    pub provider: String,
    pub resource: String,
    pub level: WarnLevel,
    pub consumed_pct: f32,
    pub ceiling: u64,
    pub consumed: u64,
}

impl BudgetWarning {
    pub fn new(
        provider: impl Into<String>,
        resource: impl Into<String>,
        level: WarnLevel,
        consumed_pct: f32,
        ceiling: u64,
        consumed: u64,
    ) -> Self {
        Self {
            provider: provider.into(),
            resource: resource.into(),
            level,
            consumed_pct,
            ceiling,
            consumed,
        }
    }

    /// Human-readable message.
    pub fn message(&self) -> String {
        match self.level {
            WarnLevel::Warning50 => format!(
                "Budget warning: {provider}:{resource} at {pct:.0}% ({consumed}/{ceiling})",
                provider = self.provider,
                resource = self.resource,
                pct = self.consumed_pct,
                consumed = self.consumed,
                ceiling = self.ceiling,
            ),
            WarnLevel::Warning80 => format!(
                "Budget warning: {provider}:{resource} at {pct:.0}% — approaching limit ({consumed}/{ceiling})",
                provider = self.provider,
                resource = self.resource,
                pct = self.consumed_pct,
                consumed = self.consumed,
                ceiling = self.ceiling,
            ),
            WarnLevel::Exceeded => format!(
                "Budget exceeded: {provider}:{resource} at {pct:.0}% — hard stop ({consumed}/{ceiling}). Requires re-approval.",
                provider = self.provider,
                resource = self.resource,
                pct = self.consumed_pct,
                consumed = self.consumed,
                ceiling = self.ceiling,
            ),
            WarnLevel::Ok => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Severity rank: Ok < Warning50 < Warning80 < Exceeded.
    fn rank(l: WarnLevel) -> u8 {
        match l {
            WarnLevel::Ok => 0,
            WarnLevel::Warning50 => 1,
            WarnLevel::Warning80 => 2,
            WarnLevel::Exceeded => 3,
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

        /// The level is monotone non-decreasing in the consumption
        /// percentage — more spend can only raise the alarm, never lower it.
        #[test]
        fn from_pct_is_monotone(a in -50.0f32..300.0, b in -50.0f32..300.0) {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            prop_assert!(rank(WarnLevel::from_pct(hi)) >= rank(WarnLevel::from_pct(lo)));
        }

        /// Each level owns an exact, half-open percentage band:
        /// (-∞,50) Ok, [50,80) Warning50, [80,100) Warning80, [100,∞) Exceeded.
        #[test]
        fn thresholds_partition_the_range(pct in -50.0f32..300.0) {
            let expected = if pct >= 100.0 {
                WarnLevel::Exceeded
            } else if pct >= 80.0 {
                WarnLevel::Warning80
            } else if pct >= 50.0 {
                WarnLevel::Warning50
            } else {
                WarnLevel::Ok
            };
            prop_assert_eq!(WarnLevel::from_pct(pct), expected);
        }

        /// A hard stop is required at exactly the Exceeded level and nowhere
        /// else; user notification fires at Warning80 and Exceeded. Both
        /// predicates are monotone in severity (a hard stop always notifies).
        #[test]
        fn stop_and_notify_track_severity(pct in -50.0f32..300.0) {
            let level = WarnLevel::from_pct(pct);
            prop_assert_eq!(level.requires_hard_stop(), level == WarnLevel::Exceeded);
            prop_assert_eq!(
                level.requires_notification(),
                matches!(level, WarnLevel::Warning80 | WarnLevel::Exceeded)
            );
            if level.requires_hard_stop() {
                prop_assert!(level.requires_notification());
            }
        }
    }

    /// Exact boundary values land in the higher band (`>=` thresholds).
    #[test]
    fn boundaries_round_up() {
        assert_eq!(WarnLevel::from_pct(49.999), WarnLevel::Ok);
        assert_eq!(WarnLevel::from_pct(50.0), WarnLevel::Warning50);
        assert_eq!(WarnLevel::from_pct(80.0), WarnLevel::Warning80);
        assert_eq!(WarnLevel::from_pct(100.0), WarnLevel::Exceeded);
    }
}
