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
