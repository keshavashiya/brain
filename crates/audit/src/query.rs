//! Query specification for audit entries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::schema::{ActionTier, AuditOutcome};

/// Specification for querying audit entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditQuerySpec {
    pub since: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
    pub source: Option<String>,
    pub tier: Option<ActionTier>,
    pub outcome: Option<AuditOutcome>,
    pub limit: Option<usize>,
}

impl AuditQuerySpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn since(mut self, since: DateTime<Utc>) -> Self {
        self.since = Some(since);
        self
    }

    pub fn before(mut self, before: DateTime<Utc>) -> Self {
        self.before = Some(before);
        self
    }

    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn tier(mut self, tier: ActionTier) -> Self {
        self.tier = Some(tier);
        self
    }

    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = Some(outcome);
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Last N entries.
    pub fn last(n: usize) -> Self {
        Self {
            limit: Some(n),
            ..Default::default()
        }
    }

    /// Entries from today.
    pub fn today() -> Self {
        let today = Utc::now().date_naive().and_hms_opt(0, 0, 0).unwrap();
        let since = DateTime::<Utc>::from_naive_utc_and_offset(today, Utc);
        Self {
            since: Some(since),
            ..Default::default()
        }
    }
}
