//! Rolling consumption ledger and budget trait.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use audit::{ActionTier, AuditEntry, AuditTrail};
use chrono::Utc;
use rusqlite::params;
use serde::{Deserialize, Serialize};
use storage::SqlitePool;
use thiserror::Error;
use tracing;
use uuid::Uuid;

use super::policy::BudgetPolicy;

/// Type of resource being budgeted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResourceKind {
    LlmInputTokens,
    LlmOutputTokens,
    ApiCall { endpoint: String },
    SandboxWallClockMs,
    AgentDelegation { agent: String },
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceKind::LlmInputTokens => write!(f, "llm_input_tokens"),
            ResourceKind::LlmOutputTokens => write!(f, "llm_output_tokens"),
            ResourceKind::ApiCall { endpoint } => write!(f, "api_call:{endpoint}"),
            ResourceKind::SandboxWallClockMs => write!(f, "sandbox_wall_clock_ms"),
            ResourceKind::AgentDelegation { agent } => write!(f, "agent_delegation:{agent}"),
        }
    }
}

/// Decision from budget check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetDecision {
    /// Under ceiling, proceed.
    Allowed,
    /// Over 50% or 80%, log + notify.
    Warn { consumed_pct: f32 },
    /// At or over 100%, requires re-approval.
    Exceeded { ceiling: u64, consumed: u64 },
}

/// Current consumption snapshot.
///
/// `*_limits` carry the configured token ceilings for each `provider:resource`
/// so callers can render the budget *envelope* (used / limit), not just raw
/// usage. Only bounded token resources are included — unset (0) and unbounded
/// (`u64::MAX`) ceilings are omitted.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BudgetStatus {
    pub hourly_consumption: HashMap<String, u64>,
    pub daily_consumption: HashMap<String, u64>,
    pub hourly_limits: HashMap<String, u64>,
    pub daily_limits: HashMap<String, u64>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Error)]
pub enum BudgetError {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),
    #[error("Budget exceeded: {0}")]
    Exceeded(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Provider not configured: {0}")]
    ProviderNotFound(String),
    #[error("Policy error: {0}")]
    Policy(#[from] super::policy::PolicyError),
}

/// Budget enforcement trait.
#[async_trait]
pub trait CostBudget: Send + Sync {
    /// Check whether `units` of `resource` on `provider` can be spent.
    /// Does not yet record the spend.
    async fn check(
        &self,
        provider: &str,
        resource: &ResourceKind,
        units: u64,
    ) -> Result<BudgetDecision, BudgetError>;

    /// Record actual consumption after a call completes.
    async fn record(
        &self,
        provider: &str,
        resource: &ResourceKind,
        units: u64,
    ) -> Result<(), BudgetError>;

    /// Current consumption snapshot.
    async fn status(&self) -> Result<BudgetStatus, BudgetError>;
}

/// SQLite-backed budget implementation.
pub struct SqliteBudget {
    db: SqlitePool,
    policy: BudgetPolicy,
    audit: Option<Arc<dyn AuditTrail>>,
}

impl SqliteBudget {
    pub fn new(db: SqlitePool, policy: BudgetPolicy) -> Self {
        Self {
            db,
            policy,
            audit: None,
        }
    }

    /// Attach an audit trail so budget breaches produce audit entries.
    pub fn with_audit(mut self, audit: Arc<dyn AuditTrail>) -> Self {
        self.audit = Some(audit);
        self
    }

    pub fn ensure_tables(&self) -> Result<(), BudgetError> {
        self.db.with_conn(|conn| {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS budget_ledger (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    provider    TEXT NOT NULL,
                    resource    TEXT NOT NULL,
                    units       INTEGER NOT NULL,
                    hour_bucket TEXT NOT NULL,
                    day_bucket  TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_budget_hour
                    ON budget_ledger(provider, hour_bucket, resource);

                CREATE INDEX IF NOT EXISTS idx_budget_day
                    ON budget_ledger(provider, day_bucket, resource);
                "#,
            )?;
            Ok(())
        })?;
        Ok(())
    }

    fn get_consumption(
        &self,
        provider: &str,
        resource: &ResourceKind,
        bucket: &str,
        column: &str,
    ) -> Result<u64, BudgetError> {
        let resource_str = resource.to_string();
        let total: Option<i64> = self
            .db
            .with_conn(|conn| {
                let result = conn.query_row(
                    &format!(
                        "SELECT COALESCE(SUM(units), 0) FROM budget_ledger \
                     WHERE provider = ? AND resource = ? AND {column} = ?",
                    ),
                    [provider, &resource_str, bucket],
                    |row| row.get(0),
                )?;
                Ok(result)
            })
            .map_err(BudgetError::Storage)?;
        Ok(total.unwrap_or(0) as u64)
    }

    fn current_hour_bucket() -> String {
        Utc::now().format("%Y-%m-%dT%H:00:00Z").to_string()
    }

    fn current_day_bucket() -> String {
        Utc::now().format("%Y-%m-%d").to_string()
    }
}

fn derive_decision(consumed: u64, units: u64, ceiling: u64) -> BudgetDecision {
    // `u64::MAX` and `0` are both "no limit" sentinels: MAX is an explicitly
    // unbounded ceiling, 0 is an *unset* one (what `get_ceiling` returns for a
    // provider with no configured token cap, and what `status()` omits from the
    // limits envelope). Both must short-circuit to Allowed *before* the
    // projected-vs-ceiling check — otherwise a 0 ceiling would make `projected
    // >= 0` (always true for u64) deny every request, even a 0-unit one.
    if ceiling == u64::MAX || ceiling == 0 {
        return BudgetDecision::Allowed;
    }
    let projected = consumed.saturating_add(units);
    if projected >= ceiling {
        return BudgetDecision::Exceeded { ceiling, consumed };
    }
    let pct = (projected as f32 / ceiling as f32) * 100.0;
    if pct >= 50.0 {
        BudgetDecision::Warn { consumed_pct: pct }
    } else {
        BudgetDecision::Allowed
    }
}

/// Order: `Exceeded` > `Warn` (higher pct wins) > `Allowed`.
fn stricter_decision(a: BudgetDecision, b: BudgetDecision) -> BudgetDecision {
    use BudgetDecision::*;
    match (a, b) {
        (Exceeded { ceiling, consumed }, _) | (_, Exceeded { ceiling, consumed }) => {
            Exceeded { ceiling, consumed }
        }
        (Warn { consumed_pct: x }, Warn { consumed_pct: y }) => Warn {
            consumed_pct: x.max(y),
        },
        (Warn { consumed_pct }, Allowed) | (Allowed, Warn { consumed_pct }) => {
            Warn { consumed_pct }
        }
        (Allowed, Allowed) => Allowed,
    }
}

#[async_trait]
impl CostBudget for SqliteBudget {
    async fn check(
        &self,
        provider: &str,
        resource: &ResourceKind,
        units: u64,
    ) -> Result<BudgetDecision, BudgetError> {
        let hour_bucket = Self::current_hour_bucket();
        let day_bucket = Self::current_day_bucket();

        let hour_consumed =
            self.get_consumption(provider, resource, &hour_bucket, "hour_bucket")?;
        let day_consumed = self.get_consumption(provider, resource, &day_bucket, "day_bucket")?;

        let hour_ceiling = self.policy.get_ceiling(provider, resource, "hourly")?;
        let day_ceiling = self.policy.get_ceiling(provider, resource, "daily")?;

        let hour_decision = derive_decision(hour_consumed, units, hour_ceiling);
        let day_decision = derive_decision(day_consumed, units, day_ceiling);

        let decision = stricter_decision(hour_decision, day_decision);

        match &decision {
            BudgetDecision::Exceeded { ceiling, consumed } => {
                tracing::warn!(
                    provider = %provider,
                    resource = %resource,
                    consumed = *consumed,
                    ceiling = *ceiling,
                    "budget exceeded"
                );
                if let Some(audit) = &self.audit {
                    let metadata = serde_json::json!({
                        "provider": provider,
                        "resource": resource.to_string(),
                        "units_requested": units,
                        "consumed": consumed,
                        "ceiling": ceiling,
                        "hour_consumed": hour_consumed,
                        "day_consumed": day_consumed,
                    });
                    let entry = AuditEntry::new(
                        format!("budget check ({provider}/{resource}, +{units} units)"),
                        "budget-exceeded",
                        format!("deny {provider} {resource}"),
                        ActionTier::External,
                    )
                    .with_source("budget")
                    .with_metadata(metadata);
                    if let Err(e) = audit.record(entry).await {
                        tracing::warn!(error = %e, "failed to record budget-exceeded audit entry");
                    }
                }
            }
            BudgetDecision::Warn { consumed_pct } => {
                if *consumed_pct >= 80.0 {
                    tracing::warn!(
                        provider = %provider,
                        resource = %resource,
                        consumed_pct = *consumed_pct,
                        "budget warning 80%"
                    );
                } else {
                    tracing::info!(
                        provider = %provider,
                        resource = %resource,
                        consumed_pct = *consumed_pct,
                        "budget warning 50%"
                    );
                }
            }
            BudgetDecision::Allowed => {}
        }

        Ok(decision)
    }

    async fn record(
        &self,
        provider: &str,
        resource: &ResourceKind,
        units: u64,
    ) -> Result<(), BudgetError> {
        let hour_bucket = Self::current_hour_bucket();
        let day_bucket = Self::current_day_bucket();

        self.db.with_conn(|conn| {
            conn.execute(
                r#"INSERT INTO budget_ledger (
                    id, timestamp, provider, resource, units, hour_bucket, day_bucket
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"#,
                params![
                    Uuid::new_v4().to_string(),
                    Utc::now().to_rfc3339(),
                    provider,
                    resource.to_string(),
                    units as i64,
                    hour_bucket,
                    day_bucket,
                ],
            )?;
            Ok(())
        })?;

        tracing::debug!(
            provider = %provider,
            resource = %resource,
            units = units,
            "budget consumption recorded"
        );
        Ok(())
    }

    async fn status(&self) -> Result<BudgetStatus, BudgetError> {
        let hour_bucket = Self::current_hour_bucket();
        let day_bucket = Self::current_day_bucket();

        let mut status = BudgetStatus::default();

        // Get hourly consumption per provider/resource
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT provider, resource, SUM(units) FROM budget_ledger \
                 WHERE hour_bucket = ? GROUP BY provider, resource",
            )?;
            let mut rows = stmt.query([&hour_bucket])?;

            while let Some(row) = rows.next()? {
                let provider: String = row.get(0)?;
                let resource: String = row.get(1)?;
                let units: i64 = row.get(2)?;
                let key = format!("{provider}:{resource}");
                status.hourly_consumption.insert(key, units as u64);
            }

            Ok(())
        })?;

        // Get daily consumption per provider/resource
        self.db.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT provider, resource, SUM(units) FROM budget_ledger \
                 WHERE day_bucket = ? GROUP BY provider, resource",
            )?;
            let mut rows = stmt.query([&day_bucket])?;

            while let Some(row) = rows.next()? {
                let provider: String = row.get(0)?;
                let resource: String = row.get(1)?;
                let units: i64 = row.get(2)?;
                let key = format!("{provider}:{resource}");
                status.daily_consumption.insert(key, units as u64);
            }

            Ok(())
        })?;

        // Surface the configured token envelope so a zero-usage status still
        // shows the limits the user is operating under. Only the bounded
        // token resources map cleanly onto the `provider:resource` key space;
        // cost- and delegation-based ceilings are left out to avoid the
        // misleading cost→token heuristic.
        for provider in self.policy.providers.keys() {
            for resource in [ResourceKind::LlmInputTokens, ResourceKind::LlmOutputTokens] {
                let key = format!("{provider}:{resource}");
                if let Ok(h) = self.policy.get_ceiling(provider, &resource, "hourly") {
                    if h != 0 && h != u64::MAX {
                        status.hourly_limits.insert(key.clone(), h);
                    }
                }
                if let Ok(d) = self.policy.get_ceiling(provider, &resource, "daily") {
                    if d != 0 && d != u64::MAX {
                        status.daily_limits.insert(key, d);
                    }
                }
            }
        }

        Ok(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_budget() -> SqliteBudget {
        let pool = storage::SqlitePool::open_memory().unwrap();
        let policy = BudgetPolicy::default();
        let budget = SqliteBudget::new(pool, policy);
        budget.ensure_tables().unwrap();
        budget
    }

    #[tokio::test]
    async fn test_check_and_record() {
        let budget = test_budget();
        let resource = ResourceKind::LlmInputTokens;

        let decision = budget.check("openai", &resource, 1000).await.unwrap();
        assert!(matches!(decision, BudgetDecision::Allowed));

        budget.record("openai", &resource, 1000).await.unwrap();

        let status = budget.status().await.unwrap();
        assert!(!status.hourly_consumption.is_empty());
    }

    #[tokio::test]
    async fn status_surfaces_configured_token_limits() {
        // The default policy bounds openai input/output tokens, so a
        // zero-usage status still reports the envelope; providers with no
        // token ceiling (claude-code, sandbox) are omitted.
        let budget = test_budget();
        let status = budget.status().await.unwrap();

        assert_eq!(
            status.hourly_limits.get("openai:llm_input_tokens").copied(),
            Some(500_000)
        );
        assert_eq!(
            status.daily_limits.get("openai:llm_input_tokens").copied(),
            Some(500_000 * 24)
        );
        assert!(!status
            .hourly_limits
            .contains_key("claude-code:llm_input_tokens"));
        // No spend recorded yet.
        assert!(status.hourly_consumption.is_empty());
    }

    #[tokio::test]
    async fn test_budget_warning() {
        let mut budget = test_budget();
        // Set a low ceiling for testing
        budget
            .policy
            .set_ceiling("openai", &ResourceKind::LlmInputTokens, "hourly", 2000);

        let resource = ResourceKind::LlmInputTokens;

        // Record 1000 (50% of 2000)
        budget.record("openai", &resource, 1000).await.unwrap();

        // Check for another 1000 → should warn at 100%
        let decision = budget.check("openai", &resource, 1000).await.unwrap();
        assert!(matches!(
            decision,
            BudgetDecision::Exceeded {
                ceiling: 2000,
                consumed: 1000
            }
        ));
    }
}
