//! Budget policy configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ledger::ResourceKind;

#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("Provider not configured: {0}")]
    ProviderNotFound(String),
    #[error("No ceiling configured for {provider}:{resource} ({period})")]
    CeilingNotFound {
        provider: String,
        resource: String,
        period: String,
    },
}

/// Per-provider budget policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPolicy {
    pub providers: HashMap<String, ProviderPolicy>,
    pub per_action: PerActionPolicy,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self {
            providers: HashMap::from([
                (
                    "openai".to_string(),
                    ProviderPolicy {
                        hourly_input_tokens: 500_000,
                        hourly_output_tokens: 100_000,
                        daily_cost_usd_soft: 5.00,
                        daily_cost_usd_hard: 10.00,
                        hourly_delegations: None,
                    },
                ),
                (
                    "claude-code".to_string(),
                    ProviderPolicy {
                        hourly_input_tokens: 0,
                        hourly_output_tokens: 0,
                        daily_cost_usd_soft: 0.0,
                        daily_cost_usd_hard: 0.0,
                        hourly_delegations: Some(20),
                    },
                ),
                (
                    "sandbox".to_string(),
                    ProviderPolicy {
                        hourly_input_tokens: 0,
                        hourly_output_tokens: 0,
                        daily_cost_usd_soft: 0.0,
                        daily_cost_usd_hard: 0.0,
                        hourly_delegations: None,
                    },
                ),
            ]),
            per_action: PerActionPolicy {
                llm_tokens_max: 50_000,
                sandbox_wall_clock_ms_max: 300_000,
            },
        }
    }
}

impl BudgetPolicy {
    /// Load from YAML configuration.
    pub fn from_yaml(yaml: &str) -> Result<Self, PolicyError> {
        serde_yaml::from_str(yaml).map_err(|e| PolicyError::ProviderNotFound(e.to_string()))
    }

    /// Get the ceiling for a specific resource and period (`"hourly"` or `"daily"`).
    ///
    /// Returns `u64::MAX` when no ceiling applies for the given period (callers
    /// treat that as "unbounded" and skip the check).
    pub fn get_ceiling(
        &self,
        provider: &str,
        resource: &ResourceKind,
        period: &str,
    ) -> Result<u64, PolicyError> {
        let provider_policy = self
            .providers
            .get(provider)
            .ok_or_else(|| PolicyError::ProviderNotFound(provider.to_string()))?;

        // Daily ceilings default to 24× hourly for token/delegation resources.
        // ApiCall uses explicit cost-based daily ceilings.
        let scale = if period == "daily" { 24 } else { 1 };

        match resource {
            ResourceKind::LlmInputTokens => {
                Ok(provider_policy.hourly_input_tokens.saturating_mul(scale))
            }
            ResourceKind::LlmOutputTokens => {
                Ok(provider_policy.hourly_output_tokens.saturating_mul(scale))
            }
            ResourceKind::AgentDelegation { .. } => Ok(provider_policy
                .hourly_delegations
                .map(|n| n.saturating_mul(scale))
                .unwrap_or(u64::MAX)),
            ResourceKind::SandboxWallClockMs => Ok(u64::MAX), // Per-action policy
            ResourceKind::ApiCall { .. } => {
                // Soft ceiling reported for hourly, hard ceiling for daily.
                let cost = match period {
                    "daily" => provider_policy.daily_cost_usd_hard,
                    _ => provider_policy.daily_cost_usd_soft,
                };
                // Convert to token-equivalent (rough heuristic: $0.00001 per token)
                Ok((cost / 0.00001) as u64)
            }
        }
    }

    /// Set a custom ceiling (for testing or dynamic adjustment).
    pub fn set_ceiling(
        &mut self,
        provider: &str,
        resource: &ResourceKind,
        _period: &str,
        ceiling: u64,
    ) {
        let provider_policy = self
            .providers
            .entry(provider.to_string())
            .or_insert_with(|| ProviderPolicy {
                hourly_input_tokens: 0,
                hourly_output_tokens: 0,
                daily_cost_usd_soft: 0.0,
                daily_cost_usd_hard: 0.0,
                hourly_delegations: None,
            });

        match resource {
            ResourceKind::LlmInputTokens => provider_policy.hourly_input_tokens = ceiling,
            ResourceKind::LlmOutputTokens => provider_policy.hourly_output_tokens = ceiling,
            ResourceKind::AgentDelegation { .. } => {
                provider_policy.hourly_delegations = Some(ceiling)
            }
            _ => {}
        }
    }

    /// Get per-action ceiling for sandbox wall clock.
    pub fn sandbox_wall_clock_max_ms(&self) -> u64 {
        self.per_action.sandbox_wall_clock_ms_max
    }

    /// Get per-action ceiling for LLM tokens.
    pub fn llm_tokens_max(&self) -> u64 {
        self.per_action.llm_tokens_max
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPolicy {
    #[serde(default)]
    pub hourly_input_tokens: u64,
    #[serde(default)]
    pub hourly_output_tokens: u64,
    #[serde(default)]
    pub daily_cost_usd_soft: f64,
    #[serde(default)]
    pub daily_cost_usd_hard: f64,
    pub hourly_delegations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerActionPolicy {
    pub llm_tokens_max: u64,
    pub sandbox_wall_clock_ms_max: u64,
}
