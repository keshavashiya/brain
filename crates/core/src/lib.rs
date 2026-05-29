//! # Brain Core
//!
//! Orchestrator that wires all brain subsystems together.
//!
//! Provides:
//! - Configuration management (figment + YAML)
//! - Subsystem initialization and dependency injection
//! - Message pipeline: Thalamus → Hippocampus → Cortex → Response
//! - Error handling and graceful degradation

pub mod auth;
pub mod config;
pub mod cors;
pub mod metrics;
pub mod security;

pub use auth::{check_auth, AuthResult};
pub use config::{
    AccessConfig, AgentEntry, AgentsConfig, ApiKeyConfig, BrainConfig, ClientRateLimitConfig,
    DeliveryConfig, LlmConfig, LogFormat, LogRotation, LoggingConfig, ProviderEntry,
};
pub use security::ActionTier;

/// Standard timeout constants for HTTP clients across Brain OS.
pub mod timeouts {
    use std::time::Duration;

    pub const EMBEDDING_OLLAMA: Duration = Duration::from_secs(120);
    pub const EMBEDDING_OPENAI: Duration = Duration::from_secs(60);
    /// 90s caps chat-style requests so a flaky upstream LLM can't pin
    /// the inbound pipeline (Telegram/HTTP) indefinitely. Long-running
    /// orchestration steps that legitimately need more time should run
    /// the LLM call themselves with a wider per-request timeout.
    pub const LLM_GENERATE: Duration = Duration::from_secs(90);
    pub const HEALTH_CHECK: Duration = Duration::from_secs(2);
    pub const DAEMON_SETUP: Duration = Duration::from_secs(30);
    pub const STATUS_CHECK: Duration = Duration::from_secs(2);
}

/// Normalize a keyword for matching: trim non-alphanumeric edges, lowercase.
pub fn normalize_keyword(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}
