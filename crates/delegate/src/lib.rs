//! # Agent Delegation (Phase 3)
//!
//! BrainOS hands off subtasks to specialist agents — Claude Code, Qwen,
//! or any subprocess/HTTP agent — through a single generic trait.
//! The orchestrator only knows the trait; implementations live here.
//!
//! Core types:
//! - [`AgentDelegate`] — the trait every specialist agent implements
//! - [`AgentTask`] / [`AgentResult`] — request/response envelope
//! - [`AgentRegistry`] — looks up delegates by name, with aliasing
//! - [`ClaudeCodeDelegate`] — subprocess adapter for the `claude` CLI

pub mod claude_code;
pub mod discovery;
pub mod escalate;
pub mod registry;
pub mod subprocess;
pub mod traits;

pub use claude_code::{ClaudeCodeConfig, ClaudeCodeDelegate};
pub use discovery::{
    default_fingerprints, AgentFingerprint, DelegateDiscovery, DiscoveredBinary, DiscoveryStatus,
    PathScanner, DEFAULT_PROBE_TIMEOUT,
};
pub use escalate::{run_with_escalation, EscalationOutcome, EscalationPolicy};
pub use registry::{
    AgentOverride, AgentRegistry, AgentSource, CustomAgentSpec, DelegateOverrides,
    RegistryAgentStatus,
};
pub use subprocess::{SubprocessAgentConfig, SubprocessAgentDelegate};
pub use traits::{
    AgentCapabilities, AgentContext, AgentDelegate, AgentError, AgentResult, AgentTask,
    AgentTaskStatus, Artifact, CredentialRef,
};
