//! # Agent Delegation
//!
//! Brain hands off subtasks to specialist agents — any CLI binary or
//! HTTP-driven agent — through a single generic trait. The orchestrator
//! only knows the trait; implementations live here.
//!
//! Core types:
//! - [`AgentDelegate`] — the trait every specialist agent implements
//! - [`AgentTask`] / [`AgentResult`] — request/response envelope
//! - [`AgentRegistry`] — looks up delegates by name, with aliasing
//! - [`SubprocessAgentDelegate`] — generic subprocess-backed adapter

pub mod definition;
pub mod discovery;
pub mod escalate;
pub mod registry;
pub mod subprocess;
pub mod traits;

pub use definition::{embedded_definitions, load_definitions, AgentDefinition};
pub use discovery::{
    DelegateDiscovery, DiscoveredBinary, DiscoveryStatus, PathScanner, DEFAULT_PROBE_TIMEOUT,
};
pub use escalate::{run_with_escalation, EscalationOutcome, EscalationPolicy};
pub use registry::{
    AgentOverride, AgentRegistry, AgentSource, DelegateOverrides, RegistryAgentStatus,
};
pub use subprocess::{SubprocessAgentConfig, SubprocessAgentDelegate};
pub use traits::{
    AgentCapabilities, AgentContext, AgentDelegate, AgentError, AgentResult, AgentTask,
    AgentTaskStatus, Artifact,
};
