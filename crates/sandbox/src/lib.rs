//! # Brain Sandbox Executor
//!
//! Isolated command execution with resource limits, filesystem allowlists,
//! network denial, and timeout enforcement.
//!
//! Phase 1a ships a stub executor (direct-exec, same privileges as daemon).
//! Phase 1b adds real isolation (setrlimit, cgroups, sandbox-exec).

pub mod allowlist;
pub mod tier;

pub use allowlist::{
    CredentialRef, ResourceUsage, SandboxCommand, SandboxError, SandboxExecutor, SandboxOutcome,
    StubSandbox,
};
pub use tier::ActionTier;
