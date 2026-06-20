//! # Brain Sandbox Executor
//!
//! Isolated command execution with resource limits, filesystem allowlists,
//! network denial, and timeout enforcement.
//!
//! Production code should use [`IsolatedSandbox`] — it applies `setrlimit`
//! via a pre-exec hook, enforces the binary allowlist, kills the process
//! group on timeout, and layers platform isolation (macOS `sandbox-exec`,
//! Linux namespaces). [`StubSandbox`] is retained for tests and scripted
//! demos where isolation is intentionally skipped.

pub mod allowlist;
pub mod harden;
pub mod isolated;
pub mod tier;

pub use allowlist::{
    CredentialRef, ResourceUsage, SandboxCommand, SandboxError, SandboxExecutor, SandboxOutcome,
    StubSandbox,
};
pub use harden::{hardened_stdio_command, StdioHardening};
pub use isolated::{IsolatedSandbox, SandboxLimits};
pub use tier::ActionTier;
