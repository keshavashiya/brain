//! Task step types — individual units of work in a task plan.

use std::path::PathBuf;

use audit::ActionTier;
use serde::{Deserialize, Serialize};

/// A single executable step in a task plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// Unique step ID (UUID).
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// What kind of action this step performs.
    pub action: StepAction,
    /// Step IDs this depends on (must complete first).
    pub depends_on: Vec<String>,
    /// Required approval tier.
    pub tier: ActionTier,
    /// Estimated LLM token cost for budget pre-check.
    pub estimated_tokens: u64,
}

/// The kind of action a step performs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StepAction {
    /// Research — search memory and/or web.
    Research { query: String },
    /// Plan — produce a plan artifact (text output).
    Plan { output: String },
    /// Implement — delegate to an agent.
    Implement { spec: String, agent: String },
    /// Execute — run a command in the sandbox via direct argv (no shell).
    /// Use for simple `binary arg arg` commands. The first token must be
    /// on the per-binary allowlist. No pipes, redirects, $VAR, or
    /// PATH-dependent lookups beyond the sanitised env.
    Execute { command: String, workdir: PathBuf },
    /// Test — run tests in the sandbox via direct argv.
    Test { command: String, workdir: PathBuf },
    /// Shell — run a command via `sh -c "<command>"` so the system
    /// shell handles pipes, redirects, escaping, $VAR, and PATH
    /// resolution. Use for any non-trivial command. The per-binary
    /// allowlist is bypassed for the wrapped command (only `sh` is
    /// gated); rlimits + Seatbelt + timeout + the explicit
    /// forbidden-list still apply.
    Shell { command: String, workdir: PathBuf },
    /// Review — present an artifact for human review.
    Review { artifact: String },
    /// Notify — send a notification via a channel.
    Notify { channel: String, message: String },
}
