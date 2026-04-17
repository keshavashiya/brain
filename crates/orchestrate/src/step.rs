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
    /// Execute — run a command in the sandbox.
    Execute { command: String, workdir: PathBuf },
    /// Test — run tests in the sandbox.
    Test { command: String, workdir: PathBuf },
    /// Review — present an artifact for human review.
    Review { artifact: String },
    /// Notify — send a notification via a channel.
    Notify { channel: String, message: String },
}
