//! Rollback coordinate tracking.

use serde::{Deserialize, Serialize};

/// A plan for rolling back an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub description: String,
    pub steps: Vec<RollbackStep>,
    pub reversible: bool,
}

/// A single rollback step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    pub action: String,
    pub command: Option<String>,
    pub description: String,
}

impl RollbackPlan {
    pub fn new(description: impl Into<String>, steps: Vec<RollbackStep>) -> Self {
        let reversible = !steps.is_empty();
        Self {
            description: description.into(),
            steps,
            reversible,
        }
    }

    pub fn irreversible(reason: impl Into<String>) -> Self {
        Self {
            description: reason.into(),
            steps: Vec::new(),
            reversible: false,
        }
    }
}
