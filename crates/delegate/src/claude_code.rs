//! ClaudeCodeDelegate — thin wrapper over [`SubprocessAgentDelegate`]
//! configured for Anthropic's `claude` CLI (`claude -p <prompt>`).
//!
//! The CLI accepts a one-shot prompt via `-p` and returns the agent's
//! final output on stdout. We default to stdin handoff (`claude -p -`)
//! so long prompts don't hit argv size limits and so we avoid shell
//! escaping hazards.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::subprocess::{SubprocessAgentConfig, SubprocessAgentDelegate};
use crate::traits::{
    AgentCapabilities, AgentDelegate, AgentError, AgentResult, AgentTask, AgentTaskStatus,
};

/// User-facing configuration for a Claude Code delegate. Kept separate
/// from the raw subprocess config so the bootstrap layer doesn't have
/// to care about stdin/args plumbing.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Registered name (default: `"claude-code"`).
    pub name: String,
    /// Binary name or absolute path. Defaults to `"claude"` — resolved
    /// via `$PATH` at spawn time.
    pub binary: String,
    /// Extra args appended after `-p -` (e.g. `--model claude-sonnet-4-6`).
    pub extra_args: Vec<String>,
    /// Default working directory. Task-level `workdir` wins when set.
    pub workdir: Option<PathBuf>,
    /// Declared capabilities. Defaults to code-edit + rust/typescript.
    pub capabilities: AgentCapabilities,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            name: "claude-code".to_string(),
            binary: "claude".to_string(),
            extra_args: Vec::new(),
            workdir: None,
            capabilities: AgentCapabilities {
                tags: vec!["code-edit".to_string(), "plan".to_string()],
                languages: vec!["rust".to_string(), "typescript".to_string()],
                max_concurrency: 1,
                needs_network: true,
            },
        }
    }
}

pub struct ClaudeCodeDelegate {
    inner: Arc<SubprocessAgentDelegate>,
    name: String,
    capabilities: AgentCapabilities,
}

impl ClaudeCodeDelegate {
    pub fn new(config: ClaudeCodeConfig) -> Self {
        // `claude -p -` reads the prompt from stdin.
        let mut args = vec!["-p".to_string(), "-".to_string()];
        args.extend(config.extra_args.iter().cloned());

        let sub_cfg = SubprocessAgentConfig {
            name: config.name.clone(),
            binary: config.binary,
            args,
            workdir: config.workdir,
            capabilities: config.capabilities.clone(),
            prompt_via_stdin: true,
        };

        Self {
            inner: Arc::new(SubprocessAgentDelegate::new(sub_cfg)),
            name: config.name,
            capabilities: config.capabilities,
        }
    }
}

#[async_trait]
impl AgentDelegate for ClaudeCodeDelegate {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> AgentCapabilities {
        self.capabilities.clone()
    }

    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        self.inner.delegate(task).await
    }

    /// Probe by running `<binary> --version` with a short timeout —
    /// confirms the CLI is installed without consuming API quota.
    async fn health_check(&self) -> bool {
        use tokio::process::Command;
        use tokio::time::{timeout, Duration};

        let probe = Command::new(self.inner.binary()).arg("--version").output();
        matches!(timeout(Duration::from_secs(5), probe).await, Ok(Ok(o)) if o.status.success())
    }

    async fn status(&self, _task_id: &str) -> Result<AgentTaskStatus, AgentError> {
        Ok(AgentTaskStatus::Running)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_registers_as_claude_code() {
        let cfg = ClaudeCodeConfig::default();
        assert_eq!(cfg.name, "claude-code");
        assert_eq!(cfg.binary, "claude");
        assert!(cfg.capabilities.needs_network);
    }

    #[test]
    fn new_delegate_exposes_name() {
        let d = ClaudeCodeDelegate::new(ClaudeCodeConfig::default());
        assert_eq!(d.name(), "claude-code");
    }
}
