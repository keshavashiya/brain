//! Core trait + envelope types for agent delegation.

use std::path::PathBuf;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Reference to a credential that should be injected at execution time.
/// The delegate resolves this against the vault — raw values never
/// appear in `AgentTask`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRef {
    /// Vault key (e.g. `"github_token"`).
    pub key: String,
    /// Environment variable to expose the value under, if any.
    pub env: Option<String>,
}

/// Token-budgeted context passed to the delegate alongside the task spec.
/// Free-form fields so different agents can lift what they need — keys
/// like `"memory_facts"`, `"recent_episodes"`, `"project_conventions"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentContext {
    /// Human-readable context segments (key → content), already truncated
    /// by the caller to fit the agent's context window.
    pub segments: Vec<(String, String)>,
    /// Approximate token count used by the caller's budget tracker.
    pub tokens_used: u64,
}

impl AgentContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, key: impl Into<String>, content: impl Into<String>) -> Self {
        self.segments.push((key.into(), content.into()));
        self
    }

    /// Render as a single prompt-friendly string — each segment becomes a
    /// `### {key}` block. Keeps the layout predictable across delegates.
    pub fn render(&self) -> String {
        if self.segments.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for (key, content) in &self.segments {
            out.push_str("### ");
            out.push_str(key);
            out.push('\n');
            out.push_str(content.trim());
            out.push_str("\n\n");
        }
        out.trim_end().to_string()
    }
}

/// One unit of work handed to a delegate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub description: String,
    pub context: AgentContext,
    /// Working directory; the delegate should cd here before executing.
    pub workdir: Option<PathBuf>,
    /// Credentials to inject (never raw values — vault refs only).
    #[serde(default)]
    pub credentials: Vec<CredentialRef>,
    /// Hard ceiling in seconds. The delegate kills the underlying
    /// process/request at or before this mark.
    pub timeout_secs: u64,
}

impl AgentTask {
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            context: AgentContext::default(),
            workdir: None,
            credentials: Vec::new(),
            timeout_secs: 300,
        }
    }

    pub fn with_context(mut self, ctx: AgentContext) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_workdir(mut self, workdir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(workdir.into());
        self
    }

    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// An artifact the delegate produced — a file, a test result, a PR URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Classifier like `"file"`, `"test"`, `"url"`.
    pub kind: String,
    /// Opaque reference — path, URL, or JSON blob — interpreted by kind.
    pub reference: String,
    /// Optional summary (one line).
    #[serde(default)]
    pub summary: Option<String>,
}

/// Terminal status of a delegated task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTaskStatus {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

impl AgentTaskStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, AgentTaskStatus::Running)
    }
    pub fn is_success(self) -> bool {
        matches!(self, AgentTaskStatus::Succeeded)
    }
}

/// What the delegate came back with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub task_id: String,
    pub status: AgentTaskStatus,
    /// Agent's own summary of what happened (first-person, human-readable).
    pub summary: String,
    /// Structured artifacts (files changed, test runs, links).
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    /// Raw stdout, truncated by the delegate to avoid blowing context.
    #[serde(default)]
    pub stdout: String,
    /// Raw stderr, likewise truncated.
    #[serde(default)]
    pub stderr: String,
    /// Exit code for subprocess-backed delegates; `None` for HTTP agents.
    pub exit_code: Option<i32>,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
}

impl AgentResult {
    pub fn duration_ms(&self) -> i64 {
        (self.completed_at - self.started_at)
            .num_milliseconds()
            .max(0)
    }
}

/// Declarative statement of what an agent can do — the orchestrator uses
/// this to pick a delegate, and the registry uses it for search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentCapabilities {
    /// Tags like `"code-edit"`, `"plan"`, `"research"`.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Preferred languages/frameworks (`"rust"`, `"typescript"`).
    #[serde(default)]
    pub languages: Vec<String>,
    /// Maximum concurrent delegations (conservative default = 1).
    #[serde(default = "default_concurrency")]
    pub max_concurrency: u32,
    /// Whether this delegate needs network — informs sandbox policy.
    #[serde(default)]
    pub needs_network: bool,
}

fn default_concurrency() -> u32 {
    1
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("delegate '{0}' not found in registry")]
    NotFound(String),
    #[error("delegate failed to launch: {0}")]
    Launch(String),
    #[error("delegate timed out after {secs}s: {task_id}")]
    Timeout { task_id: String, secs: u64 },
    #[error("delegate returned non-zero exit {code}: {stderr}")]
    NonZeroExit { code: i32, stderr: String },
    #[error("delegate output unparseable: {0}")]
    Parse(String),
    #[error("delegate cancelled: {0}")]
    Cancelled(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("other: {0}")]
    Other(String),
}

/// The trait every specialist agent implements.
#[async_trait]
pub trait AgentDelegate: Send + Sync {
    /// Stable identifier registered with the registry.
    fn name(&self) -> &str;

    /// Declared capabilities — used by the orchestrator to pick a delegate.
    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::default()
    }

    /// Hand off a task. Blocks until the delegate finishes, fails, or
    /// hits `task.timeout_secs`.
    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError>;

    /// Best-effort cancellation. Default is no-op.
    async fn cancel(&self, _task_id: &str) -> Result<(), AgentError> {
        Ok(())
    }

    /// Best-effort status poll. Most implementations don't keep per-task
    /// state and return `Running` during `delegate` via other channels.
    async fn status(&self, _task_id: &str) -> Result<AgentTaskStatus, AgentError> {
        Ok(AgentTaskStatus::Running)
    }

    /// Cheap liveness probe — the registry calls this during startup and
    /// before routing to mark the delegate healthy/unhealthy. Default is
    /// "assume healthy" so stateless delegates don't have to implement it.
    async fn health_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_render_round_trips() {
        let ctx = AgentContext::new()
            .push("memory", "fact A\nfact B")
            .push("conventions", "use tabs");
        let rendered = ctx.render();
        assert!(rendered.contains("### memory"));
        assert!(rendered.contains("fact A"));
        assert!(rendered.contains("### conventions"));
        assert!(rendered.contains("use tabs"));
    }

    #[test]
    fn context_render_empty_returns_empty() {
        assert!(AgentContext::new().render().is_empty());
    }

    #[test]
    fn agent_task_new_generates_uuid() {
        let a = AgentTask::new("a");
        let b = AgentTask::new("b");
        assert_ne!(a.id, b.id);
        assert_eq!(a.timeout_secs, 300);
    }

    #[test]
    fn task_status_terminality() {
        assert!(!AgentTaskStatus::Running.is_terminal());
        assert!(AgentTaskStatus::Succeeded.is_terminal());
        assert!(AgentTaskStatus::Failed.is_terminal());
        assert!(AgentTaskStatus::TimedOut.is_terminal());
        assert!(AgentTaskStatus::Cancelled.is_terminal());
        assert!(AgentTaskStatus::Succeeded.is_success());
        assert!(!AgentTaskStatus::Failed.is_success());
    }
}
