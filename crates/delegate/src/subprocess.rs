//! Generic subprocess-backed agent delegate.
//!
//! Spawns a configured binary with templated args, streams the task spec
//! through stdin, captures stdout/stderr, and returns an [`AgentResult`].
//! [`ClaudeCodeDelegate`] is a thin wrapper over this for the `claude`
//! CLI; bespoke agents (custom shell scripts, other CLIs) can reuse
//! `SubprocessAgentDelegate` directly.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::traits::{
    AgentCapabilities, AgentDelegate, AgentError, AgentResult, AgentTask, AgentTaskStatus,
};

/// How many bytes of stdout/stderr we keep — beyond this we truncate
/// and annotate. Keeps delegates from blowing the orchestrator's
/// in-memory task state.
const CAPTURE_CAP: usize = 64 * 1024;

/// Configuration for a subprocess delegate.
#[derive(Debug, Clone)]
pub struct SubprocessAgentConfig {
    /// Logical name registered with the registry (`"claude-code"`).
    pub name: String,
    /// Binary to spawn. Resolved via `$PATH` unless absolute.
    pub binary: String,
    /// Arguments passed before the prompt. Use `{prompt}` or `{task_id}`
    /// in any element and they're substituted at spawn time; otherwise
    /// the prompt goes on stdin.
    pub args: Vec<String>,
    /// Optional working directory override. When `None`, the task's
    /// `workdir` is used, falling back to the current process cwd.
    pub workdir: Option<PathBuf>,
    /// Declared capabilities.
    pub capabilities: AgentCapabilities,
    /// If `true`, the rendered prompt is written to the child's stdin
    /// instead of being templated into `args`.
    pub prompt_via_stdin: bool,
}

impl SubprocessAgentConfig {
    pub fn new(name: impl Into<String>, binary: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            binary: binary.into(),
            args: Vec::new(),
            workdir: None,
            capabilities: AgentCapabilities::default(),
            prompt_via_stdin: true,
        }
    }

    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_workdir(mut self, workdir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(workdir.into());
        self
    }

    pub fn with_capabilities(mut self, caps: AgentCapabilities) -> Self {
        self.capabilities = caps;
        self
    }

    pub fn with_prompt_via_stdin(mut self, via_stdin: bool) -> Self {
        self.prompt_via_stdin = via_stdin;
        self
    }
}

pub struct SubprocessAgentDelegate {
    config: SubprocessAgentConfig,
}

impl SubprocessAgentDelegate {
    pub fn new(config: SubprocessAgentConfig) -> Self {
        Self { config }
    }

    /// Binary path as configured — used by layered delegates (e.g.
    /// `ClaudeCodeDelegate`) for health probes.
    pub fn binary(&self) -> &str {
        &self.config.binary
    }

    /// The prompt we'd hand to the underlying binary. Exposed so
    /// [`ClaudeCodeDelegate`] (and tests) can reuse the format.
    pub fn render_prompt(task: &AgentTask) -> String {
        let mut out = String::new();
        out.push_str("# Task\n");
        out.push_str(task.description.trim());
        out.push_str("\n\n");
        let ctx = task.context.render();
        if !ctx.is_empty() {
            out.push_str("# Context\n");
            out.push_str(&ctx);
            out.push_str("\n\n");
        }
        out.push_str("# Workdir\n");
        match &task.workdir {
            Some(p) => out.push_str(&p.display().to_string()),
            None => out.push_str("<inherit>"),
        }
        out.push('\n');
        out
    }

    fn build_args(&self, prompt: &str, task: &AgentTask) -> Vec<String> {
        self.config
            .args
            .iter()
            .map(|a| a.replace("{prompt}", prompt).replace("{task_id}", &task.id))
            .collect()
    }
}

#[async_trait]
impl AgentDelegate for SubprocessAgentDelegate {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn capabilities(&self) -> AgentCapabilities {
        self.config.capabilities.clone()
    }

    async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
        let started_at = Utc::now();
        let prompt = Self::render_prompt(&task);
        let args = self.build_args(&prompt, &task);

        let workdir = self
            .config
            .workdir
            .clone()
            .or_else(|| task.workdir.clone())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let mut cmd = Command::new(&self.config.binary);
        cmd.args(&args)
            .current_dir(&workdir)
            .stdin(if self.config.prompt_via_stdin {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        tracing::debug!(
            agent = %self.config.name,
            task_id = %task.id,
            binary = %self.config.binary,
            workdir = %workdir.display(),
            "Spawning subprocess delegate"
        );

        let mut child = cmd
            .spawn()
            .map_err(|e| AgentError::Launch(format!("spawn {}: {e}", self.config.binary)))?;

        if self.config.prompt_via_stdin {
            if let Some(stdin) = child.stdin.as_mut() {
                match stdin.write_all(prompt.as_bytes()).await {
                    Ok(()) => {}
                    // A child that exits before reading stdin (e.g. `false`,
                    // or a CLI that rejects the prompt) closes the pipe and
                    // our write lands on EPIPE. That's not our error to
                    // surface — the exit status below is the real signal.
                    Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                        tracing::debug!(
                            agent = %self.config.name,
                            task_id = %task.id,
                            "child closed stdin before prompt was fully written"
                        );
                    }
                    Err(e) => return Err(AgentError::Io(format!("writing stdin: {e}"))),
                }
            }
            // Drop stdin so the child gets EOF and can finish.
            drop(child.stdin.take());
        }

        let wait = async {
            child
                .wait_with_output()
                .await
                .map_err(|e| AgentError::Io(format!("waiting on {}: {e}", self.config.name)))
        };

        let output = match timeout(Duration::from_secs(task.timeout_secs), wait).await {
            Ok(res) => res?,
            Err(_) => {
                tracing::warn!(
                    agent = %self.config.name,
                    task_id = %task.id,
                    secs = task.timeout_secs,
                    "Delegate timed out"
                );
                return Err(AgentError::Timeout {
                    task_id: task.id.clone(),
                    secs: task.timeout_secs,
                });
            }
        };

        let completed_at = Utc::now();
        let stdout = truncate(String::from_utf8_lossy(&output.stdout).into_owned());
        let stderr = truncate(String::from_utf8_lossy(&output.stderr).into_owned());
        let code = output.status.code().unwrap_or(-1);

        let status = if output.status.success() {
            AgentTaskStatus::Succeeded
        } else {
            AgentTaskStatus::Failed
        };

        let summary = if status == AgentTaskStatus::Succeeded {
            first_line(&stdout, &task.description)
        } else {
            format!("{} failed (exit {})", self.config.name, code)
        };

        Ok(AgentResult {
            task_id: task.id,
            status,
            summary,
            artifacts: vec![],
            stdout,
            stderr,
            exit_code: Some(code),
            started_at,
            completed_at,
        })
    }
}

fn truncate(mut s: String) -> String {
    if s.len() <= CAPTURE_CAP {
        return s;
    }
    s.truncate(CAPTURE_CAP);
    s.push_str("\n…[truncated]");
    s
}

fn first_line(stdout: &str, fallback: &str) -> String {
    stdout
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(|l| l.to_string())
        .unwrap_or_else(|| format!("completed: {fallback}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::AgentContext;

    #[test]
    fn render_prompt_includes_description_and_context() {
        let task = AgentTask::new("add a test").with_context(
            AgentContext::new()
                .push("memory", "project uses clap 4")
                .push("conventions", "tabs, trailing newline"),
        );
        let rendered = SubprocessAgentDelegate::render_prompt(&task);
        assert!(rendered.starts_with("# Task"));
        assert!(rendered.contains("add a test"));
        assert!(rendered.contains("### memory"));
        assert!(rendered.contains("### conventions"));
        assert!(rendered.contains("# Workdir"));
    }

    #[test]
    fn truncate_respects_cap() {
        let s = "a".repeat(CAPTURE_CAP + 500);
        let out = truncate(s);
        assert!(out.ends_with("…[truncated]"));
        assert!(out.len() <= CAPTURE_CAP + "\n…[truncated]".len());
    }

    #[test]
    fn build_args_substitutes_placeholders() {
        let cfg = SubprocessAgentConfig::new("mock", "echo")
            .with_args(["--task", "{task_id}", "--prompt", "{prompt}"])
            .with_prompt_via_stdin(false);
        let delegate = SubprocessAgentDelegate::new(cfg);
        let task = AgentTask::new("hi");
        let args = delegate.build_args("PROMPT", &task);
        assert_eq!(args[0], "--task");
        assert_eq!(args[1], task.id);
        assert_eq!(args[2], "--prompt");
        assert_eq!(args[3], "PROMPT");
    }

    // End-to-end using `/bin/cat` (POSIX; skipped on Windows CI).
    #[cfg(unix)]
    #[tokio::test]
    async fn echoes_prompt_via_stdin() {
        let cfg = SubprocessAgentConfig::new("cat-agent", "/bin/cat");
        let delegate = SubprocessAgentDelegate::new(cfg);
        let task = AgentTask::new("say hello").with_timeout_secs(5);
        let result = delegate.delegate(task).await.unwrap();
        assert_eq!(result.status, AgentTaskStatus::Succeeded);
        assert!(result.stdout.contains("# Task"));
        assert!(result.stdout.contains("say hello"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn times_out_when_child_hangs() {
        let cfg = SubprocessAgentConfig::new("sleep-agent", "/bin/sleep").with_args(["30"]);
        let delegate = SubprocessAgentDelegate::new(cfg);
        let task = AgentTask::new("noop").with_timeout_secs(1);
        let err = delegate.delegate(task).await.unwrap_err();
        matches!(err, AgentError::Timeout { .. });
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn nonzero_exit_surfaces_as_failed_status() {
        let cfg = SubprocessAgentConfig::new("false-agent", "/usr/bin/false");
        let delegate = SubprocessAgentDelegate::new(cfg);
        let task = AgentTask::new("noop").with_timeout_secs(5);
        let result = delegate.delegate(task).await.unwrap();
        assert_eq!(result.status, AgentTaskStatus::Failed);
        assert_ne!(result.exit_code, Some(0));
    }
}
