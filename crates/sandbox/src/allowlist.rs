//! Filesystem path allowlists and command validation.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing;

use super::tier::ActionTier;

/// Resource usage metrics from a sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    pub cpu_time_ms: u64,
    pub memory_peak_bytes: u64,
    pub disk_io_bytes: u64,
}

/// Credential reference for injection at execution time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialRef {
    pub tool: String,
    pub key: String,
}

/// Command to execute in the sandbox.
///
/// Two execution modes:
/// - **Argv** (default, `shell_mode = false`): the binary is looked up via
///   the per-binary allowlist and exec'd directly with no shell. Safe but
///   restrictive — no pipes, redirects, or PATH lookups beyond the
///   sanitised env.
/// - **Shell** (`shell_mode = true`, set via [`SandboxCommand::shell`]):
///   the command is wrapped in `sh -c "<command>"` so the system shell
///   handles pipes, redirects, escaping, and PATH resolution. The
///   per-binary allowlist is bypassed for the wrapped command — gating
///   shifts to the daemon's ambient PATH plus the existing rlimits +
///   Seatbelt + timeout. Use this for any non-trivial command the LLM
///   produced that argv mode can't run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxCommand {
    pub binary: String,
    pub args: Vec<String>,
    pub workdir: PathBuf,
    pub env: std::collections::HashMap<String, String>,
    pub tier: ActionTier,
    pub timeout: Duration,
    /// `true` when this command should be wrapped in `sh -c` and inherit
    /// ambient PATH. Set by [`SandboxCommand::shell`]; the per-binary
    /// allowlist is bypassed in this mode (only `sh` itself is gated).
    #[serde(default)]
    pub shell_mode: bool,
}

impl SandboxCommand {
    pub fn new(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            binary: binary.into(),
            args,
            workdir: std::env::current_dir().unwrap_or_default(),
            env: std::collections::HashMap::new(),
            tier: ActionTier::Execute,
            timeout: Duration::from_secs(300),
            shell_mode: false,
        }
    }

    /// Build a shell-wrapped command. The string is passed verbatim to
    /// `sh -c`, so it can contain pipes, redirects, $VAR expansion,
    /// quoted arguments, and any other shell construct. The sandbox
    /// still applies rlimits, network deny (macOS Seatbelt), and the
    /// configured timeout — but does NOT enforce a per-binary allowlist
    /// on what the shell ends up calling.
    pub fn shell(command: impl Into<String>) -> Self {
        Self {
            binary: "sh".to_string(),
            args: vec!["-c".to_string(), command.into()],
            workdir: std::env::current_dir().unwrap_or_default(),
            env: std::collections::HashMap::new(),
            tier: ActionTier::Execute,
            timeout: Duration::from_secs(300),
            shell_mode: true,
        }
    }

    pub fn with_workdir(mut self, workdir: PathBuf) -> Self {
        self.workdir = workdir;
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn with_tier(mut self, tier: ActionTier) -> Self {
        self.tier = tier;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Outcome from a sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxOutcome {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub duration: Duration,
    pub resource_usage: ResourceUsage,
    pub interrupted: bool,
}

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("Execution failed: {0}")]
    Execution(String),
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    #[error("Command not allowed: {0}")]
    Forbidden(String),
    #[error("Path not in allowlist: {0}")]
    PathNotAllowed(String),
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Sandbox executor trait.
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Execute a command in the sandbox.
    async fn run(&self, command: SandboxCommand) -> Result<SandboxOutcome, SandboxError>;

    /// Execute a command with credential injection.
    async fn run_with_credentials(
        &self,
        command: SandboxCommand,
        creds: Vec<CredentialRef>,
    ) -> Result<SandboxOutcome, SandboxError>;
}

/// Stub sandbox executor. Runs commands directly with no isolation —
/// same privileges as daemon. Clearly labeled as un-sandboxed.
pub struct StubSandbox {
    allowed_paths: HashSet<PathBuf>,
    forbidden_commands: HashSet<String>,
}

impl Default for StubSandbox {
    fn default() -> Self {
        Self {
            allowed_paths: HashSet::from([PathBuf::from("/tmp"), PathBuf::from(".")]),
            forbidden_commands: HashSet::from([
                "rm -rf /".to_string(),
                "dd".to_string(),
                "mkfs".to_string(),
            ]),
        }
    }
}

impl StubSandbox {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_allowed_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.allowed_paths.extend(paths);
        self
    }

    pub fn with_forbidden_commands(mut self, commands: Vec<String>) -> Self {
        for cmd in commands {
            self.forbidden_commands.insert(cmd);
        }
        self
    }

    fn is_forbidden(&self, command: &SandboxCommand) -> Option<String> {
        // Check if binary or full command is forbidden
        let full_cmd = format!("{} {}", command.binary, command.args.join(" "));

        for forbidden in &self.forbidden_commands {
            if full_cmd.contains(forbidden) || command.binary.contains(forbidden) {
                return Some(format!("forbidden command: {forbidden}"));
            }
        }

        // Block cloud metadata IPs
        for arg in &command.args {
            if arg.contains("169.254.169.254") {
                return Some("cloud metadata IP blocked".to_string());
            }
        }

        None
    }

    fn is_path_allowed(&self, workdir: &PathBuf) -> Result<(), String> {
        // Stub allows all paths but logs warnings.
        // The real sandbox (`IsolatedSandbox`) enforces strict allowlists.
        if !self.allowed_paths.iter().any(|p| workdir.starts_with(p)) {
            tracing::warn!(path = ?workdir, "path not in allowlist (stub permits)");
        }
        Ok(())
    }
}

#[async_trait]
impl SandboxExecutor for StubSandbox {
    async fn run(&self, command: SandboxCommand) -> Result<SandboxOutcome, SandboxError> {
        // Validate command
        if let Some(reason) = self.is_forbidden(&command) {
            return Err(SandboxError::Forbidden(reason));
        }

        self.is_path_allowed(&command.workdir)
            .map_err(SandboxError::PathNotAllowed)?;

        tracing::warn!(
            binary = %command.binary,
            args = ?command.args,
            workdir = ?command.workdir,
            "stub sandbox executing command (NO ISOLATION)"
        );

        let start = std::time::Instant::now();

        // Execute the command
        let mut cmd = tokio::process::Command::new(&command.binary);
        cmd.args(&command.args);
        cmd.current_dir(&command.workdir);

        for (k, v) in &command.env {
            cmd.env(k, v);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        // On unix, put the child in its own process group so a timeout can
        // kill the whole group (child plus any grandchildren it forked). With
        // the default PID-only kill, grandchildren leak after timeout.
        #[cfg(unix)]
        cmd.process_group(0);

        let child = cmd.spawn().map_err(SandboxError::Io)?;

        #[cfg(unix)]
        let child_pid = child.id();

        // Apply timeout. If it fires, send SIGKILL to the child's process
        // group before returning so no subprocesses leak.
        let output = match tokio::time::timeout(command.timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(SandboxError::Io(e)),
            Err(_) => {
                #[cfg(unix)]
                if let Some(pid) = child_pid {
                    // SAFETY: killpg with SIGKILL is safe to call; pid is the
                    // child leader, and we set process_group(0) above so the
                    // child's PGID equals its PID.
                    unsafe {
                        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
                    }
                    tracing::warn!(pid = pid, "sandbox timeout: sent SIGKILL to process group");
                }
                return Err(SandboxError::Timeout(command.timeout));
            }
        };

        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout)
            .chars()
            .take(1_048_576) // Cap at 1MB
            .collect();
        let stderr = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(1_048_576)
            .collect();

        let exit_code = output.status.code().unwrap_or(-1);

        tracing::info!(
            exit_code = exit_code,
            duration_ms = duration.as_millis(),
            "stub sandbox execution complete"
        );

        Ok(SandboxOutcome {
            stdout,
            stderr,
            exit_code,
            duration,
            resource_usage: ResourceUsage::default(), // Stub doesn track
            interrupted: false,
        })
    }

    async fn run_with_credentials(
        &self,
        command: SandboxCommand,
        _creds: Vec<CredentialRef>,
    ) -> Result<SandboxOutcome, SandboxError> {
        // Stub doesn't inject credentials.
        // The real vault-backed sandbox handles this.
        tracing::warn!("stub sandbox: credential injection not implemented");
        self.run(command).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_sandbox_echo() {
        let sandbox = StubSandbox::new();
        let cmd = SandboxCommand::new("echo", vec!["hello".to_string()]);
        let outcome = sandbox.run(cmd).await.unwrap();
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_forbidden_command() {
        let sandbox = StubSandbox::new();
        let cmd = SandboxCommand::new("rm", vec!["-rf".to_string(), "/".to_string()]);
        let result = sandbox.run(cmd).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SandboxError::Forbidden(_)));
    }
}
