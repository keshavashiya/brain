//! Real sandbox executor.
//!
//! Applies resource limits (`setrlimit`) via a pre-exec hook, enforces the
//! configured binary allowlist + filesystem allowlist, and kills the child's
//! process group on timeout.
//!
//! Platform layers (best-effort, applied on top of the MVP):
//! - macOS: wraps the invocation in `sandbox-exec -f <profile>` with a
//!   generated Seatbelt profile that denies outbound network.
//! - Linux: pre-exec `unshare(CLONE_NEWNET | CLONE_NEWIPC | CLONE_NEWUTS)`
//!   when running with the required privileges; silently falls back to
//!   rlimits-only when the kernel rejects the call.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use tracing;

use super::allowlist::{
    CredentialRef, ResourceUsage, SandboxCommand, SandboxError, SandboxExecutor, SandboxOutcome,
};

/// Per-process resource ceilings enforced via `setrlimit` before exec.
#[derive(Debug, Clone, Copy)]
pub struct SandboxLimits {
    /// CPU seconds (RLIMIT_CPU soft limit). Hard limit is +1 so the kernel
    /// sends SIGKILL after a grace window following SIGXCPU.
    pub cpu_seconds: u64,
    /// Address space ceiling in bytes (RLIMIT_AS).
    pub memory_bytes: u64,
    /// Max open file descriptors (RLIMIT_NOFILE).
    pub nofile: u64,
    /// Max file size the process may create (RLIMIT_FSIZE).
    pub fsize_bytes: u64,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            cpu_seconds: 60,
            memory_bytes: 1024 * 1024 * 1024, // 1 GiB
            nofile: 256,
            fsize_bytes: 256 * 1024 * 1024, // 256 MiB
        }
    }
}

/// Real isolated sandbox executor.
pub struct IsolatedSandbox {
    command_allowlist: HashSet<String>,
    allowed_paths: HashSet<PathBuf>,
    forbidden_commands: HashSet<String>,
    limits: SandboxLimits,
    default_timeout: Duration,
    #[cfg(target_os = "macos")]
    macos_profile_path: Option<PathBuf>,
}

impl IsolatedSandbox {
    pub fn new(command_allowlist: Vec<String>, default_timeout: Duration) -> Self {
        let allowlist: HashSet<String> = command_allowlist.into_iter().collect();

        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut sb = Self {
            command_allowlist: allowlist,
            allowed_paths: HashSet::new(),
            forbidden_commands: HashSet::from([
                "dd".to_string(),
                "mkfs".to_string(),
                "shutdown".to_string(),
                "reboot".to_string(),
            ]),
            limits: SandboxLimits::default(),
            default_timeout,
            #[cfg(target_os = "macos")]
            macos_profile_path: None,
        };

        #[cfg(target_os = "macos")]
        {
            sb.macos_profile_path = Self::write_macos_profile().ok();
            if sb.macos_profile_path.is_none() {
                tracing::warn!(
                    "failed to write macOS sandbox-exec profile; running with rlimits only"
                );
            }
        }

        sb
    }

    pub fn with_allowed_paths(mut self, paths: Vec<PathBuf>) -> Self {
        self.allowed_paths.extend(paths);
        self
    }

    pub fn with_limits(mut self, limits: SandboxLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_forbidden_commands(mut self, commands: Vec<String>) -> Self {
        self.forbidden_commands.extend(commands);
        self
    }

    fn binary_basename(binary: &str) -> &str {
        Path::new(binary)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(binary)
    }

    fn validate(&self, command: &SandboxCommand) -> Result<(), SandboxError> {
        // Binary allowlist — empty allowlist means "reject everything"
        // (fail-closed; the config default ships a non-empty list).
        // Shell-mode commands (`sh -c "..."`) only gate `sh` itself —
        // by opting in via SandboxCommand::shell the caller has accepted
        // that the wrapped command can call any binary on PATH. The
        // remaining safety controls (rlimits, Seatbelt, timeout, and
        // the explicit `forbidden_commands` deny-list) still apply.
        let basename = Self::binary_basename(&command.binary);
        if self.command_allowlist.is_empty() {
            return Err(SandboxError::Forbidden(
                "sandbox allowlist is empty; configure security.exec_allowlist".into(),
            ));
        }
        if !command.shell_mode && !self.command_allowlist.contains(basename) {
            return Err(SandboxError::Forbidden(format!(
                "binary '{basename}' not in allowlist"
            )));
        }
        if command.shell_mode && basename != "sh" {
            return Err(SandboxError::Forbidden(format!(
                "shell_mode requires binary='sh', got '{basename}'"
            )));
        }

        for forbidden in &self.forbidden_commands {
            if basename == forbidden {
                return Err(SandboxError::Forbidden(format!(
                    "binary '{basename}' is explicitly forbidden"
                )));
            }
            // For shell mode also reject if the wrapped command names a
            // forbidden binary as its first token. Cheap pre-screen —
            // a determined caller can still hide it behind variables,
            // but most of the time this catches obvious cases.
            if command.shell_mode {
                if let Some(wrapped) = command.args.get(1) {
                    if let Some(first_token) = wrapped.split_whitespace().next() {
                        let wrapped_base = std::path::Path::new(first_token)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or(first_token);
                        if wrapped_base == forbidden {
                            return Err(SandboxError::Forbidden(format!(
                                "shell command starts with forbidden binary '{forbidden}'"
                            )));
                        }
                    }
                }
            }
        }

        for arg in &command.args {
            if arg.contains("169.254.169.254") {
                return Err(SandboxError::Forbidden("cloud metadata IP blocked".into()));
            }
        }

        if !self.allowed_paths.is_empty()
            && !self
                .allowed_paths
                .iter()
                .any(|p| command.workdir.starts_with(p))
        {
            return Err(SandboxError::PathNotAllowed(format!(
                "workdir {:?} not in allowlist",
                command.workdir
            )));
        }

        Ok(())
    }

    /// Build the concrete invocation (potentially wrapped with sandbox-exec).
    #[cfg(target_os = "macos")]
    fn resolve_invocation(&self, command: &SandboxCommand) -> (String, Vec<String>) {
        if let Some(profile) = &self.macos_profile_path {
            let mut wrapped = vec![
                "-f".to_string(),
                profile.to_string_lossy().into_owned(),
                command.binary.clone(),
            ];
            wrapped.extend(command.args.iter().cloned());
            ("/usr/bin/sandbox-exec".to_string(), wrapped)
        } else {
            (command.binary.clone(), command.args.clone())
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn resolve_invocation(&self, command: &SandboxCommand) -> (String, Vec<String>) {
        (command.binary.clone(), command.args.clone())
    }

    #[cfg(target_os = "macos")]
    fn write_macos_profile() -> std::io::Result<PathBuf> {
        // Seatbelt profile: deny outbound network by default, let everything
        // else inherit macOS defaults. Keeps compatibility with git/cargo/etc.
        // which need to read system frameworks and write inside the workdir.
        let profile = r#"(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (local ip))
(allow network-outbound (remote unix-socket))
"#;
        let path = std::env::temp_dir().join(format!("brain-sandbox-{}.sb", std::process::id()));
        std::fs::write(&path, profile)?;
        Ok(path)
    }
}

#[async_trait]
impl SandboxExecutor for IsolatedSandbox {
    async fn run(&self, command: SandboxCommand) -> Result<SandboxOutcome, SandboxError> {
        self.validate(&command)?;

        let timeout = if command.timeout.is_zero() {
            self.default_timeout
        } else {
            command.timeout
        };

        let (binary, args) = self.resolve_invocation(&command);

        tracing::info!(
            binary = %binary,
            args = ?args,
            workdir = ?command.workdir,
            cpu_s = self.limits.cpu_seconds,
            mem_mb = self.limits.memory_bytes / (1024 * 1024),
            nofile = self.limits.nofile,
            "sandbox: executing with rlimits"
        );

        let start = std::time::Instant::now();

        let mut cmd = tokio::process::Command::new(&binary);
        cmd.args(&args);
        cmd.current_dir(&command.workdir);

        // Drop inherited environment; re-inject only what the caller set so
        // the child can't rely on ambient secrets/paths from the daemon.
        cmd.env_clear();
        for (k, v) in &command.env {
            cmd.env(k, v);
        }
        // PATH is essential for most binaries to resolve subcommands.
        // Shell-mode commands additionally get the user's toolchain
        // dirs prepended (`~/.cargo/bin`, `/opt/homebrew/bin`, etc.) so
        // a plan that runs `cargo` works regardless of how the daemon
        // was launched. Argv-mode keeps the original behaviour — the
        // per-binary allowlist is the gate, and absolute paths still
        // work.
        if !command.env.contains_key("PATH") {
            let mut path = std::env::var("PATH")
                .unwrap_or_else(|_| "/usr/local/bin:/usr/bin:/bin".to_string());
            if command.shell_mode {
                if let Some(home) = std::env::var_os("HOME") {
                    let extras = [
                        std::path::PathBuf::from(&home).join(".cargo/bin"),
                        std::path::PathBuf::from(&home).join(".rustup/toolchains"),
                        std::path::PathBuf::from("/opt/homebrew/bin"),
                        std::path::PathBuf::from("/usr/local/sbin"),
                    ];
                    for p in extras.iter().filter(|p| p.exists()) {
                        path = format!("{}:{}", p.display(), path);
                    }
                }
                // HOME itself is needed by cargo/git for config lookup.
                if let Some(home) = std::env::var_os("HOME") {
                    cmd.env("HOME", home);
                }
            }
            cmd.env("PATH", path);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        #[cfg(unix)]
        cmd.process_group(0);

        #[cfg(unix)]
        {
            let limits = self.limits;
            // SAFETY: pre_exec runs in the forked child between fork and exec.
            // We only call async-signal-safe syscalls (setrlimit, unshare).
            // No allocations, no locks.
            unsafe {
                use std::os::unix::process::CommandExt;
                cmd.as_std_mut().pre_exec(move || {
                    apply_rlimits(&limits)?;
                    #[cfg(target_os = "linux")]
                    apply_linux_namespaces();
                    Ok(())
                });
            }
        }

        let child = cmd.spawn().map_err(SandboxError::Io)?;

        #[cfg(unix)]
        let child_pid = child.id();

        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Err(SandboxError::Io(e)),
            Err(_) => {
                #[cfg(unix)]
                if let Some(pid) = child_pid {
                    unsafe {
                        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
                    }
                    tracing::warn!(pid, "sandbox timeout: SIGKILL sent to process group");
                }
                return Err(SandboxError::Timeout(timeout));
            }
        };

        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout)
            .chars()
            .take(1_048_576)
            .collect();
        let stderr = String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(1_048_576)
            .collect();

        let exit_code = output.status.code().unwrap_or(-1);

        // Detect rlimit-triggered signals so the caller can tell a crash
        // apart from a resource-limit kill.
        #[cfg(unix)]
        let interrupted = {
            use std::os::unix::process::ExitStatusExt;
            output.status.signal().is_some()
        };
        #[cfg(not(unix))]
        let interrupted = false;

        tracing::info!(
            exit_code,
            duration_ms = duration.as_millis(),
            interrupted,
            "sandbox: execution complete"
        );

        Ok(SandboxOutcome {
            stdout,
            stderr,
            exit_code,
            duration,
            resource_usage: ResourceUsage::default(),
            interrupted,
        })
    }

    async fn run_with_credentials(
        &self,
        command: SandboxCommand,
        creds: Vec<CredentialRef>,
    ) -> Result<SandboxOutcome, SandboxError> {
        if !creds.is_empty() {
            // Injection is vault's job; sandbox just carries the command. The
            // caller should have resolved creds into env vars on `command`.
            tracing::debug!(
                count = creds.len(),
                "sandbox: credentials ignored (caller must pre-resolve into env)"
            );
        }
        self.run(command).await
    }
}

#[cfg(unix)]
fn apply_rlimits(limits: &SandboxLimits) -> std::io::Result<()> {
    let cpu = libc::rlimit {
        rlim_cur: limits.cpu_seconds as libc::rlim_t,
        rlim_max: (limits.cpu_seconds + 1) as libc::rlim_t,
    };
    let mem = libc::rlimit {
        rlim_cur: limits.memory_bytes as libc::rlim_t,
        rlim_max: limits.memory_bytes as libc::rlim_t,
    };
    let nofile = libc::rlimit {
        rlim_cur: limits.nofile as libc::rlim_t,
        rlim_max: limits.nofile as libc::rlim_t,
    };
    let fsize = libc::rlimit {
        rlim_cur: limits.fsize_bytes as libc::rlim_t,
        rlim_max: limits.fsize_bytes as libc::rlim_t,
    };

    // SAFETY: setrlimit is async-signal-safe; we're between fork and exec.
    unsafe {
        if libc::setrlimit(libc::RLIMIT_CPU, &cpu) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        // RLIMIT_AS is not honored on macOS (returns EINVAL); ignore failures
        // there so the process still starts with CPU/NOFILE/FSIZE enforced.
        let as_rc = libc::setrlimit(libc::RLIMIT_AS, &mem);
        #[cfg(not(target_os = "macos"))]
        if as_rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(target_os = "macos")]
        let _ = as_rc;

        if libc::setrlimit(libc::RLIMIT_NOFILE, &nofile) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        if libc::setrlimit(libc::RLIMIT_FSIZE, &fsize) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_linux_namespaces() {
    // Best-effort: CLONE_NEWNET/IPC/UTS without CAP_SYS_ADMIN will fail with
    // EPERM. Callers without user-namespace setup will simply run with
    // rlimits only; we deliberately don't fail the exec here.
    unsafe {
        let _ = libc::unshare(libc::CLONE_NEWNET | libc::CLONE_NEWIPC | libc::CLONE_NEWUTS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_allowlist() -> Vec<String> {
        vec![
            "echo".into(),
            "true".into(),
            "false".into(),
            "sleep".into(),
            "sh".into(),
        ]
    }

    #[tokio::test]
    async fn echo_runs() {
        let sandbox = IsolatedSandbox::new(default_allowlist(), Duration::from_secs(5));
        let cmd = SandboxCommand::new("echo", vec!["hi".into()]);
        let outcome = sandbox.run(cmd).await.unwrap();
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("hi"));
    }

    #[tokio::test]
    async fn binary_not_in_allowlist_is_rejected() {
        let sandbox = IsolatedSandbox::new(vec!["ls".into()], Duration::from_secs(5));
        let cmd = SandboxCommand::new("rm", vec!["-rf".into(), "/".into()]);
        let err = sandbox.run(cmd).await.unwrap_err();
        assert!(matches!(err, SandboxError::Forbidden(_)));
    }

    #[tokio::test]
    async fn empty_allowlist_rejects_all() {
        let sandbox = IsolatedSandbox::new(vec![], Duration::from_secs(5));
        let cmd = SandboxCommand::new("echo", vec!["hi".into()]);
        let err = sandbox.run(cmd).await.unwrap_err();
        assert!(matches!(err, SandboxError::Forbidden(_)));
    }

    #[tokio::test]
    async fn timeout_kills_process_group() {
        let sandbox = IsolatedSandbox::new(default_allowlist(), Duration::from_secs(5));
        let cmd = SandboxCommand::new("sleep", vec!["30".into()])
            .with_timeout(Duration::from_millis(200));
        let err = sandbox.run(cmd).await.unwrap_err();
        assert!(matches!(err, SandboxError::Timeout(_)));
    }

    #[tokio::test]
    async fn cloud_metadata_ip_blocked() {
        let sandbox = IsolatedSandbox::new(vec!["curl".into()], Duration::from_secs(5));
        let cmd = SandboxCommand::new("curl", vec!["http://169.254.169.254/meta".into()]);
        let err = sandbox.run(cmd).await.unwrap_err();
        assert!(matches!(err, SandboxError::Forbidden(_)));
    }

    #[tokio::test]
    async fn shell_mode_runs_pipelines() {
        // Argv mode can't pipe; shell mode delegates to /bin/sh which
        // can. Verifies the new SandboxCommand::shell tier works
        // end-to-end through the IsolatedSandbox.
        let sandbox = IsolatedSandbox::new(default_allowlist(), Duration::from_secs(5));
        let cmd = SandboxCommand::shell("printf 'a\\nb\\nc\\n' | wc -l");
        let outcome = sandbox.run(cmd).await.unwrap();
        assert_eq!(outcome.exit_code, 0, "stderr: {:?}", outcome.stderr);
        assert!(
            outcome.stdout.trim().ends_with("3"),
            "expected line count 3, got {:?}",
            outcome.stdout
        );
    }

    #[tokio::test]
    async fn shell_mode_bypasses_per_binary_allowlist() {
        // `wc` isn't on this allowlist, but shell mode wraps in `sh -c`
        // and only `sh` itself is gated. The wrapped `wc` runs because
        // the caller opted into shell mode.
        let sandbox = IsolatedSandbox::new(vec!["sh".into()], Duration::from_secs(5));
        let cmd = SandboxCommand::shell("echo hi | wc -c");
        let outcome = sandbox.run(cmd).await.unwrap();
        assert_eq!(outcome.exit_code, 0, "stderr: {:?}", outcome.stderr);
    }

    #[tokio::test]
    async fn shell_mode_still_blocks_forbidden_first_token() {
        // The forbidden_commands deny-list still applies even in shell
        // mode when the wrapped command's first token names a banned
        // binary. Defense in depth — rlimits would also catch dd, but
        // the early reject is clearer.
        let sandbox = IsolatedSandbox::new(vec!["sh".into()], Duration::from_secs(5));
        let cmd = SandboxCommand::shell("dd if=/dev/zero of=/tmp/x bs=1");
        let err = sandbox.run(cmd).await.unwrap_err();
        assert!(
            matches!(err, SandboxError::Forbidden(_)),
            "expected Forbidden, got {err:?}"
        );
    }

    #[tokio::test]
    async fn nofile_rlimit_takes_effect() {
        // Spawn a shell that tries to open more FDs than the limit allows.
        let limits = SandboxLimits {
            nofile: 16,
            ..SandboxLimits::default()
        };
        let sandbox =
            IsolatedSandbox::new(default_allowlist(), Duration::from_secs(5)).with_limits(limits);
        // `sh -c 'ulimit -n'` prints the current soft nofile limit.
        let cmd = SandboxCommand::new("sh", vec!["-c".into(), "ulimit -n".into()]);
        let outcome = sandbox.run(cmd).await.unwrap();
        assert_eq!(outcome.exit_code, 0);
        let reported: u64 = outcome.stdout.trim().parse().unwrap_or(0);
        assert_eq!(reported, 16, "stdout was: {:?}", outcome.stdout);
    }
}
