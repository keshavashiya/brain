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
            // Issue 128: extend deny-list beyond the original
            // disk/admin-destructive set with common process-control,
            // interpreter, and networking binaries. Interpreters
            // (`python`, `node`, `perl`, `ruby`, `php`) bypass the
            // per-binary allowlist by hosting arbitrary code; netcat
            // family (`nc`, `ncat`, `socat`) exfils data even when
            // Seatbelt/namespaces fail to deny network; `kill` and
            // friends terminate sibling processes (including Brain).
            forbidden_commands: HashSet::from([
                "dd".to_string(),
                "mkfs".to_string(),
                "shutdown".to_string(),
                "reboot".to_string(),
                "halt".to_string(),
                "poweroff".to_string(),
                "kill".to_string(),
                "killall".to_string(),
                "pkill".to_string(),
                "python".to_string(),
                "python3".to_string(),
                "node".to_string(),
                "deno".to_string(),
                "perl".to_string(),
                "ruby".to_string(),
                "php".to_string(),
                "nc".to_string(),
                "ncat".to_string(),
                "socat".to_string(),
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

        // Issue 59: the pre_exec `unshare(CLONE_NEWNET|IPC|UTS)` call is
        // silently best-effort — without CAP_SYS_ADMIN or unprivileged
        // user namespaces it returns EPERM and the child runs without
        // namespace isolation. Probe the kernel state at construction
        // time and emit a one-shot warning so operators are not surprised
        // when network isolation isn't actually in effect.
        #[cfg(target_os = "linux")]
        {
            let euid = unsafe { libc::geteuid() };
            if euid != 0 && !linux_userns_likely_available() {
                tracing::warn!(
                    "linux sandbox isolation: namespace unshare requires CAP_SYS_ADMIN or \
                     unprivileged user namespaces enabled — falling back to rlimits-only. \
                     Set `kernel.unprivileged_userns_clone=1` (Debian/Ubuntu) or raise \
                     `user.max_user_namespaces` (Fedora/RHEL) for full isolation."
                );
            } else {
                tracing::info!("linux sandbox isolation: namespace unshare available");
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

/// Expand a leading `~` or `~/` in an argv token to `home`. Used only on the
/// direct (non-shell) exec path, where there is no shell to do it — argv
/// tokens otherwise reach the binary verbatim and `ls ~/.brain` fails with
/// "No such file or directory". `~user` forms are intentionally left literal:
/// we only resolve the current user's home, never look up other users.
fn expand_leading_tilde(arg: &str, home: &str) -> String {
    if arg == "~" {
        home.to_string()
    } else if let Some(rest) = arg.strip_prefix("~/") {
        format!("{}/{}", home.trim_end_matches('/'), rest)
    } else {
        arg.to_string()
    }
}

#[async_trait]
impl SandboxExecutor for IsolatedSandbox {
    async fn run(&self, command: SandboxCommand) -> Result<SandboxOutcome, SandboxError> {
        self.validate(&command)?;

        // On the direct (non-shell) path no shell expands `~`, so resolve a
        // leading tilde in each argv token to the user's home before spawn.
        // The `sh -c` shell tier already expands it, so leave that untouched.
        let mut command = command;
        if !command.shell_mode {
            let home = command
                .env
                .get("HOME")
                .cloned()
                .or_else(|| std::env::var("HOME").ok());
            if let Some(home) = home {
                for arg in &mut command.args {
                    *arg = expand_leading_tilde(arg, &home);
                }
            }
        }

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
    // `IsolatedSandbox::new` already warned at startup if this is likely
    // to fail (Issue 59), so we don't repeat the warning per exec —
    // which would also be unsafe from pre_exec context.
    unsafe {
        let _ = libc::unshare(libc::CLONE_NEWNET | libc::CLONE_NEWIPC | libc::CLONE_NEWUTS);
    }
}

/// Issue 59: cheap heuristic for whether unprivileged user namespaces
/// will let our subsequent `unshare(CLONE_NEW*)` succeed. We can't probe
/// without committing to a real namespace, so read the well-known
/// kernel sysfs flags. False negatives are fine — the warning is just a
/// hint, the actual exec still tries.
#[cfg(target_os = "linux")]
fn linux_userns_likely_available() -> bool {
    // Debian/Ubuntu toggle. `1` = enabled.
    if std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone")
        .map(|s| s.trim() == "1")
        .unwrap_or(false)
    {
        return true;
    }
    // Fedora/RHEL surface — non-zero quota means user namespaces are
    // allowed for unprivileged users.
    std::fs::read_to_string("/proc/sys/user/max_user_namespaces")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|n| n > 0)
        .unwrap_or(false)
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

    #[test]
    fn expand_leading_tilde_resolves_home_forms() {
        let home = "/home/alice";
        assert_eq!(expand_leading_tilde("~", home), "/home/alice");
        assert_eq!(expand_leading_tilde("~/.brain", home), "/home/alice/.brain");
        assert_eq!(expand_leading_tilde("~/a/b", home), "/home/alice/a/b");
        // Trailing slash on home doesn't double up.
        assert_eq!(expand_leading_tilde("~/x", "/home/alice/"), "/home/alice/x");
        // Non-leading tilde and `~user` forms stay literal.
        assert_eq!(expand_leading_tilde("~root/x", home), "~root/x");
        assert_eq!(expand_leading_tilde("/tmp/~/x", home), "/tmp/~/x");
        assert_eq!(expand_leading_tilde("plain", home), "plain");
    }

    #[tokio::test]
    async fn argv_mode_expands_leading_tilde() {
        // `cat ~/<file>` against an overridden HOME should read the file —
        // proving the leading `~` was expanded before exec (no shell here).
        let dir = std::env::temp_dir().join(format!("brain-tilde-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker.txt"), "tilde-ok").unwrap();

        let sandbox = IsolatedSandbox::new(vec!["cat".into()], Duration::from_secs(5));
        let cmd = SandboxCommand::new("cat", vec!["~/marker.txt".into()])
            .with_env("HOME", dir.to_string_lossy().into_owned());
        let outcome = sandbox.run(cmd).await.unwrap();

        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(outcome.exit_code, 0, "stderr: {:?}", outcome.stderr);
        assert!(
            outcome.stdout.contains("tilde-ok"),
            "stdout was: {:?}",
            outcome.stdout
        );
    }

    #[tokio::test]
    async fn shell_mode_leaves_tilde_for_the_shell() {
        // In shell mode the `~` must reach `sh`, which expands it against the
        // child's HOME — we must NOT pre-expand and double-resolve it.
        let dir = std::env::temp_dir().join(format!("brain-tilde-sh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker.txt"), "shell-tilde-ok").unwrap();

        let sandbox = IsolatedSandbox::new(vec!["sh".into()], Duration::from_secs(5));
        // Pin PATH so the shell-mode HOME-injection branch (which would
        // otherwise force the daemon's HOME) is skipped and our HOME stands.
        let cmd = SandboxCommand::shell("cat ~/marker.txt")
            .with_env("HOME", dir.to_string_lossy().into_owned())
            .with_env("PATH", "/bin:/usr/bin");
        let outcome = sandbox.run(cmd).await.unwrap();

        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(outcome.exit_code, 0, "stderr: {:?}", outcome.stderr);
        assert!(outcome.stdout.contains("shell-tilde-ok"));
    }

    // ── Property tests ────────────────────────────────────────────────
    //
    // `validate` is the pure, fail-closed gate every execution passes through.
    // These pin its security invariants over arbitrary commands without
    // spawning anything: an empty allowlist denies everything, the deny-list
    // overrides the allow-list, a clean allowlisted binary is never spuriously
    // refused, and the cloud-metadata IP is blocked wherever it appears.
    mod props {
        use super::super::*;
        use proptest::prelude::*;

        /// Binaries that are valid allowlist entries and are *not* in the
        /// default deny-list seeded by `IsolatedSandbox::new`.
        fn safe_basename() -> impl Strategy<Value = String> {
            prop_oneof![
                Just("echo".to_string()),
                Just("true".to_string()),
                Just("false".to_string()),
                Just("sleep".to_string()),
                Just("ls".to_string()),
                Just("cat".to_string()),
                Just("grep".to_string()),
                Just("mytool".to_string()),
            ]
        }

        /// An argument with no dots, so it can never contain the metadata IP.
        fn clean_arg() -> impl Strategy<Value = String> {
            "[a-z0-9_-]{0,10}".prop_map(|s| s.to_string())
        }

        proptest! {
            #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

            /// Fail-closed: with an empty allowlist no command validates —
            /// any binary, any args, shell mode or not.
            #[test]
            fn empty_allowlist_denies_everything(
                binary in "[a-zA-Z0-9_/.-]{1,16}",
                args in proptest::collection::vec("[a-zA-Z0-9 _/.-]{0,12}", 0..4),
                shell_mode in any::<bool>(),
            ) {
                let sb = IsolatedSandbox::new(vec![], Duration::from_secs(5));
                let mut cmd = SandboxCommand::new(binary, args);
                cmd.shell_mode = shell_mode;
                prop_assert!(sb.validate(&cmd).is_err());
            }

            /// The deny-list overrides the allow-list: a basename present in
            /// both is rejected, whether invoked bare or via an absolute path.
            #[test]
            fn forbidden_basename_rejected_even_when_allowlisted(
                base in safe_basename(),
                with_dir in any::<bool>(),
                args in proptest::collection::vec(clean_arg(), 0..3),
            ) {
                let sb = IsolatedSandbox::new(vec![base.clone()], Duration::from_secs(5))
                    .with_forbidden_commands(vec![base.clone()]);
                let binary = if with_dir { format!("/usr/bin/{base}") } else { base.clone() };
                let cmd = SandboxCommand::new(binary, args);
                prop_assert!(sb.validate(&cmd).is_err());
            }

            /// No spurious refusal: an allowlisted, non-forbidden binary with
            /// clean args validates — its basename resolves through an
            /// absolute path too, and an empty path-allowlist imposes no
            /// workdir restriction.
            #[test]
            fn allowlisted_clean_binary_passes(
                base in safe_basename(),
                with_dir in any::<bool>(),
                args in proptest::collection::vec(clean_arg(), 0..3),
            ) {
                let sb = IsolatedSandbox::new(vec![base.clone()], Duration::from_secs(5));
                let binary = if with_dir { format!("/usr/bin/{base}") } else { base.clone() };
                let cmd = SandboxCommand::new(binary, args);
                prop_assert!(sb.validate(&cmd).is_ok());
            }

            /// The cloud-metadata IP is blocked wherever it sits in the args,
            /// even for an otherwise-allowed binary.
            #[test]
            fn cloud_metadata_ip_in_any_arg_is_rejected(
                base in safe_basename(),
                prefix in "[a-z/:.]{0,8}",
                suffix in "[a-z/]{0,8}",
                pos in 0usize..4,
                extra in proptest::collection::vec(clean_arg(), 0..3),
            ) {
                let mut args = extra;
                let at = pos.min(args.len());
                args.insert(at, format!("{prefix}169.254.169.254{suffix}"));
                let sb = IsolatedSandbox::new(vec![base.clone()], Duration::from_secs(5));
                let cmd = SandboxCommand::new(base, args);
                prop_assert!(sb.validate(&cmd).is_err());
            }
        }
    }
}
