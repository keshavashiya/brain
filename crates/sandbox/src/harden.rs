//! Reusable spawn hardening for **long-lived** child processes — e.g. an MCP
//! stdio server the host keeps alive and speaks JSON-RPC to.
//!
//! [`IsolatedSandbox`](crate::IsolatedSandbox) runs one-shot commands and
//! captures their output; it can't host a process whose stdin/stdout the
//! caller keeps open. This module exposes the same controls (rlimits +
//! outbound-network denial) as a `tokio::process::Command` builder the caller
//! owns: it sets program/args and installs the pre-exec hook, leaving stdio,
//! env, and cwd for the caller (the transport layer wires those).
//!
//! Platform behaviour mirrors `IsolatedSandbox`:
//! - **all Unix**: `setrlimit` ceilings applied in a pre-exec hook.
//! - **macOS**: when network is denied, the invocation is wrapped in
//!   `sandbox-exec -f <profile>` with a `(deny default)` Seatbelt profile that
//!   blocks IP networking. `sandbox-exec` exec-replaces itself with the
//!   target, so the child keeps the caller's piped stdio.
//! - **Linux**: when network is denied, a best-effort `unshare(CLONE_NEWNET)`
//!   runs in the pre-exec hook (no-op without the required privileges, exactly
//!   as in `IsolatedSandbox`).

use crate::isolated::SandboxLimits;

/// What to enforce when spawning a hardened child. The default is fail-closed:
/// no network, `SandboxLimits::default()` ceilings.
#[derive(Debug, Clone, Default)]
pub struct StdioHardening {
    /// Allow outbound network. Default `false` — denied (macOS Seatbelt /
    /// Linux network namespace).
    pub network: bool,
    /// Per-process resource ceilings (rlimits).
    pub limits: SandboxLimits,
}

/// Build a hardened [`tokio::process::Command`] for `program` + `args`.
///
/// The returned command has its program/args set and (on Unix) a pre-exec hook
/// installed that applies the rlimit ceilings and, when network is denied, the
/// Linux network-namespace unshare. On macOS with network denied the program
/// is `sandbox-exec` wrapping the target. The caller still sets stdio, env, and
/// cwd before spawning.
pub fn hardened_stdio_command(
    program: &str,
    args: &[String],
    opts: &StdioHardening,
) -> tokio::process::Command {
    #[cfg(target_os = "macos")]
    let (prog, full_args) = if opts.network {
        (program.to_string(), args.to_vec())
    } else {
        match write_macos_deny_network_profile("brain-mcp-stdio") {
            Ok(profile) => {
                let mut wrapped = vec![
                    "-f".to_string(),
                    profile.to_string_lossy().into_owned(),
                    program.to_string(),
                ];
                wrapped.extend(args.iter().cloned());
                ("/usr/bin/sandbox-exec".to_string(), wrapped)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "failed to write sandbox-exec profile; spawning stdio child with rlimits only"
                );
                (program.to_string(), args.to_vec())
            }
        }
    };
    #[cfg(not(target_os = "macos"))]
    let (prog, full_args) = (program.to_string(), args.to_vec());

    let mut cmd = tokio::process::Command::new(&prog);
    cmd.args(&full_args);

    #[cfg(unix)]
    {
        let limits = opts.limits;
        let deny_network = !opts.network;
        // SAFETY: pre_exec runs in the forked child between fork and exec. We
        // only call async-signal-safe syscalls (setrlimit, unshare). No
        // allocations, no locks.
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.as_std_mut().pre_exec(move || {
                crate::isolated::apply_rlimits(&limits)?;
                #[cfg(target_os = "linux")]
                if deny_network {
                    // Best-effort: EPERM without CAP_SYS_ADMIN / userns. The
                    // child then runs without netns isolation, same as
                    // IsolatedSandbox.
                    let _ = libc::unshare(libc::CLONE_NEWNET);
                }
                #[cfg(not(target_os = "linux"))]
                let _ = deny_network;
                Ok(())
            });
        }
    }

    cmd
}

/// Seatbelt profile that denies IP networking while permitting ordinary local
/// work. Shared verbatim by the one-shot [`IsolatedSandbox`](crate::IsolatedSandbox)
/// and this long-lived stdio host — both want exactly the same network posture.
///
/// Uses `(deny default)` with an explicit allowlist rather than the
/// `(allow default)(deny network-outbound)` form: on recent macOS the latter
/// does not actually block outbound traffic, because a trailing
/// `(... (local ip))` allow matches every IP socket (the filter means "the
/// local endpoint is an IP", i.e. all of them), silently re-permitting the
/// network. The allowlist below blocks all IP networking while still letting
/// interpreters and binaries do their non-network work (file/dyld reads,
/// process/mach/ipc, sysctl) and talk over unix sockets. Validated to block
/// outbound TCP from bash/python while running node/python/cat and the
/// `shell.exec` toolchain (git/cargo and friends).
#[cfg(target_os = "macos")]
pub(crate) const MACOS_DENY_NETWORK_PROFILE: &str = r#"(version 1)
(deny default)
(allow process*)
(allow file*)
(allow sysctl*)
(allow mach*)
(allow ipc*)
(allow signal)
(allow system*)
(allow iokit*)
(allow pseudo-tty)
(allow network* (remote unix-socket))
"#;

/// Write [`MACOS_DENY_NETWORK_PROFILE`] to a uniquely-named temp file and return
/// its path, ready to pass to `sandbox-exec -f`. The `prefix` only names the
/// file (for debuggability); the policy is identical for every caller.
///
/// The filename carries a process-wide sequence suffix so concurrent writers
/// (notably parallel tests sharing one process) don't truncate each other's
/// profile mid-read — a reader would otherwise see an empty file and
/// `sandbox-exec` fails with "no version specified".
#[cfg(target_os = "macos")]
pub(crate) fn write_macos_deny_network_profile(
    prefix: &str,
) -> std::io::Result<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{}.sb", std::process::id(), seq));
    std::fs::write(&path, MACOS_DENY_NETWORK_PROFILE)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Stdio;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A hardened child still does normal stdio round-trips — the control that
    /// matters for an MCP stdio server (it speaks JSON-RPC over the same pipes).
    /// On macOS this also proves the `sandbox-exec` wrap doesn't break stdio.
    #[tokio::test]
    async fn hardened_child_round_trips_stdio() {
        let mut cmd = hardened_stdio_command("cat", &[], &StdioHardening::default());
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn hardened cat");

        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(b"ping\n").await.unwrap();
        drop(stdin); // EOF so cat exits

        let mut out = String::new();
        child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut out)
            .await
            .unwrap();
        child.wait().await.unwrap();
        assert_eq!(out, "ping\n");
    }

    /// With network denied (the default), an outbound TCP connect from inside
    /// the hardened child fails. Uses the platform sandbox, so it only asserts
    /// on macOS (Seatbelt) where denial is enforced without extra privileges;
    /// elsewhere it just confirms the child runs.
    #[tokio::test]
    async fn network_denied_child_cannot_connect() {
        // `bash -c` so we can run a connect attempt with the shell's /dev/tcp.
        // 1.1.1.1:80 is a stable public endpoint; the connect must fail closed.
        let script = "exec 3<>/dev/tcp/1.1.1.1/80 && echo CONNECTED || echo BLOCKED";
        let mut cmd = hardened_stdio_command(
            "bash",
            &["-c".to_string(), script.to_string()],
            &StdioHardening::default(),
        );
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::null());
        let out = cmd.output().await.expect("run hardened bash");
        let stdout = String::from_utf8_lossy(&out.stdout);

        #[cfg(target_os = "macos")]
        assert!(
            stdout.contains("BLOCKED"),
            "expected outbound connect to be blocked by Seatbelt, got: {stdout:?}"
        );
        #[cfg(not(target_os = "macos"))]
        let _ = stdout; // denial needs privileges on Linux; don't assert here
    }
}
