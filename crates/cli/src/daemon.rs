//! Daemon lifecycle — PID management, spawn, and the high-level `start` /
//! `stop` commands.
//!
//! `cmd_start` and `cmd_stop` carry the `brain start` and `brain stop`
//! semantics end-to-end so that `main.rs::run()` stays a thin clap dispatcher
//! (see Issue 110 in the v0.5.0 wave plan).

use crate::bootstrap;
#[cfg(feature = "encryption")]
use crate::encryption;
use brain::BrainConfig;

pub(crate) fn pid_path(config: &BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("brain.pid")
}

pub(crate) fn read_pid(config: &BrainConfig) -> Option<u32> {
    std::fs::read_to_string(pid_path(config))
        .ok()?
        .trim()
        .parse()
        .ok()
}

pub(crate) fn write_pid(config: &BrainConfig, pid: u32) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let path = pid_path(config);
    std::fs::write(&path, pid.to_string())?;
    let mut perms = std::fs::metadata(&path)?.permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms)?;
    Ok(())
}

pub(crate) fn remove_pid(config: &BrainConfig) {
    let _ = std::fs::remove_file(pid_path(config));
}

/// Check whether a process with the given PID is still alive.
///
/// - Unix: uses `kill -0 <pid>` (sends no signal, just validates the PID exists)
/// - Windows: opens the process handle with `OpenProcess`; success means alive
pub(crate) fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[cfg(windows)]
    {
        let out = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .output();
        match out {
            Ok(o) => {
                let text = String::from_utf8_lossy(&o.stdout);
                text.contains(&pid.to_string())
            }
            Err(_) => false,
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        false
    }
}

/// Terminate a running Brain daemon process.
pub(crate) fn stop_process(pid: u32) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let status = std::process::Command::new("kill")
            .arg(pid.to_string())
            .status()?;
        if !status.success() {
            anyhow::bail!("Failed to send SIGTERM to PID {}", pid);
        }
        Ok(())
    }

    #[cfg(windows)]
    {
        let status = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string()])
            .status()?;
        if !status.success() {
            anyhow::bail!("taskkill failed for PID {}", pid);
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    {
        anyhow::bail!("stop_process not supported on this platform (PID {})", pid)
    }
}

/// Check whether a login service (launchd/systemd/schtasks) is installed
/// and actively managing the Brain daemon.
pub(crate) fn is_service_installed() -> bool {
    #[cfg(target_os = "macos")]
    {
        // Check if launchd plist exists AND is loaded
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        if let Some(home) = home {
            let plist = home
                .join("Library")
                .join("LaunchAgents")
                .join("com.brain.plist");
            if plist.exists() {
                let out = std::process::Command::new("launchctl")
                    .arg("list")
                    .output()
                    .ok();
                if let Some(out) = out {
                    return String::from_utf8_lossy(&out.stdout).contains("com.brain");
                }
            }
        }
        false
    }

    #[cfg(target_os = "linux")]
    {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        if let Some(home) = home {
            let unit = home
                .join(".config")
                .join("systemd")
                .join("user")
                .join("brain.service");
            if unit.exists() {
                let out = std::process::Command::new("systemctl")
                    .args(["--user", "is-active", "brain.service"])
                    .output()
                    .ok();
                if let Some(out) = out {
                    return String::from_utf8_lossy(&out.stdout).trim() == "active";
                }
            }
        }
        false
    }

    #[cfg(target_os = "windows")]
    {
        let out = std::process::Command::new("schtasks")
            .args(["/Query", "/TN", "Brain OS", "/V", "/FO", "LIST"])
            .output()
            .ok();
        if let Some(out) = out {
            return String::from_utf8_lossy(&out.stdout).contains("Brain OS");
        }
        false
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        false
    }
}

/// Async health check — probes the daemon's HTTP /health endpoint.
/// Returns true if a daemon is responding on the configured HTTP port.
///
/// Uses async reqwest because the CLI runs under #[tokio::main]; the blocking
/// client spawns its own runtime and panics when dropped inside an async
/// context (`Cannot drop a runtime in a context where blocking is not allowed`).
pub(crate) async fn is_daemon_running(config: &brain::BrainConfig) -> bool {
    let host = &config.adapters.http.host;
    let port = config.adapters.http.port;
    let health_url = format!("http://{host}:{port}/health");

    let Ok(client) = reqwest::Client::builder()
        .timeout(brain::timeouts::HEALTH_CHECK)
        .build()
    else {
        return false;
    };

    matches!(
        client.get(&health_url).send().await,
        Ok(resp) if resp.status().is_success()
    )
}

/// Stop the login service (if installed), preventing auto-restart.
/// On macOS, this unloads the plist (not just `stop`) to prevent launchd respawn.
/// On Linux, stops the systemd unit.
/// On Windows, ends the scheduled task.
pub(crate) fn stop_service() {
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        if let Some(home) = home {
            let plist = home
                .join("Library")
                .join("LaunchAgents")
                .join("com.brain.plist");
            if plist.exists() {
                if let Some(plist_str) = plist.to_str() {
                    // Unload (not stop) — this prevents launchd from respawning.
                    // The plist file remains so `brain start` can reload it.
                    let _ = std::process::Command::new("launchctl")
                        .args(["unload", plist_str])
                        .output();
                }
                // Kill any remaining process
                let _ = std::process::Command::new("pkill")
                    .args(["-f", "brain serve"])
                    .output();
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        if let Some(home) = home {
            let unit = home
                .join(".config")
                .join("systemd")
                .join("user")
                .join("brain.service");
            if unit.exists() {
                // Disable stops and prevents auto-start; --now also stops if running
                let _ = std::process::Command::new("systemctl")
                    .args(["--user", "disable", "--now", "brain.service"])
                    .output();
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // End the task if running, then disable it (prevents auto-start)
        let _ = std::process::Command::new("schtasks")
            .args(["/End", "/TN", "Brain OS"])
            .output();
        let _ = std::process::Command::new("schtasks")
            .args(["/Change", "/TN", "Brain OS", "/DISABLE"])
            .output();
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        // no-op
    }
}

/// Spawn `brain serve` as a detached background process and return its PID.
pub(crate) fn spawn_daemon(
    log_path: &std::path::Path,
    passphrase: Option<&str>,
) -> anyhow::Result<u32> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // The `serve` process now writes its structured, rotating logs to
    // `log_path` (brain.log) itself, via the tracing file appender (see
    // `logging::init`). The child's raw stdout/stderr only carry panics and
    // any stray prints, so capture those in a sibling `brain.stderr.log`
    // rather than redirecting them onto the structured log file.
    let raw_path = log_path
        .parent()
        .map(|p| p.join("brain.stderr.log"))
        .unwrap_or_else(|| std::path::PathBuf::from("brain.stderr.log"));
    let raw_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&raw_path)?;

    let exe = std::env::current_exe()?;

    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve")
        .stdout(raw_file.try_clone()?)
        .stderr(raw_file)
        .stdin(std::process::Stdio::null());

    if let Some(pp) = passphrase {
        cmd.env("BRAIN_PASSPHRASE", pp);
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000208);
    }

    let child = cmd.spawn()?;
    Ok(child.id())
}

// ── start / stop commands ────────────────────────────────────────────────

/// On macOS, if a launchd plist is installed, prefer the service manager so
/// the daemon survives terminal close and gets a proper supervised lifecycle.
/// Returns `Ok(true)` when the daemon is up via launchd; `Ok(false)` means
/// "fall through to direct spawn".
#[cfg(target_os = "macos")]
async fn try_start_via_launchd(config: &BrainConfig) -> bool {
    use std::path::Path;

    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        return false;
    };
    let plist = home
        .join("Library")
        .join("LaunchAgents")
        .join("com.brain.plist");
    if !Path::exists(&plist) {
        return false;
    }
    let Some(plist_str) = plist.to_str() else {
        return false;
    };

    // If launchd already has the service loaded and the daemon answers,
    // we're done — caller will report "already awake".
    let list_out = std::process::Command::new("launchctl")
        .arg("list")
        .output()
        .ok();
    let service_loaded = list_out
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("com.brain"))
        .unwrap_or(false);
    if service_loaded && bootstrap::detect_running_daemon(config).await.is_some() {
        println!("Brain is already awake (launchd service).");
        println!("  Logs → {}", config.data_dir().join("logs").display());
        println!("Run `brain stop` to put it to sleep first.");
        return true;
    }

    // Plist exists but daemon not running — (re)load via `load -w` so it
    // works even when the service was unloaded by `brain stop`.
    let _ = std::process::Command::new("launchctl")
        .args(["load", "-w", plist_str])
        .output();

    // Wait for it to come alive (max ~5s).
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if bootstrap::detect_running_daemon(config).await.is_some() {
            println!("Brain started via launchd service.");
            println!("  Logs → {}", config.data_dir().join("logs").display());
            return true;
        }
    }

    tracing::warn!("launchd service failed to start — falling back to direct spawn");
    false
}

#[cfg(not(target_os = "macos"))]
async fn try_start_via_launchd(_config: &BrainConfig) -> bool {
    false
}

/// `brain start` — bring up the daemon in the background.
pub(crate) async fn cmd_start(config: &BrainConfig) -> anyhow::Result<()> {
    match config.validate() {
        Err(hard_err) => anyhow::bail!("Configuration error: {}", hard_err),
        Ok(warnings) => {
            for w in &warnings {
                eprintln!("WARNING: {w}");
            }
        }
    }

    // PID files can drift (macOS `process_group(0)` etc.), so a live HTTP
    // health probe is the only reliable "already running" signal.
    if let Some(url) = bootstrap::detect_running_daemon(config).await {
        println!("Brain is already awake ({url}).");
        println!("  Logs → {}", config.data_dir().join("logs").display());
        println!("Run `brain stop` to put it to sleep first.");
        return Ok(());
    }

    if try_start_via_launchd(config).await {
        return Ok(());
    }

    // No service / launchd unavailable — spawn the daemon directly.
    // First, clear any stale PID + ports.
    if let Some(pid) = read_pid(config) {
        if is_process_running(pid) {
            tracing::warn!(pid, "Stale daemon PID found — killing");
            let _ = stop_process(pid);
            for _ in 0..10 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if bootstrap::detect_running_daemon(config).await.is_none() {
                    break;
                }
            }
        }
        remove_pid(config);
    }

    #[cfg(feature = "encryption")]
    let passphrase = encryption::resolve_start_passphrase(config)?;
    #[cfg(not(feature = "encryption"))]
    let passphrase: Option<String> = None;

    let log_path = config.data_dir().join("logs/brain.log");
    let pid = spawn_daemon(&log_path, passphrase.as_deref())?;
    write_pid(config, pid)?;

    println!("Brain is awake (PID {}).", pid);
    println!(
        "  Synapse HTTP  → http://127.0.0.1:{}",
        config.adapters.http.port
    );
    println!(
        "  Synapse WS    → ws://127.0.0.1:{}",
        config.adapters.ws.port
    );
    println!(
        "  Synapse MCP   → http://127.0.0.1:{}",
        config.adapters.mcp.port
    );
    println!("  Synapse gRPC  → 127.0.0.1:{}", config.adapters.grpc.port);
    println!("  Logs          → {}", log_path.display());
    println!("\nRun `brain stop` to put it to sleep.");
    Ok(())
}

/// `brain stop` — put the daemon to sleep.
pub(crate) async fn cmd_stop(config: &BrainConfig) -> anyhow::Result<()> {
    // Snapshot liveness BEFORE stopping the service manager so the
    // post-stop branch can distinguish "we stopped it" from "already down".
    let was_running = is_daemon_running(config).await;

    // Always stop the service manager first (if installed) so it can't
    // respawn the daemon a moment later.
    if is_service_installed() {
        stop_service();
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    match read_pid(config) {
        Some(pid) if is_process_running(pid) => {
            stop_process(pid)?;
            for _ in 0..10 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if !is_process_running(pid) {
                    break;
                }
            }
            remove_pid(config);
            println!("Brain is asleep (PID {}).", pid);
        }
        Some(_) => {
            remove_pid(config);
            println!("Brain was already asleep (stale PID file cleaned up).");
        }
        None => {
            // No PID file — daemon may have been started by service manager.
            if was_running {
                for _ in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if !is_daemon_running(config).await {
                        break;
                    }
                }
                if is_daemon_running(config).await {
                    println!("Brain stop requested but daemon did not exit cleanly.");
                } else {
                    println!("Brain is asleep.");
                }
            } else {
                println!("Brain is already asleep.");
            }
        }
    }
    Ok(())
}
