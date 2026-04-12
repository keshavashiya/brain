//! Daemon lifecycle helpers — PID management, spawn, and stop.

use brain_core::BrainConfig;

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
            let plist = home.join("Library").join("LaunchAgents").join("com.brain.plist");
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
            let unit = home.join(".config").join("systemd").join("user").join("brain.service");
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

/// Sync health check — probes the daemon's HTTP /health endpoint.
/// Returns true if a daemon is responding on the configured HTTP port.
pub(crate) fn is_daemon_running(config: &brain_core::BrainConfig) -> bool {
    let host = &config.adapters.http.host;
    let port = config.adapters.http.port;
    let health_url = format!("http://{host}:{port}/health");

    let client = reqwest::blocking::Client::builder()
        .timeout(brain_core::timeouts::HEALTH_CHECK)
        .build()
        .ok();
    if let Some(client) = client {
        match client.get(&health_url).send() {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    } else {
        false
    }
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
            let plist = home.join("Library").join("LaunchAgents").join("com.brain.plist");
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
            let unit = home.join(".config").join("systemd").join("user").join("brain.service");
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

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    let exe = std::env::current_exe()?;

    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("serve")
        .stdout(log_file.try_clone()?)
        .stderr(log_file)
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
