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
    std::fs::write(pid_path(config), pid.to_string())?;
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
