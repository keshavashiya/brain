//! OS service install/uninstall — launchd, systemd, Task Scheduler.

use clap::Subcommand;

#[derive(Subcommand)]
pub(crate) enum ServiceAction {
    /// Connect the brainstem — register as a login service and wake immediately.
    Install,
    /// Sever the brainstem — remove login service and stop auto-start.
    Uninstall,
}

#[allow(clippy::needless_return)]
pub(crate) async fn cmd_service_install() -> anyhow::Result<()> {
    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("Cannot determine Brain binary path: {e}"))?;
    let exe_str = exe
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Binary path contains non-UTF-8 characters"))?;

    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("$HOME is not set"))?;

    #[cfg(target_os = "macos")]
    {
        let agents_dir = home.join("Library").join("LaunchAgents");
        std::fs::create_dir_all(&agents_dir)?;
        let plist_path = agents_dir.join("com.brain.plist");

        let log_dir = home.join(".brain").join("logs");
        std::fs::create_dir_all(&log_dir)?;

        let plist = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.brain</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log}/brain.log</string>
    <key>StandardErrorPath</key>
    <string>{log}/brain.log</string>
    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
"#,
            exe = exe_str,
            log = log_dir.display(),
        );

        // If a previous plist exists, unload it first (stop the launchd-managed instance).
        let plist_str = plist_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path contains non-UTF-8 characters"))?;
        if plist_path.exists() {
            let _ = std::process::Command::new("launchctl")
                .args(["unload", plist_str])
                .output();
        }

        // Stop any existing Brain daemon — check HTTP health endpoint first (single
        // source of truth), fall back to PID file. This prevents RuVector lock
        // contention when launchd starts the new instance.
        use crate::daemon;
        let config = brain::BrainConfig::load().ok();
        if let Some(ref cfg) = config {
            if daemon::is_daemon_running(cfg).await {
                // Daemon respondinging to HTTP — kill via PID if we have it.
                if let Some(pid) = daemon::read_pid(cfg) {
                    let _ = daemon::stop_process(pid);
                } else {
                    // No PID but HTTP responds — find and kill the process
                    // by scanning for "brain serve" processes.
                    let out = std::process::Command::new("pkill")
                        .args(["-f", "brain serve"])
                        .output()
                        .ok();
                    if let Some(out) = out {
                        if !out.status.success() {
                            tracing::debug!(
                                "pkill brain serve: {}",
                                String::from_utf8_lossy(&out.stderr)
                            );
                        }
                    }
                }
                // Wait for process to exit and release file locks (max 5s).
                for _ in 0..10 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if !daemon::is_daemon_running(cfg).await {
                        break;
                    }
                }
                daemon::remove_pid(cfg);
            } else if let Some(pid) = daemon::read_pid(cfg) {
                // HTTP dead but PID exists — stale process
                if daemon::is_process_running(pid) {
                    let _ = daemon::stop_process(pid);
                    for _ in 0..10 {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        if !daemon::is_process_running(pid) {
                            break;
                        }
                    }
                }
                daemon::remove_pid(cfg);
            }
        }

        std::fs::write(&plist_path, &plist)?;

        let out = std::process::Command::new("launchctl")
            .args(["load", "-w", plist_str])
            .output()
            .map_err(|e| anyhow::anyhow!("launchctl load failed: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            anyhow::bail!("launchctl load failed: {stderr}");
        }

        println!("Brainstem wired (launchd).");
        println!("  Plist:  {}", plist_path.display());
        println!("  Log:    {}/brain.log", log_dir.display());
        println!("  Brain will wake automatically on every login.");
        println!("  To sever: brain service uninstall");
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        let service_dir = home.join(".config").join("systemd").join("user");
        std::fs::create_dir_all(&service_dir)?;
        let service_path = service_dir.join("brain.service");

        let log_dir = home.join(".brain").join("logs");
        std::fs::create_dir_all(&log_dir)?;

        let unit = format!(
            r#"[Unit]
Description=Brain OS — your AI's long-term memory
After=network.target

[Service]
Type=simple
ExecStart={exe} serve
Restart=on-failure
RestartSec=10
StandardOutput=append:{log}/brain.log
StandardError=append:{log}/brain.log

[Install]
WantedBy=default.target
"#,
            exe = exe_str,
            log = log_dir.display(),
        );

        std::fs::write(&service_path, &unit)?;

        let reload = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status();
        let enable = std::process::Command::new("systemctl")
            .args(["--user", "enable", "--now", "brain.service"])
            .status();

        if reload.is_err() || enable.is_err() {
            println!(
                "Brainstem partially wired — unit file written to {}.",
                service_path.display()
            );
            println!("  Run manually:");
            println!("    systemctl --user daemon-reload");
            println!("    systemctl --user enable --now brain.service");
        } else {
            println!("Brainstem wired (systemd user).");
            println!("  Unit:   {}", service_path.display());
            println!("  Log:    {}/brain.log", log_dir.display());
            println!("  Brain will wake automatically on every login.");
            println!("  To sever: brain service uninstall");
        }
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let task_name = "Brain OS";
        let cmd = format!("{exe_str} serve");

        let out = std::process::Command::new("schtasks")
            .args([
                "/Create", "/TN", task_name, "/TR", &cmd, "/SC", "ONLOGON", "/RL", "HIGHEST", "/F",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("schtasks not found: {e}"))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            anyhow::bail!("schtasks /Create failed: {stderr}");
        }

        let _ = std::process::Command::new("schtasks")
            .args(["/Run", "/TN", task_name])
            .output();

        println!("Brainstem wired (Windows Task Scheduler).");
        println!("  Task:   {task_name}");
        println!("  Brain will wake automatically on every login.");
        println!("  To sever: brain service uninstall");
        return Ok(());
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        anyhow::bail!(
            "brain service install is not supported on this OS.\n\
             Manually configure your system's service manager to run: {exe_str} serve",
        )
    }
}

#[allow(clippy::needless_return)]
pub(crate) fn cmd_service_uninstall() -> anyhow::Result<()> {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("$HOME is not set"))?;

    #[cfg(target_os = "macos")]
    {
        let plist_path = home
            .join("Library")
            .join("LaunchAgents")
            .join("com.brain.plist");

        if !plist_path.exists() {
            println!("No brainstem found (no plist installed).");
            return Ok(());
        }

        let plist_str = plist_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Path contains non-UTF-8 characters"))?;
        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w", plist_str])
            .output();

        std::fs::remove_file(&plist_path)?;
        println!("Brainstem severed.");
        println!("  Removed: {}", plist_path.display());
        println!("  Brain will no longer wake automatically on login.");
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        let service_path = home
            .join(".config")
            .join("systemd")
            .join("user")
            .join("brain.service");

        if !service_path.exists() {
            println!("No brainstem found (no unit file installed).");
            return Ok(());
        }

        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "brain.service"])
            .output();

        std::fs::remove_file(&service_path)?;

        let _ = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .output();

        println!("Brainstem severed.");
        println!("  Removed: {}", service_path.display());
        println!("  Brain will no longer wake automatically on login.");
        return Ok(());
    }

    #[cfg(target_os = "windows")]
    {
        let task_name = "Brain OS";

        let _ = std::process::Command::new("schtasks")
            .args(["/End", "/TN", task_name])
            .output();

        let out = std::process::Command::new("schtasks")
            .args(["/Delete", "/TN", task_name, "/F"])
            .output()
            .map_err(|e| anyhow::anyhow!("schtasks not found: {e}"))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("cannot find") || stderr.contains("not exist") {
                println!("No brainstem found (no task registered).");
                return Ok(());
            }
            anyhow::bail!("schtasks /Delete failed: {stderr}");
        }

        println!("Brainstem severed.");
        println!("  Task '{task_name}' removed from Windows Task Scheduler.");
        println!("  Brain will no longer wake automatically on login.");
        return Ok(());
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        anyhow::bail!("brain service uninstall is not supported on this OS.")
    }
}
