use std::{collections::HashMap, sync::Arc};

use clap::{Parser, Subcommand};
use crossterm::cursor;
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::terminal;
use crossterm::ExecutableCommand;
use rustyline::DefaultEditor;
use std::io::{stdout, Write};

/// Brain OS — your AI's long-term memory
#[derive(Parser)]
#[command(name = "brain", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the neural pathways (~/.brain/ data + config)
    Init {
        /// Overwrite existing config file
        #[arg(long)]
        force: bool,
        /// Seal the blood-brain barrier — enable encryption at rest
        /// (AES-256-GCM). Generates a salt and prompts for a passphrase.
        #[arg(long)]
        encrypt: bool,
    },

    /// Open a synapse — interactive chat session
    Chat {
        /// Optional initial message (non-interactive mode)
        message: Option<String>,
    },

    /// Run a brain scan — show system vitals
    Status,

    /// Wake the brain — start all services as a background daemon.
    ///
    /// All synapses (HTTP, WebSocket, gRPC, MCP HTTP) bind to their
    /// configured ports and keep running after the terminal closes.
    /// Logs go to ~/.brain/logs/brain.log. Use `brain stop` to sleep.
    Start,

    /// Put the brain to sleep — stop the running daemon
    Stop,

    /// Keep the brain conscious — run services in the foreground (dev mode).
    ///
    /// With no flags all four synapses start concurrently.
    /// Use flags to activate only specific synapses.
    ///
    /// Background tasks also start when configured:
    /// - Memory consolidation (enabled by default, every 24h)
    /// - Habit detection + open-loop reminders (opt-in: proactivity.enabled)
    ///
    /// Examples:
    ///   brain serve                  # all synapses
    ///   brain serve --http           # HTTP only
    ///   brain serve --http --ws      # HTTP + WebSocket
    Serve {
        /// Activate the HTTP synapse
        #[arg(long)]
        http: bool,
        /// Activate the WebSocket synapse
        #[arg(long)]
        ws: bool,
        /// Activate the gRPC synapse
        #[arg(long)]
        grpc: bool,
        /// Activate the MCP HTTP synapse
        #[arg(long)]
        mcp: bool,
        /// Host to bind all synapses to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Expose a nerve ending — MCP stdio server for external AI clients.
    ///
    /// Used when an MCP client spawns Brain as a subprocess and communicates
    /// over stdin/stdout. This runs in the foreground by design.
    Mcp,

    /// Dump a memory engram — export all facts + episodes to JSON.
    ///
    /// Examples:
    ///   brain export                      # print JSON to stdout
    ///   brain export --output backup.json # write to file
    Export {
        /// Output file path (default: stdout)
        #[arg(long, short)]
        output: Option<String>,
    },

    /// Implant a memory engram — import facts + episodes from JSON backup.
    ///
    /// Examples:
    ///   brain import backup.json
    ///   brain import backup.json --dry-run
    Import {
        /// Path to the backup JSON file
        file: String,
        /// Preview what would be imported without writing to the database
        #[arg(long)]
        dry_run: bool,
    },

    /// Wire the brainstem — manage auto-start on login.
    ///
    /// On macOS:   installs a launchd agent in ~/Library/LaunchAgents/.
    /// On Linux:   installs a systemd user service in ~/.config/systemd/user/.
    /// On Windows: registers a Task Scheduler task (no admin required).
    ///
    /// Examples:
    ///   brain service install    # wire the brainstem
    ///   brain service uninstall  # sever the brainstem
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },

    /// Manage external dependencies (SearXNG) via Docker.
    ///
    /// Runs `docker compose` with the bundled docker/docker-compose.yml.
    ///
    /// Examples:
    ///   brain deps up       # start SearXNG container
    ///   brain deps down     # stop container
    ///   brain deps status   # show container status
    Deps {
        #[command(subcommand)]
        action: DepsAction,
    },
}

#[derive(Subcommand)]
enum DepsAction {
    /// Start external service containers (SearXNG).
    Up,
    /// Stop external service containers.
    Down,
    /// Show external service container status.
    Status,
}

#[derive(Subcommand)]
enum ServiceAction {
    /// Connect the brainstem — register as a login service and wake immediately.
    ///
    /// macOS: launchd plist with KeepAlive. Linux: systemd user unit.
    /// Windows: Task Scheduler task at ONLOGON (no admin required).
    Install,
    /// Sever the brainstem — remove login service and stop auto-start.
    Uninstall,
}

// ─── Daemon helpers ───────────────────────────────────────────────────────────

fn pid_path(config: &brain_core::BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("brain.pid")
}

fn read_pid(config: &brain_core::BrainConfig) -> Option<u32> {
    std::fs::read_to_string(pid_path(config))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn write_pid(config: &brain_core::BrainConfig, pid: u32) -> anyhow::Result<()> {
    std::fs::write(pid_path(config), pid.to_string())?;
    Ok(())
}

fn remove_pid(config: &brain_core::BrainConfig) {
    let _ = std::fs::remove_file(pid_path(config));
}

/// Check whether a process with the given PID is still alive.
///
/// - Unix: uses `kill -0 <pid>` (sends no signal, just validates the PID exists)
/// - Windows: opens the process handle with `OpenProcess`; success means alive
fn is_process_running(pid: u32) -> bool {
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
        // `tasklist /FI "PID eq <pid>" /NH` prints a line for each matching process.
        // If nothing matches, output is the "No tasks are running..." message.
        // We consider the process alive when tasklist exits 0 and the PID appears.
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
///
/// - Unix: sends SIGTERM via the `kill` command
/// - Windows: uses `taskkill /PID <pid>` for a graceful termination request
fn stop_process(pid: u32) -> anyhow::Result<()> {
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
///
/// stdout and stderr are redirected to the log file.
///
/// - Unix: child is placed in its own process group so it survives terminal close / SIGHUP
/// - Windows: `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` flags achieve the same
fn spawn_daemon(log_path: &std::path::Path, passphrase: Option<&str>) -> anyhow::Result<u32> {
    // Ensure log directory exists
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

    // Pass passphrase via env so the detached daemon doesn't need a terminal
    if let Some(pp) = passphrase {
        cmd.env("BRAIN_PASSPHRASE", pp);
    }

    // Unix: detach from the terminal's process group
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    // Windows: CREATE_NEW_PROCESS_GROUP (0x00000200) | DETACHED_PROCESS (0x00000008)
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000208);
    }

    let child = cmd.spawn()?;
    Ok(child.id())
}

// ─── Encryption helpers ───────────────────────────────────────────────────────

fn salt_path(config: &brain_core::BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("db/salt")
}

fn load_salt(config: &brain_core::BrainConfig) -> Option<[u8; 16]> {
    let bytes = std::fs::read(salt_path(config)).ok()?;
    if bytes.len() == 16 {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        Some(arr)
    } else {
        None
    }
}

fn write_salt(config: &brain_core::BrainConfig, salt: &[u8; 16]) -> anyhow::Result<()> {
    let path = salt_path(config);
    std::fs::write(&path, salt.as_slice())?;
    // Restrict to owner-read/write only so the salt is not world-readable.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

/// Build an `Encryptor` from config + passphrase, or `None` when encryption is disabled.
///
/// Passphrase is read from `BRAIN_PASSPHRASE` env var (daemon/CI) or prompted
/// interactively via `rpassword`.
fn resolve_encryptor(
    config: &brain_core::BrainConfig,
) -> anyhow::Result<Option<storage::Encryptor>> {
    if !config.encryption.enabled {
        return Ok(None);
    }

    let salt = load_salt(config).ok_or_else(|| {
        anyhow::anyhow!(
            "Encryption is enabled but no salt file found at {}.\n\
             Run `brain init --encrypt` to generate one.",
            salt_path(config).display()
        )
    })?;

    let passphrase = if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
        p
    } else {
        rpassword::prompt_password("Brain passphrase: ")
            .map_err(|e| anyhow::anyhow!("Failed to read passphrase: {e}"))?
    };

    let encryptor = storage::Encryptor::from_passphrase(&passphrase, &salt)
        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;

    Ok(Some(encryptor))
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "brain=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let config = brain_core::BrainConfig::load().unwrap_or_else(|e| {
        tracing::warn!("Failed to load config, using defaults: {e}");
        brain_core::BrainConfig::default()
    });

    config.ensure_data_dirs()?;

    match cli.command {
        // ── init ──────────────────────────────────────────────────────────────
        Commands::Init { force, encrypt } => {
            let data_dir = config.data_dir();
            println!("Forming neural pathways...");
            println!("  Cortex:    {}", data_dir.display());

            let generated_key = match brain_core::BrainConfig::write_default_config(force)? {
                Some((path, key)) => {
                    println!("  Genome:    {} (written)", path.display());
                    Some(key)
                }
                None => {
                    println!(
                        "  Genome:    {} (already exists, use --force to overwrite)",
                        brain_core::BrainConfig::user_config_path().display()
                    );
                    None
                }
            };

            let subdirs = ["db", "ruvector", "models", "logs", "exports"];
            for sub in &subdirs {
                println!("  Region:    {}", data_dir.join(sub).display());
            }

            println!(
                "\n  Sensory cortex: {} (pull with `ollama pull {}`)",
                config.embedding.model, config.embedding.model
            );

            // ── Encryption setup ──────────────────────────────────────────────
            if encrypt {
                let salt = storage::Encryptor::generate_salt();
                write_salt(&config, &salt)?;

                // Enable encryption in the config file automatically
                let config_path = brain_core::BrainConfig::user_config_path();
                if let Ok(yaml) = std::fs::read_to_string(&config_path) {
                    let patched = yaml.replace(
                        "enabled: false               # Run `brain init --encrypt` to generate a salt and enable",
                        "enabled: true                # Activated by `brain init --encrypt`",
                    );
                    let _ = std::fs::write(&config_path, patched);
                }

                println!(
                    "\n  Blood-brain barrier: sealed (salt → {})",
                    salt_path(&config).display()
                );
                println!("  Set BRAIN_PASSPHRASE env var for the daemon, or");
                println!("  Brain will prompt you for a passphrase on startup.");
            }

            if let Some(key) = generated_key {
                println!("\n  API key:   {}", key);
                println!("  Use this key for HTTP/WS/MCP authentication.");
            }

            println!(
                "\nNeural pathways formed. Edit {} to customize your genome.",
                brain_core::BrainConfig::user_config_path().display()
            );
        }

        // ── chat ──────────────────────────────────────────────────────────────
        Commands::Chat { message } => {
            if let Some(msg) = message {
                chat_non_interactive(&config, &msg).await?;
            } else {
                chat_interactive(&config).await?;
            }
        }

        // ── status ────────────────────────────────────────────────────────────
        Commands::Status => {
            show_status(&config).await?;
        }

        // ── start (daemon) ────────────────────────────────────────────────────
        Commands::Start => {
            // Validate config before launching daemon — fail fast on hard errors.
            match config.validate() {
                Err(hard_err) => anyhow::bail!("Configuration error: {}", hard_err),
                Ok(warnings) => {
                    for w in &warnings {
                        eprintln!("WARNING: {w}");
                    }
                }
            }

            // Prevent double-start
            if let Some(pid) = read_pid(&config) {
                if is_process_running(pid) {
                    println!("Brain is already awake (PID {}).", pid);
                    println!(
                        "  Logs → {}",
                        config.data_dir().join("logs/brain.log").display()
                    );
                    println!("Run `brain stop` to put it to sleep first.");
                    return Ok(());
                }
                // Stale PID file — clean it up and continue
                remove_pid(&config);
            }

            // If encryption is enabled, resolve passphrase now (while we have a terminal)
            // so the detached daemon receives it via env var.
            let passphrase = if config.encryption.enabled {
                if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
                    Some(p)
                } else {
                    // Validate passphrase against the salt before spawning
                    let salt = load_salt(&config).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Encryption is enabled but no salt file found.\n\
                             Run `brain init --encrypt` to generate one."
                        )
                    })?;
                    let p = rpassword::prompt_password("Brain passphrase: ")
                        .map_err(|e| anyhow::anyhow!("Failed to read passphrase: {e}"))?;
                    // Verify key derivation succeeds before handing off to daemon
                    storage::Encryptor::from_passphrase(&p, &salt)
                        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;
                    Some(p)
                }
            } else {
                None
            };

            let log_path = config.data_dir().join("logs/brain.log");
            let pid = spawn_daemon(&log_path, passphrase.as_deref())?;
            write_pid(&config, pid)?;

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
        }

        // ── stop (daemon) ─────────────────────────────────────────────────────
        Commands::Stop => match read_pid(&config) {
            Some(pid) if is_process_running(pid) => {
                stop_process(pid)?;
                remove_pid(&config);
                println!("Brain is asleep (PID {}).", pid);
            }
            Some(_) => {
                remove_pid(&config);
                println!("Brain was already asleep (stale PID file cleaned up).");
            }
            None => {
                println!("Brain is already asleep.");
            }
        },

        // ── serve (foreground) ────────────────────────────────────────────────
        Commands::Serve {
            http,
            ws,
            grpc,
            mcp,
            host,
        } => {
            // Validate config before starting — bail on hard errors, print soft warnings.
            match config.validate() {
                Err(hard_err) => anyhow::bail!("Configuration error: {}", hard_err),
                Ok(warnings) => {
                    for w in &warnings {
                        tracing::warn!(warning = %w, "config warning");
                        eprintln!("WARNING: {w}");
                    }
                }
            }

            let run_all = !http && !ws && !grpc && !mcp;
            let encryptor = resolve_encryptor(&config)?;
            let mut processor =
                signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?;

            // Wire the notification router for proactive delivery
            {
                let db = processor.episodic().pool().clone();
                let delivery_config = config.proactivity.delivery.clone();
                let mut router =
                    signal::notification::NotificationRouter::new(db, delivery_config);

                // Attach webhook sender if messaging channels are configured
                if config.actions.messaging.enabled
                    && !config.actions.messaging.channels.is_empty()
                {
                    let res = &config.actions.resilience;
                    match WebhookMessageBackend::new(
                        &config.actions.messaging.channels,
                        config.actions.messaging.timeout_ms,
                        res,
                    ) {
                        Ok(sender) => {
                            router = router.with_webhook_sender(Box::new(sender));
                            tracing::info!("Notification webhook sender attached");
                        }
                        Err(e) => {
                            tracing::warn!("Failed to init notification webhook sender: {e}");
                        }
                    }
                }

                processor = processor.with_notification_router(router);
            }
            let processor = Arc::new(processor);

            println!("Waking Brain OS...");

            let mut set = tokio::task::JoinSet::new();

            if run_all || http {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.http.port;
                println!("  Synapse HTTP  → http://{}:{}", h, port);
                set.spawn(async move { httpadapter::serve(p, &h, port).await });
            }

            if run_all || ws {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.ws.port;
                println!("  Synapse WS    → ws://{}:{}", h, port);
                set.spawn(async move { wsadapter::serve(p, &h, port).await });
            }

            if run_all || grpc {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.grpc.port;
                println!("  Synapse gRPC  → {}:{}", h, port);
                set.spawn(async move { grpcadapter::serve(p, &h, port).await });
            }

            if run_all || mcp {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.mcp.port;
                println!("  Synapse MCP   → http://{}:{}", h, port);
                set.spawn(async move { mcp::serve_http(p, &h, port).await });
            }

            // ── Proactivity / habit engine background task ────────────────────
            // Runs the HabitEngine on a schedule (default: every 60 min).
            // When a recurring pattern is detected at the current time slot
            // and all rate limits are met, the proactive message is delivered
            // through the NotificationRouter (outbox + broadcast + webhooks).
            if config.proactivity.enabled {
                let p = processor.clone();
                let habit_cfg = ganglia::HabitConfig {
                    max_per_day: config.proactivity.max_per_day,
                    min_interval_minutes: config.proactivity.min_interval_minutes,
                    quiet_start: config.proactivity.quiet_hours.start.clone(),
                    quiet_end: config.proactivity.quiet_hours.end.clone(),
                    ..Default::default()
                };
                set.spawn(async move {
                    let engine =
                        ganglia::HabitEngine::new(p.episodic().pool().clone(), habit_cfg.clone());
                    if let Err(e) = engine.ensure_tables() {
                        tracing::warn!("HabitEngine table init failed: {e}");
                        return Ok(());
                    }
                    let check_interval = tokio::time::Duration::from_secs(
                        habit_cfg.min_interval_minutes as u64 * 60,
                    );
                    let mut ticker = tokio::time::interval(check_interval);
                    ticker.tick().await; // skip first tick
                    loop {
                        ticker.tick().await;
                        match engine.generate_proactive() {
                            Ok(Some(msg)) => {
                                tracing::info!(
                                    triggered_by = %msg.triggered_by,
                                    "Proactive: {}",
                                    msg.content
                                );
                                // Deliver through NotificationRouter (outbox + broadcast + webhooks)
                                if let Some(router) = p.notification_router() {
                                    router.deliver(msg.into()).await;
                                }
                            }
                            Ok(None) => {}
                            Err(e) => tracing::warn!("HabitEngine error: {e}"),
                        }
                    }
                });
                tracing::info!(
                    interval_minutes = config.proactivity.min_interval_minutes,
                    "Proactivity engine scheduled"
                );
            }

            // ── Open-loop detection background task ───────────────────────────
            // Scans episodic memory for unresolved commitments ("I need to...",
            // "remind me to...") and generates reminders when no resolution is
            // found within the configured window.
            if config.proactivity.enabled && config.proactivity.open_loop.enabled {
                let p = processor.clone();
                let ol_cfg = config.proactivity.open_loop.clone();
                set.spawn(async move {
                    let detector = ganglia::OpenLoopDetector::new(
                        p.episodic().pool().clone(),
                        ganglia::OpenLoopConfig {
                            scan_window_hours: ol_cfg.scan_window_hours,
                            resolution_window_hours: ol_cfg.resolution_window_hours,
                            max_reminders: 3,
                        },
                    );
                    let check_interval = tokio::time::Duration::from_secs(
                        ol_cfg.check_interval_minutes as u64 * 60,
                    );
                    let mut ticker = tokio::time::interval(check_interval);
                    ticker.tick().await; // skip first tick
                    loop {
                        ticker.tick().await;
                        match detector.generate_reminders() {
                            Ok(reminders) if !reminders.is_empty() => {
                                if let Some(router) = p.notification_router() {
                                    for msg in reminders {
                                        tracing::info!(
                                            triggered_by = %msg.triggered_by,
                                            "Open loop: {}",
                                            msg.content
                                        );
                                        router.deliver(msg.into()).await;
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(e) => tracing::warn!("OpenLoopDetector error: {e}"),
                        }
                    }
                });
                tracing::info!(
                    interval_minutes = ol_cfg.check_interval_minutes,
                    "Open-loop detector scheduled"
                );
            }

            // ── Memory consolidation background task ──────────────────────────
            // Runs the forgetting-curve pruner on a schedule (default: every 24h).
            // Aborted cleanly alongside adapters on shutdown.
            if config.memory.consolidation.enabled {
                let p = processor.clone();
                let interval_hours = config.memory.consolidation.interval_hours;
                let prune_threshold = config.memory.consolidation.forgetting_threshold;
                set.spawn(async move {
                    let consolidator =
                        hippocampus::Consolidator::new(hippocampus::ConsolidationConfig {
                            prune_threshold,
                            ..Default::default()
                        });
                    // Skip the first tick so consolidation doesn't run at startup.
                    let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(
                        interval_hours as u64 * 3600,
                    ));
                    ticker.tick().await;
                    loop {
                        ticker.tick().await;
                        match consolidator.consolidate(p.episodic()) {
                            Ok(r) => {
                                let promoted_now =
                                    promote_candidates(p.as_ref(), &r.promotion_candidates).await;

                                // Prune delivered/stale outbox notifications
                                if let Some(router) = p.notification_router() {
                                    router.prune();
                                }

                                tracing::info!(
                                    pruned = r.episodes_pruned,
                                    promotion_candidates = r.episodes_promoted,
                                    promoted = promoted_now,
                                    remaining = r.episodes_remaining,
                                    "Memory consolidation complete"
                                );
                            }
                            Err(e) => tracing::warn!("Memory consolidation error: {e}"),
                        }
                    }
                });
                tracing::info!(interval_hours, "Memory consolidation scheduled");
            }

            println!("\nBrain is conscious. Press Ctrl+C to sleep.\n");

            // ── Graceful shutdown ─────────────────────────────────────────────
            // Race between: any adapter task finishing (error path) or a
            // shutdown signal arriving (SIGTERM from `brain stop`, or Ctrl+C).
            // On signal we abort all remaining adapter tasks, then flush the
            // SQLite WAL so no committed writes are lost.

            // Build a SIGTERM future: resolves on Unix, pends forever elsewhere.
            #[cfg(unix)]
            let mut sigterm_listener = {
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Failed to register SIGTERM handler")
            };
            // Wrap in a named async block so both branches have the same type.
            let sigterm_fut = async {
                #[cfg(unix)]
                {
                    sigterm_listener.recv().await;
                }
                #[cfg(not(unix))]
                {
                    std::future::pending::<()>().await;
                }
            };

            tokio::select! {
                // An adapter task returned — usually an error
                result = set.join_next() => {
                    if let Some(r) = result {
                        match r {
                            Ok(Err(e)) => eprintln!("Adapter error: {e}"),
                            Err(e) => eprintln!("Task panicked: {e}"),
                            Ok(Ok(())) => {}
                        }
                    }
                }

                // Ctrl+C (interactive terminal)
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C — shutting down");
                }

                // SIGTERM (sent by `brain stop` on Unix)
                _ = sigterm_fut => {
                    tracing::info!("Received SIGTERM — shutting down");
                }
            }

            // Abort any adapters still running
            set.abort_all();

            // Flush SQLite WAL before exit
            processor.shutdown();
            tracing::info!("Brain OS is asleep");
        }

        // ── mcp stdio ─────────────────────────────────────────────────────────
        Commands::Mcp => {
            let encryptor = resolve_encryptor(&config)?;
            let processor =
                signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?;
            mcp::serve_stdio(processor).await?;
        }

        // ── export ────────────────────────────────────────────────────────────
        Commands::Export { output } => {
            cmd_export(&config, output.as_deref())?;
        }

        // ── import ────────────────────────────────────────────────────────────
        Commands::Import { file, dry_run } => {
            cmd_import(&config, &file, dry_run).await?;
        }

        // ── service ───────────────────────────────────────────────────────────
        Commands::Service { action } => match action {
            ServiceAction::Install => cmd_service_install()?,
            ServiceAction::Uninstall => cmd_service_uninstall()?,
        },

        // ── deps ─────────────────────────────────────────────────────────────
        Commands::Deps { action } => {
            cmd_deps(action)?;
        }
    }

    Ok(())
}

// ─── Status ───────────────────────────────────────────────────────────────────

async fn show_status(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Scan");
    println!("  DNA:          v{}", env!("CARGO_PKG_VERSION"));
    println!("  Cortex:       {}", config.data_dir().display());

    // Daemon state
    match read_pid(config) {
        Some(pid) if is_process_running(pid) => {
            println!("  State:        awake (PID {})", pid);
        }
        Some(_) => {
            println!("  State:        asleep (stale PID file)");
        }
        None => {
            println!("  State:        asleep (run `brain start` to wake)");
        }
    }

    println!(
        "  Cortex LLM:   {} ({})",
        config.llm.model, config.llm.provider
    );
    println!(
        "  Sensory:      {} ({}d)",
        config.embedding.model, config.embedding.dimensions
    );
    println!(
        "  Barrier:      {}",
        if config.encryption.enabled {
            "sealed"
        } else {
            "open"
        }
    );
    println!("  Hippocampus:  {}", config.sqlite_path().display());
    println!("  Neural mesh:  {}", config.ruvector_path().display());
    println!(
        "  Genome:       {}",
        brain_core::BrainConfig::user_config_path().display()
    );

    println!("\n  Synapses:");
    let h = &config.adapters.http;
    println!(
        "    HTTP      : port {} ({})",
        h.port,
        if h.enabled { "active" } else { "dormant" }
    );
    let w = &config.adapters.ws;
    println!(
        "    WebSocket : port {} ({})",
        w.port,
        if w.enabled { "active" } else { "dormant" }
    );
    let m = &config.adapters.mcp;
    println!(
        "    MCP       : port {} ({})",
        m.port,
        if m.enabled { "active" } else { "dormant" }
    );
    let g = &config.adapters.grpc;
    println!(
        "    gRPC      : port {} ({})",
        g.port,
        if g.enabled { "active" } else { "dormant" }
    );

    // LLM health
    let llm_cfg = cortex::llm::ProviderConfig {
        provider: config.llm.provider.clone(),
        base_url: config.llm.base_url.clone(),
        api_key: None,
        model: config.llm.model.clone(),
        temperature: config.llm.temperature,
        max_tokens: config.llm.max_tokens as i32,
    };
    let provider = cortex::llm::create_provider(&llm_cfg);
    let llm_healthy = provider.health_check().await;
    println!(
        "  Cortex:       {}",
        if llm_healthy {
            "responsive"
        } else {
            "unresponsive"
        }
    );

    // Database stats
    match storage::SqlitePool::open(&config.sqlite_path()) {
        Ok(pool) => match pool.table_stats() {
            Ok(stats) => {
                println!("\n  Memory Regions:");
                for (table, count) in stats {
                    println!("    {}: {} rows", table, count);
                }
            }
            Err(e) => println!("\n  Hippocampus: error reading stats — {}", e),
        },
        Err(e) => println!("\n  Hippocampus: error opening — {}", e),
    }

    // External service health
    let searxng_ep = config.actions.web_search.endpoint.trim().trim_end_matches('/');
    if !searxng_ep.is_empty() {
        println!("\n  External Services:");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_default();
        let health_url = format!("{}/healthz", searxng_ep);
        let healthy = client.get(&health_url).send().await.is_ok_and(|r| r.status().is_success());
        println!(
            "    {:<10}: {} ({})",
            "SearXNG",
            if healthy { "running" } else { "stopped" },
            searxng_ep
        );
    }

    Ok(())
}

// ─── Deps management ─────────────────────────────────────────────────────────

fn find_compose_file() -> Option<std::path::PathBuf> {
    // Check relative to the binary's location first (dev builds), then fallback
    // to common install paths.
    let candidates = [
        // Running from source checkout (cargo run)
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("docker/docker-compose.yml")),
        // Installed alongside binary
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("../share/brain/docker/docker-compose.yml"))),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn cmd_deps(action: DepsAction) -> anyhow::Result<()> {
    let compose_file = find_compose_file().ok_or_else(|| {
        anyhow::anyhow!(
            "docker/docker-compose.yml not found.\n\
             If installed from release, run from the Brain source directory."
        )
    })?;

    // Verify docker is available
    let docker_ok = std::process::Command::new("docker")
        .arg("info")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !docker_ok {
        anyhow::bail!(
            "Docker is not running or not installed.\n\
             Install Docker Desktop: https://docs.docker.com/get-docker/"
        );
    }

    let compose_dir = compose_file.parent().unwrap();
    let run = |args: &[&str]| -> anyhow::Result<()> {
        let status = std::process::Command::new("docker")
            .arg("compose")
            .args(["-f", compose_file.to_str().unwrap()])
            .args(["--project-directory", compose_dir.to_str().unwrap()])
            .args(args)
            .status()?;
        if !status.success() {
            anyhow::bail!("docker compose {} failed", args.join(" "));
        }
        Ok(())
    };

    match action {
        DepsAction::Up => {
            println!("Starting Brain external services...");
            run(&["up", "-d"])?;
            println!("\nServices started:");
            println!("  SearXNG → http://127.0.0.1:8888");
            println!("\nRun `brain status` to verify connectivity.");
        }
        DepsAction::Down => {
            println!("Stopping Brain external services...");
            run(&["down"])?;
            println!("Services stopped.");
        }
        DepsAction::Status => {
            run(&["ps"])?;
        }
    }
    Ok(())
}

// ─── Service install / uninstall ──────────────────────────────────────────────

/// Install Brain as a login service so it auto-starts on every login.
///
/// • macOS   — creates `~/Library/LaunchAgents/com.brain.plist` and loads it
///             with `launchctl load`.
/// • Linux   — creates `~/.config/systemd/user/brain.service`, runs
///             `systemctl --user daemon-reload` and `systemctl --user enable`.
/// • Windows — registers a Task Scheduler task (`schtasks`) that runs
///             `brain serve` at every user login, no admin required.
fn cmd_service_install() -> anyhow::Result<()> {
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

        std::fs::write(&plist_path, &plist)?;

        // Unload first (ignore error if not loaded yet) then load
        let _ = std::process::Command::new("launchctl")
            .args(["unload", plist_path.to_str().unwrap()])
            .output();
        let out = std::process::Command::new("launchctl")
            .args(["load", "-w", plist_path.to_str().unwrap()])
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
        // Windows Task Scheduler — runs brain serve at every user login.
        // No admin rights required for /SC ONLOGON tasks in the current user's account.
        let task_name = "Brain OS";
        let cmd = format!("{exe_str} serve");

        // /F overwrites any existing task with the same name (idempotent)
        let out = std::process::Command::new("schtasks")
            .args([
                "/Create", "/TN", task_name, "/TR", &cmd, "/SC", "ONLOGON", "/RL",
                "HIGHEST", // run with the highest privileges the user has
                "/F",      // overwrite if already exists
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("schtasks not found: {e}"))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            anyhow::bail!("schtasks /Create failed: {stderr}");
        }

        // Start immediately so the user doesn't have to log out and back in
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

/// Remove the Brain login service.
fn cmd_service_uninstall() -> anyhow::Result<()> {
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

        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w", plist_path.to_str().unwrap()])
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

        // Stop the running task first (ignore error if not running)
        let _ = std::process::Command::new("schtasks")
            .args(["/End", "/TN", task_name])
            .output();

        let out = std::process::Command::new("schtasks")
            .args(["/Delete", "/TN", task_name, "/F"])
            .output()
            .map_err(|e| anyhow::anyhow!("schtasks not found: {e}"))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // Treat "task not found" as a no-op so uninstall is idempotent
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

// ─── Export / Import ──────────────────────────────────────────────────────────

/// JSON envelope written / read by `brain export` / `brain import`.
#[derive(serde::Serialize, serde::Deserialize)]
struct MemoryExport {
    version: String,
    exported_at: String,
    facts: Vec<ExportFact>,
    episodes: Vec<ExportEpisode>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportFact {
    id: String,
    namespace: String,
    category: String,
    subject: String,
    predicate: String,
    object: String,
    confidence: f64,
    source_episode_id: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ExportEpisode {
    id: String,
    session_id: String,
    session_channel: String,
    #[serde(default = "default_export_namespace")]
    namespace: String,
    role: String,
    content: String,
    timestamp: String,
    importance: f64,
    reinforcement_count: i32,
}

fn default_export_namespace() -> String {
    "personal".to_string()
}

fn cmd_export(config: &brain_core::BrainConfig, output: Option<&str>) -> anyhow::Result<()> {
    let db = storage::SqlitePool::open(&config.sqlite_path())?;

    // Query all semantic facts
    let facts: Vec<ExportFact> = db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT id, namespace, category, subject, predicate, object,
                    confidence, source_episode_id
             FROM semantic_facts
             ORDER BY id ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ExportFact {
                    id: row.get(0)?,
                    namespace: row.get(1)?,
                    category: row.get(2)?,
                    subject: row.get(3)?,
                    predicate: row.get(4)?,
                    object: row.get(5)?,
                    confidence: row.get(6)?,
                    source_episode_id: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    })?;

    // Query all episodes with their session channel
    let episodes: Vec<ExportEpisode> = db.with_conn(|conn| {
        let mut stmt = conn.prepare(
            "SELECT e.id, e.session_id, COALESCE(s.channel, 'cli'),
                    e.namespace, e.role, e.content, e.timestamp,
                    e.importance, e.reinforcement_count
             FROM episodes e
             LEFT JOIN sessions s ON s.id = e.session_id
             ORDER BY e.timestamp ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(ExportEpisode {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    session_channel: row.get(2)?,
                    namespace: row.get(3)?,
                    role: row.get(4)?,
                    content: row.get(5)?,
                    timestamp: row.get(6)?,
                    importance: row.get(7)?,
                    reinforcement_count: row.get(8)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    })?;

    let n_facts = facts.len();
    let n_episodes = episodes.len();

    let export = MemoryExport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        exported_at: chrono::Utc::now().to_rfc3339(),
        facts,
        episodes,
    };

    let json = serde_json::to_string_pretty(&export)?;

    match output {
        Some(path) => {
            std::fs::write(path, &json)?;
            println!("Exported {n_facts} facts and {n_episodes} episodes to {path}");
        }
        None => {
            println!("{}", json);
        }
    }

    Ok(())
}

async fn cmd_import(
    config: &brain_core::BrainConfig,
    file: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let raw =
        std::fs::read_to_string(file).map_err(|e| anyhow::anyhow!("Cannot read {file}: {e}"))?;
    let export: MemoryExport =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("Invalid export file: {e}"))?;

    println!(
        "Import preview: {} facts, {} episodes (exported at {})",
        export.facts.len(),
        export.episodes.len(),
        export.exported_at,
    );

    if dry_run {
        println!("Dry-run: no changes written.");
        return Ok(());
    }

    let db = storage::SqlitePool::open(&config.sqlite_path())?;

    // Collect unique sessions needed for episodes
    let mut sessions: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for ep in &export.episodes {
        sessions
            .entry(ep.session_id.clone())
            .or_insert_with(|| ep.session_channel.clone());
    }

    let mut facts_imported = 0usize;
    let mut episodes_imported = 0usize;

    // Collect IDs of newly imported facts for re-embedding
    let mut new_fact_ids: Vec<usize> = Vec::new();

    db.with_conn(|conn| {
        // Insert sessions (skip if already exist)
        for (sid, channel) in &sessions {
            conn.execute(
                "INSERT INTO sessions (id, channel) VALUES (?1, ?2)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![sid, channel],
            )?;
        }

        // Insert facts (skip duplicates by id)
        for (idx, f) in export.facts.iter().enumerate() {
            let n = conn.execute(
                "INSERT INTO semantic_facts
                    (id, namespace, category, subject, predicate, object,
                     confidence, source_episode_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![
                    f.id,
                    f.namespace,
                    f.category,
                    f.subject,
                    f.predicate,
                    f.object,
                    f.confidence,
                    f.source_episode_id
                ],
            )?;
            if n > 0 {
                new_fact_ids.push(idx);
            }
            facts_imported += n;
        }

        // Insert episodes (skip duplicates by id)
        for e in &export.episodes {
            let n = conn.execute(
                "INSERT INTO episodes
                    (id, session_id, namespace, role, content, timestamp,
                     importance, reinforcement_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![
                    e.id,
                    e.session_id,
                    e.namespace,
                    e.role,
                    e.content,
                    e.timestamp,
                    e.importance,
                    e.reinforcement_count
                ],
            )?;
            episodes_imported += n;
        }

        Ok(())
    })?;

    println!(
        "Imported: {} new facts, {} new episodes ({} facts and {} episodes already existed).",
        facts_imported,
        episodes_imported,
        export.facts.len() - facts_imported,
        export.episodes.len() - episodes_imported,
    );

    // Re-embed newly imported facts into RuVector so they're visible to vector search
    if !new_fact_ids.is_empty() {
        let embedding_dim = config.embedding.dimensions as usize;
        let ruv_result = storage::RuVectorStore::open(&config.ruvector_path(), embedding_dim).await;

        match ruv_result {
            Ok(ruv) => {
                ruv.ensure_tables().await.ok();

                // Create embedder
                let embedder = match config.llm.provider.as_str() {
                    "openai" => {
                        let api_key = std::env::var("BRAIN_LLM__API_KEY").unwrap_or_default();
                        hippocampus::Embedder::for_openai(
                            &config.llm.base_url,
                            &config.embedding.model,
                            &api_key,
                        )
                    }
                    _ => hippocampus::Embedder::for_ollama(
                        &config.llm.base_url,
                        &config.embedding.model,
                    ),
                };

                let mut embedded = 0usize;
                let mut failed = 0usize;

                for &idx in &new_fact_ids {
                    let f = &export.facts[idx];
                    let text = format!("{} {} {}", f.subject, f.predicate, f.object);

                    match embedder.embed(&text).await {
                        Ok(vector) => {
                            let now = chrono::Utc::now().to_rfc3339();
                            if let Err(e) = ruv
                                .add_vectors(
                                    "facts_vec",
                                    vec![f.id.clone()],
                                    vec![text],
                                    vec![vector],
                                    vec![now],
                                    "semantic",
                                )
                                .await
                            {
                                tracing::warn!("RuVector insert failed for fact {}: {e}", f.id);
                                failed += 1;
                            } else {
                                embedded += 1;
                            }
                        }
                        Err(e) => {
                            if embedded == 0 && failed == 0 {
                                // First failure — likely embedding service is down
                                println!(
                                    "Warning: Embedding unavailable ({e}). \
                                     Imported facts will not appear in vector search until re-embedded."
                                );
                                break;
                            }
                            failed += 1;
                        }
                    }
                }

                if embedded > 0 {
                    println!("Re-embedded {embedded} facts into vector index.");
                }
                if failed > 0 {
                    println!("Warning: {failed} facts failed to embed.");
                }
            }
            Err(e) => {
                println!(
                    "Warning: RuVector unavailable ({e}). \
                     Imported facts visible in SQLite but not vector search."
                );
            }
        }
    }

    Ok(())
}

async fn promote_candidates(
    processor: &signal::SignalProcessor,
    candidates: &[hippocampus::PromotionCandidate],
) -> usize {
    let mut promoted_now = 0usize;

    for candidate in candidates {
        let already_promoted = processor
            .episodic()
            .pool()
            .with_conn(|conn| {
                let exists: i64 = conn.query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM episode_promotions
                        WHERE episode_id = ?1
                    )",
                    [&candidate.episode_id],
                    |row| row.get(0),
                )?;
                Ok(exists > 0)
            })
            .unwrap_or(false);

        if already_promoted {
            continue;
        }

        let (subject, predicate, object) =
            thalamus::IntentClassifier::parse_store_fact_content(&candidate.content);

        if object.trim().is_empty() {
            continue;
        }

        match processor
            .store_fact_direct(
                &candidate.namespace,
                "consolidated",
                &subject,
                &predicate,
                &object,
            )
            .await
        {
            Ok(fact_id) => {
                if let Err(e) = processor.episodic().pool().with_conn(|conn| {
                    conn.execute(
                        "INSERT INTO episode_promotions (episode_id, fact_id)
                         VALUES (?1, ?2)
                         ON CONFLICT(episode_id) DO NOTHING",
                        rusqlite::params![&candidate.episode_id, fact_id],
                    )?;
                    Ok(())
                }) {
                    tracing::warn!(
                        episode_id = %candidate.episode_id,
                        "Failed to persist promotion marker: {e}"
                    );
                } else {
                    promoted_now += 1;
                }
            }
            Err(e) => tracing::warn!(
                episode_id = %candidate.episode_id,
                "Failed to promote episode: {e}"
            ),
        }
    }

    promoted_now
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

async fn chat_non_interactive(
    config: &brain_core::BrainConfig,
    message: &str,
) -> anyhow::Result<()> {
    use futures::StreamExt;

    let mut brain = BrainSession::new(config).await?;

    // Drain pending outbox notifications before responding
    show_proactive_nudges(&brain, config);

    match brain.prepare_context(message).await? {
        PrepareResult::ActionResult(text) => {
            println!("{text}");
        }
        PrepareResult::LlmReady(messages) => {
            // Try streaming first
            match brain.llm.generate_stream(&messages).await {
                Ok(mut stream) => {
                    let mut full_response = String::new();
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(c) => {
                                print!("{}", c.content);
                                let _ = stdout().flush();
                                full_response.push_str(&c.content);
                                if c.is_done {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("\nStream error: {e}");
                                break;
                            }
                        }
                    }
                    println!();
                    brain.finalize_response(message, &full_response)?;
                }
                Err(_) => {
                    // Fallback to non-streaming
                    let response = brain.llm.generate(&messages).await?;
                    println!("{}", response.content);
                    brain.finalize_response(message, &response.content)?;
                }
            }
        }
    }
    Ok(())
}

/// Display proactive nudges at session start: pending outbox items and open loops.
fn show_proactive_nudges(brain: &BrainSession, config: &brain_core::BrainConfig) {
    let mut nudges: Vec<String> = Vec::new();

    // 1. Drain pending outbox notifications
    if let Ok(pending) = brain.db().pending_notifications(5) {
        for n in &pending {
            nudges.push(n.content.clone());
            let _ = brain.db().mark_notification_delivered(&n.id);
        }
    }

    // 2. Run open-loop detector inline (if enabled)
    if config.proactivity.enabled && config.proactivity.open_loop.enabled {
        let detector = ganglia::OpenLoopDetector::new(
            brain.db().clone(),
            ganglia::OpenLoopConfig {
                scan_window_hours: config.proactivity.open_loop.scan_window_hours,
                resolution_window_hours: config.proactivity.open_loop.resolution_window_hours,
                max_reminders: 3,
            },
        );
        if let Ok(reminders) = detector.generate_reminders() {
            for r in reminders {
                nudges.push(r.content);
            }
        }
    }

    if !nudges.is_empty() {
        println!("\x1b[33m📌 Nudges:\x1b[0m");
        for nudge in &nudges {
            println!("  \x1b[33m• {nudge}\x1b[0m");
        }
        println!();
    }
}

async fn chat_interactive(config: &brain_core::BrainConfig) -> anyhow::Result<()> {
    let ver = env!("CARGO_PKG_VERSION");
    let title = format!("Brain v{ver}");
    let tagline = "Your AI's long-term memory";
    let width = 37; // inner width between ║ borders
    println!("╔═{}═╗", "═".repeat(width));
    println!("║ {:^w$} ║", title, w = width);
    println!("║ {:^w$} ║", tagline, w = width);
    println!("╚═{}═╝", "═".repeat(width));
    println!();
    println!("  Cortex:  {}", config.llm.model);
    println!("  Memory:  {}", config.data_dir().display());
    println!();
    println!("Signals: /status  /clear  /quit");
    println!();

    let mut brain = BrainSession::new(config).await?;
    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);

    // Show proactive nudges at session start (open loops + pending outbox)
    show_proactive_nudges(&brain, config);

    loop {
        match rl.readline("You: ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(input);

                match input {
                    "/quit" | "/exit" | "/q" => {
                        println!("Going dormant...");
                        break;
                    }
                    "/status" => {
                        show_status(config).await?;
                        continue;
                    }
                    "/clear" => {
                        brain.clear_history();
                        println!("Short-term memory cleared.");
                        continue;
                    }
                    s if s.starts_with('/') => {
                        println!("Unknown signal: {s}");
                        println!("Available: /status  /clear  /quit");
                        continue;
                    }
                    _ => {}
                }

                // Spinner runs from recall through LLM prompt processing,
                // until the first token arrives (or an action completes).
                let phase = Arc::new(std::sync::atomic::AtomicU8::new(0));
                let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let phase_c = Arc::clone(&phase);
                let stop_c = Arc::clone(&stop);
                let spinner_handle = tokio::spawn(async move {
                    let frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
                    let mut i = 0;
                    while !stop_c.load(std::sync::atomic::Ordering::Relaxed) {
                        let label = match phase_c.load(std::sync::atomic::Ordering::Relaxed) {
                            0 => "Recalling memories",
                            _ => "Thinking",
                        };
                        {
                            let mut out = stdout();
                            let _ = out.execute(cursor::MoveToColumn(0));
                            let _ = out.execute(terminal::Clear(terminal::ClearType::CurrentLine));
                            let _ = write!(
                                out,
                                "\x1b[90m  {} {}\x1b[0m",
                                frames[i % frames.len()],
                                label
                            );
                            let _ = out.flush();
                        }
                        i += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                    }
                    let mut out = stdout();
                    let _ = out.execute(cursor::MoveToColumn(0));
                    let _ = out.execute(terminal::Clear(terminal::ClearType::CurrentLine));
                    let _ = out.flush();
                });

                let prepare_result = brain.prepare_context(input).await;

                // Wrap handle in Option so we can .take() it exactly once
                let mut spinner_handle = Some(spinner_handle);
                let dismiss_spinner =
                    |stop: &Arc<std::sync::atomic::AtomicBool>,
                     handle: &mut Option<tokio::task::JoinHandle<()>>| {
                        stop.store(true, std::sync::atomic::Ordering::Relaxed);
                        handle.take() // returns the JoinHandle (if not already taken)
                    };

                match prepare_result {
                    Ok(PrepareResult::ActionResult(text)) => {
                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                            let _ = h.await;
                        }
                        let mut out = stdout();
                        out.execute(SetForegroundColor(Color::Green))?;
                        out.execute(Print("Brain: "))?;
                        out.execute(ResetColor)?;
                        println!("{text}");
                    }
                    Ok(PrepareResult::LlmReady(messages)) => {
                        // Switch spinner to "Thinking" while we wait for first token
                        phase.store(1, std::sync::atomic::Ordering::Relaxed);

                        // Try streaming — spinner keeps running until first token
                        let stream_result = brain.llm.generate_stream(&messages).await;
                        match stream_result {
                            Ok(mut stream) => {
                                use futures::StreamExt;
                                let mut full_response = String::new();
                                while let Some(chunk) = stream.next().await {
                                    match chunk {
                                        Ok(c) => {
                                            // Stop spinner & print prefix on first token
                                            if let Some(h) =
                                                dismiss_spinner(&stop, &mut spinner_handle)
                                            {
                                                let _ = h.await;
                                                let mut out = stdout();
                                                out.execute(SetForegroundColor(Color::Green))?;
                                                out.execute(Print("Brain: "))?;
                                                out.execute(ResetColor)?;
                                                let _ = out.flush();
                                            }
                                            print!("{}", c.content);
                                            let _ = stdout().flush();
                                            full_response.push_str(&c.content);
                                            if c.is_done {
                                                break;
                                            }
                                        }
                                        Err(e) => {
                                            if let Some(h) =
                                                dismiss_spinner(&stop, &mut spinner_handle)
                                            {
                                                let _ = h.await;
                                            }
                                            eprintln!("\nStream error: {e}");
                                            break;
                                        }
                                    }
                                }
                                // If stream ended with zero tokens, dismiss spinner
                                if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                                    let _ = h.await;
                                }
                                println!();
                                if let Err(e) = brain.finalize_response(input, &full_response) {
                                    tracing::warn!("Failed to store response: {e}");
                                }
                            }
                            Err(_) => {
                                // Fallback to non-streaming (spinner keeps running)
                                match brain.llm.generate(&messages).await {
                                    Ok(response) => {
                                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle)
                                        {
                                            let _ = h.await;
                                        }
                                        let mut out = stdout();
                                        out.execute(SetForegroundColor(Color::Green))?;
                                        out.execute(Print("Brain: "))?;
                                        out.execute(ResetColor)?;
                                        println!("{}", response.content);
                                        if let Err(e) =
                                            brain.finalize_response(input, &response.content)
                                        {
                                            tracing::warn!("Failed to store response: {e}");
                                        }
                                    }
                                    Err(e) => {
                                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle)
                                        {
                                            let _ = h.await;
                                        }
                                        let msg = e.to_string();
                                        if msg.contains("timed out") || msg.contains("Timeout") {
                                            eprintln!("LLM timed out — model may still be loading. Try again.");
                                        } else if msg.contains("error sending request")
                                            || msg.contains("connection refused")
                                            || msg.contains("Connection refused")
                                        {
                                            eprintln!("LLM unreachable — is Ollama running? (`ollama serve`)");
                                        } else {
                                            eprintln!("Error: {msg}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if let Some(h) = dismiss_spinner(&stop, &mut spinner_handle) {
                            let _ = h.await;
                        }
                        let msg = e.to_string();
                        if msg.contains("timed out") || msg.contains("Timeout") {
                            eprintln!("LLM timed out — model may still be loading. Try again.");
                        } else if msg.contains("error sending request")
                            || msg.contains("connection refused")
                            || msg.contains("Connection refused")
                        {
                            eprintln!("LLM unreachable — is Ollama running? (`ollama serve`)");
                        } else {
                            eprintln!("Error: {msg}");
                        }
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                println!("Going dormant...");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}

// ─── BrainSession ─────────────────────────────────────────────────────────────

/// Result of context preparation (Phase 0).
enum PrepareResult {
    /// Thalamus dispatched an action — response is ready.
    ActionResult(String),
    /// Messages are assembled and ready for the LLM.
    LlmReady(Vec<cortex::llm::Message>),
}

#[derive(Clone)]
struct CliMemoryBackend {
    semantic: Option<hippocampus::SemanticStore>,
    embedder: Arc<tokio::sync::Mutex<Option<hippocampus::Embedder>>>,
    embedding_dim: usize,
}

#[async_trait::async_trait]
impl cortex::actions::MemoryBackend for CliMemoryBackend {
    async fn store_fact(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let content = format!("{subject} {predicate} {object}");
        let vector = {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(&content).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, &content)
                    }
                    Err(e) => {
                        tracing::warn!("CLI ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            &content,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(
                    &content,
                    self.embedding_dim,
                )
            }
        };

        semantic
            .store_fact(
                namespace, category, subject, predicate, object, 1.0, None, vector, None,
            )
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))
    }

    async fn recall(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<cortex::actions::MemoryFact>, cortex::actions::ActionError> {
        let Some(semantic) = &self.semantic else {
            return Err(cortex::actions::ActionError::ExecutionFailed(
                "Semantic store unavailable".to_string(),
            ));
        };

        let vector = {
            let mut guard = self.embedder.lock().await;
            if let Some(embedder) = guard.as_mut() {
                match embedder.embed(query).await {
                    Ok(v) => {
                        hippocampus::embedding::sanitize_embedding(v, self.embedding_dim, query)
                    }
                    Err(e) => {
                        tracing::warn!("CLI ActionDispatcher embedding failed: {e}");
                        hippocampus::embedding::deterministic_fallback_embedding(
                            query,
                            self.embedding_dim,
                        )
                    }
                }
            } else {
                hippocampus::embedding::deterministic_fallback_embedding(query, self.embedding_dim)
            }
        };

        let results = semantic
            .search_similar(vector, top_k.max(1), namespace, None)
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| cortex::actions::MemoryFact {
                namespace: r.fact.namespace,
                subject: r.fact.subject,
                predicate: r.fact.predicate,
                object: r.fact.object,
                confidence: r.fact.confidence,
            })
            .collect())
    }
}

// ─── Resilience: retry + circuit breaker ─────────────────────────────────────

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Tracks consecutive failures and opens a circuit after a threshold is reached.
/// While the circuit is open, requests fail immediately without hitting the network.
struct CircuitBreaker {
    consecutive_failures: AtomicU32,
    /// Epoch millis of the last failure (used for cooldown calculation).
    last_failure_epoch_ms: AtomicU64,
    threshold: u32,
    cooldown_ms: u64,
    name: String,
}

impl CircuitBreaker {
    fn new(name: &str, threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            last_failure_epoch_ms: AtomicU64::new(0),
            threshold,
            cooldown_ms: cooldown_secs * 1000,
            name: name.to_string(),
        }
    }

    fn is_open(&self) -> bool {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures < self.threshold {
            return false;
        }
        // Check if cooldown has elapsed
        let last_fail = self.last_failure_epoch_ms.load(Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        if now.saturating_sub(last_fail) >= self.cooldown_ms {
            // Cooldown elapsed — allow a probe request (half-open)
            return false;
        }
        true
    }

    fn record_success(&self) {
        let prev = self.consecutive_failures.swap(0, Ordering::Relaxed);
        if prev >= self.threshold {
            tracing::info!(backend = %self.name, "Circuit breaker closed (backend recovered)");
        }
    }

    fn record_failure(&self) {
        let prev = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_failure_epoch_ms.store(now, Ordering::Relaxed);
        if prev + 1 == self.threshold {
            tracing::warn!(
                backend = %self.name,
                threshold = self.threshold,
                cooldown_secs = self.cooldown_ms / 1000,
                "Circuit breaker OPEN — backend disabled until cooldown elapses"
            );
        }
    }
}

/// Returns true if the error is transient (worth retrying).
fn is_transient(err: &reqwest::Error) -> bool {
    if err.is_timeout() || err.is_connect() {
        return true;
    }
    if let Some(status) = err.status() {
        return status.is_server_error(); // 5xx
    }
    false
}

/// Returns true if the HTTP status is transient (worth retrying).
fn is_transient_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error() // 5xx
}

/// Send an HTTP request with retry + circuit breaker.
///
/// `build_request` is called each attempt to produce a fresh `RequestBuilder`
/// (since `RequestBuilder` is consumed on `.send()`).
async fn resilient_send<F>(
    build_request: F,
    circuit_breaker: &CircuitBreaker,
    max_retries: u32,
    retry_base_ms: u64,
) -> Result<reqwest::Response, cortex::actions::ActionError>
where
    F: Fn() -> reqwest::RequestBuilder,
{
    if circuit_breaker.is_open() {
        return Err(cortex::actions::ActionError::ExecutionFailed(format!(
            "{} circuit breaker is open — backend disabled until cooldown elapses",
            circuit_breaker.name
        )));
    }

    let attempts = 1 + max_retries;
    let mut last_err = None;

    for attempt in 0..attempts {
        if attempt > 0 {
            let delay = retry_base_ms * (1u64 << (attempt - 1).min(5));
            tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
        }

        match build_request().send().await {
            Ok(response) => {
                if response.status().is_success() || !is_transient_status(response.status()) {
                    // Success or non-retryable status (4xx) — return as-is
                    circuit_breaker.record_success();
                    return Ok(response);
                }
                // 5xx — retryable
                let status = response.status();
                tracing::debug!(
                    backend = %circuit_breaker.name,
                    attempt = attempt + 1,
                    status = %status,
                    "Transient HTTP error, will retry"
                );
                last_err = Some(format!("HTTP {}", status));
            }
            Err(e) => {
                if !is_transient(&e) {
                    // Non-transient error — fail immediately
                    circuit_breaker.record_failure();
                    return Err(cortex::actions::ActionError::ExecutionFailed(e.to_string()));
                }
                tracing::debug!(
                    backend = %circuit_breaker.name,
                    attempt = attempt + 1,
                    error = %e,
                    "Transient error, will retry"
                );
                last_err = Some(e.to_string());
            }
        }
    }

    // All attempts exhausted
    circuit_breaker.record_failure();
    Err(cortex::actions::ActionError::ExecutionFailed(
        last_err.unwrap_or_else(|| "all retry attempts exhausted".to_string()),
    ))
}

// ─── Web Search Backends ─────────────────────────────────────────────────────

/// Parse a JSON array of search results into `SearchHit`s with flexible field names.
fn parse_search_results(
    candidates: Vec<serde_json::Value>,
    top_k: usize,
) -> Vec<cortex::actions::SearchHit> {
    candidates
        .into_iter()
        .filter_map(|entry| {
            let title = entry
                .get("title")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("name").and_then(serde_json::Value::as_str))
                .unwrap_or("untitled")
                .to_string();
            let url = entry
                .get("url")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("link").and_then(serde_json::Value::as_str))
                .unwrap_or_default()
                .to_string();
            if url.is_empty() {
                return None;
            }
            let snippet = entry
                .get("snippet")
                .and_then(serde_json::Value::as_str)
                .or_else(|| entry.get("description").and_then(serde_json::Value::as_str))
                .or_else(|| entry.get("content").and_then(serde_json::Value::as_str))
                .unwrap_or_default()
                .to_string();
            Some(cortex::actions::SearchHit {
                title,
                url,
                snippet,
            })
        })
        .take(top_k.max(1))
        .collect()
}

fn build_search_client(timeout_ms: u64) -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
        .build()
        .map_err(|e| anyhow::anyhow!("search client init failed: {e}"))
}

/// SearXNG provider — self-hosted metasearch engine.
/// GET `{endpoint}/search?q={query}&format=json`
struct SearxngSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl SearxngSearchBackend {
    fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "searxng",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for SearxngSearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let url = format!("{}/search", self.endpoint);
        let client = self.client.clone();
        let url_clone = url.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client
                    .get(&url_clone)
                    .query(&[("q", query_owned.as_str()), ("format", "json")])
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "SearXNG returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates = match body.get("results").and_then(|v| v.as_array()) {
            Some(arr) => arr.clone(),
            None => {
                tracing::warn!(backend = "searxng", "Response missing 'results' array — returning empty");
                Vec::new()
            }
        };

        Ok(parse_search_results(candidates, top_k))
    }
}

/// Tavily provider — AI-focused search API (1000 free/month, no CC).
/// POST `{endpoint}/search` with Bearer auth.
struct TavilySearchBackend {
    endpoint: String,
    api_key: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl TavilySearchBackend {
    fn new(
        endpoint: &str,
        api_key: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "tavily",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for TavilySearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let url = format!("{}/search", self.endpoint);
        let client = self.client.clone();
        let url_clone = url.clone();
        let api_key = self.api_key.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client
                    .post(&url_clone)
                    .bearer_auth(&api_key)
                    .json(&serde_json::json!({
                        "query": query_owned,
                        "max_results": top_k,
                        "search_depth": "basic",
                    }))
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "Tavily returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates = match body.get("results").and_then(|v| v.as_array()) {
            Some(arr) => {
                // Schema validation: Tavily results should have `url` fields
                if !arr.is_empty() && arr[0].get("url").is_none() {
                    tracing::warn!(backend = "tavily", "Results missing 'url' field — response schema may have changed");
                }
                arr.clone()
            }
            None => {
                tracing::warn!(backend = "tavily", "Response missing 'results' array — returning empty");
                Vec::new()
            }
        };

        Ok(parse_search_results(candidates, top_k))
    }
}

/// Custom provider — raw JSON POST to a user-configured endpoint (backward-compatible).
struct CustomSearchBackend {
    endpoint: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl CustomSearchBackend {
    fn new(
        endpoint: &str,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            endpoint: endpoint.to_string(),
            client: build_search_client(timeout_ms)?,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "custom-search",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl cortex::actions::WebSearchBackend for CustomSearchBackend {
    async fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<cortex::actions::SearchHit>, cortex::actions::ActionError> {
        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let query_owned = query.to_string();
        let response = resilient_send(
            || {
                client.post(&endpoint).json(&serde_json::json!({
                    "query": query_owned,
                    "top_k": top_k,
                }))
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "search endpoint returned HTTP {}",
                response.status()
            )));
        }

        let body = response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        let candidates: Vec<serde_json::Value> = body
            .get("hits")
            .and_then(|v| v.as_array())
            .cloned()
            .or_else(|| body.get("results").and_then(|v| v.as_array()).cloned())
            .or_else(|| body.as_array().cloned())
            .unwrap_or_default();

        Ok(parse_search_results(candidates, top_k))
    }
}

#[derive(Clone)]
struct CliSchedulingBackend {
    db: storage::SqlitePool,
    mode: brain_core::config::SchedulingMode,
}

#[async_trait::async_trait]
impl cortex::actions::SchedulingBackend for CliSchedulingBackend {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<cortex::actions::ScheduleOutcome, cortex::actions::ActionError> {
        if self.mode != brain_core::config::SchedulingMode::PersistOnly {
            return Err(cortex::actions::ActionError::InvalidArguments(format!(
                "Unsupported scheduling mode: {:?}",
                self.mode
            )));
        }

        let metadata = serde_json::json!({
            "source": "action_dispatcher",
            "mode": "persist_only",
        })
        .to_string();

        let schedule_id = self
            .db
            .insert_scheduled_intent(description, cron, namespace, Some(&metadata))
            .map_err(|e| cortex::actions::ActionError::ExecutionFailed(e.to_string()))?;

        Ok(cortex::actions::ScheduleOutcome {
            schedule_id,
            status: "scheduled".to_string(),
        })
    }
}

// ─── Message backend ─────────────────────────────────────────────────────────

const DEFAULT_MESSAGE_BODY: &str = r#"{"channel":"{{channel}}","recipient":"{{recipient}}","content":"{{content}}","namespace":"{{namespace}}","timestamp":"{{timestamp}}"}"#;

/// JSON-escape a string value (without surrounding quotes).
fn json_escape(s: &str) -> String {
    let escaped = serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s));
    escaped[1..escaped.len() - 1].to_string()
}

/// Render a message template by replacing `{{placeholder}}` tokens.
fn render_message_template(
    template: &str,
    channel: &str,
    recipient: &str,
    content: &str,
    namespace: &str,
    timestamp: &str,
) -> String {
    template
        .replace("{{channel}}", &json_escape(channel))
        .replace("{{recipient}}", &json_escape(recipient))
        .replace("{{content}}", &json_escape(content))
        .replace("{{namespace}}", &json_escape(namespace))
        .replace("{{timestamp}}", &json_escape(timestamp))
}

struct WebhookMessageBackend {
    channels: HashMap<String, brain_core::config::ChannelConfig>,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    max_retries: u32,
    retry_base_ms: u64,
}

impl WebhookMessageBackend {
    fn new(
        channels: &HashMap<String, brain_core::config::ChannelConfig>,
        timeout_ms: u64,
        resilience: &brain_core::config::ResilienceConfig,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(timeout_ms.max(1)))
            .build()
            .map_err(|e| anyhow::anyhow!("message client init failed: {e}"))?;
        Ok(Self {
            channels: channels
                .iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
                .collect(),
            client,
            circuit_breaker: Arc::new(CircuitBreaker::new(
                "webhook-message",
                resilience.circuit_breaker_threshold,
                resilience.circuit_breaker_cooldown_secs,
            )),
            max_retries: resilience.max_retries,
            retry_base_ms: resilience.retry_base_ms,
        })
    }
}

#[async_trait::async_trait]
impl signal::notification::WebhookSender for WebhookMessageBackend {
    async fn send_notification(
        &self,
        channel: &str,
        content: &str,
        namespace: &str,
    ) -> Result<(), String> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| format!("No webhook mapping for channel '{channel}'"))?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    "",
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await
        .map_err(|e| format!("webhook send failed: {e}"))?;

        if !response.status().is_success() {
            return Err(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            ));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl cortex::actions::MessageBackend for WebhookMessageBackend {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<cortex::actions::MessageOutcome, cortex::actions::ActionError> {
        let channel_cfg = self
            .channels
            .get(&channel.to_ascii_lowercase())
            .ok_or_else(|| {
                cortex::actions::ActionError::InvalidArguments(format!(
                    "No webhook mapping for channel '{}'",
                    channel
                ))
            })?
            .clone();

        let client = self.client.clone();
        let template = if channel_cfg.body.is_empty() {
            DEFAULT_MESSAGE_BODY.to_string()
        } else {
            channel_cfg.body.clone()
        };
        let url = channel_cfg.url.clone();
        let headers = channel_cfg.headers.clone();
        let channel_owned = channel.to_string();
        let recipient_owned = recipient.to_string();
        let content_owned = content.to_string();
        let namespace_owned = namespace.to_string();

        let response = resilient_send(
            || {
                let timestamp = chrono::Utc::now().to_rfc3339();
                let rendered = render_message_template(
                    &template,
                    &channel_owned,
                    &recipient_owned,
                    &content_owned,
                    &namespace_owned,
                    &timestamp,
                );
                let mut req = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(rendered);
                for (key, value) in &headers {
                    req = req.header(key.as_str(), value.as_str());
                }
                req
            },
            &self.circuit_breaker,
            self.max_retries,
            self.retry_base_ms,
        )
        .await?;

        if !response.status().is_success() {
            return Err(cortex::actions::ActionError::ExecutionFailed(format!(
                "webhook for channel '{}' returned HTTP {}",
                channel,
                response.status()
            )));
        }

        let body = response.text().await.unwrap_or_default();
        let mut delivery_id = format!("msg-{}", chrono::Utc::now().timestamp_micros());
        let mut status = "accepted".to_string();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
            if let Some(id) = value
                .get("id")
                .or_else(|| value.get("delivery_id"))
                .and_then(serde_json::Value::as_str)
            {
                delivery_id = id.to_string();
            }
            if let Some(s) = value.get("status").and_then(serde_json::Value::as_str) {
                status = s.to_string();
            }
        }

        Ok(cortex::actions::MessageOutcome {
            delivery_id,
            status,
        })
    }
}

#[allow(dead_code)]
struct BrainSession {
    _config: brain_core::BrainConfig,
    db: storage::SqlitePool,
    episodic: hippocampus::EpisodicStore,
    semantic: Option<hippocampus::SemanticStore>,
    embedder: Arc<tokio::sync::Mutex<Option<hippocampus::Embedder>>>,
    embedding_dim: usize,
    recall_engine: hippocampus::RecallEngine,
    llm: Box<dyn cortex::llm::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
    action_dispatcher: cortex::actions::ActionDispatcher,
    namespace: String,
    conversation_history: Vec<cortex::llm::Message>,
    session_id: String,
}

impl BrainSession {
    async fn new(config: &brain_core::BrainConfig) -> anyhow::Result<Self> {
        let db = storage::SqlitePool::open(&config.sqlite_path())?;
        let episodic = hippocampus::EpisodicStore::new(db.clone());

        let semantic = if let Ok(ruv) = storage::RuVectorStore::open(
            &config.ruvector_path(),
            config.embedding.dimensions as usize,
        )
        .await
        {
            ruv.ensure_tables().await.ok();
            Some(hippocampus::SemanticStore::new(db.clone(), ruv))
        } else {
            None
        };

        // Create embedder (same logic as SignalProcessor)
        let embedding_dim = config.embedding.dimensions as usize;
        let embedder = Arc::new(tokio::sync::Mutex::new(
            match config.llm.provider.as_str() {
                "openai" => {
                    let api_key = std::env::var("BRAIN_LLM__API_KEY").unwrap_or_default();
                    Some(hippocampus::Embedder::for_openai(
                        &config.llm.base_url,
                        &config.embedding.model,
                        &api_key,
                    ))
                }
                _ => Some(hippocampus::Embedder::for_ollama(
                    &config.llm.base_url,
                    &config.embedding.model,
                )),
            },
        ));

        let action_backend = Arc::new(CliMemoryBackend {
            semantic: semantic.clone(),
            embedder: Arc::clone(&embedder),
            embedding_dim,
        });
        let action_config = cortex::actions::ActionConfig {
            command_allowlist: config.security.exec_allowlist.clone(),
            command_timeout_secs: config.security.exec_timeout_seconds as u64,
            enable_web_search: config.actions.web_search.enabled,
            enable_scheduling: config.actions.scheduling.enabled,
            enable_channel_send: config.actions.messaging.enabled,
            web_search_top_k: config.actions.web_search.default_top_k,
        };
        let mut action_dispatcher =
            cortex::actions::ActionDispatcher::with_memory_backend(action_config, action_backend);
        action_dispatcher.set_namespace("personal");

        if config.actions.web_search.enabled {
            let ws = &config.actions.web_search;
            let timeout = ws.timeout_ms;
            let endpoint = ws.endpoint.trim();
            let res = &config.actions.resilience;

            let backend_result: Result<
                Option<Arc<dyn cortex::actions::WebSearchBackend>>,
                anyhow::Error,
            > = match ws.provider {
                brain_core::config::WebSearchProvider::Searxng => {
                    let ep = if endpoint.is_empty() {
                        "http://localhost:8888"
                    } else {
                        endpoint
                    };
                    SearxngSearchBackend::new(ep, timeout, res).map(|b| Some(Arc::new(b) as _))
                }
                brain_core::config::WebSearchProvider::Tavily => {
                    let api_key = ws.api_key.trim();
                    if api_key.is_empty() {
                        tracing::warn!("actions.web_search.provider=tavily but api_key is empty; backend not configured");
                        Ok(None)
                    } else {
                        let ep = if endpoint.is_empty() {
                            "https://api.tavily.com"
                        } else {
                            endpoint
                        };
                        TavilySearchBackend::new(ep, api_key, timeout, res)
                            .map(|b| Some(Arc::new(b) as _))
                    }
                }
                brain_core::config::WebSearchProvider::Custom => {
                    if endpoint.is_empty() {
                        tracing::warn!("actions.web_search.provider=custom but endpoint is empty; backend not configured");
                        Ok(None)
                    } else {
                        CustomSearchBackend::new(endpoint, timeout, res)
                            .map(|b| Some(Arc::new(b) as _))
                    }
                }
            };

            match backend_result {
                Ok(Some(backend)) => {
                    tracing::info!(
                        provider = %serde_json::to_string(&ws.provider).unwrap_or_default().trim_matches('"'),
                        "Web search backend configured"
                    );
                    action_dispatcher =
                        action_dispatcher.with_web_search_backend(backend);
                }
                Ok(None) => {} // warning already logged above
                Err(e) => tracing::warn!("Web search backend init failed: {e}"),
            }
        }

        if config.actions.scheduling.enabled {
            let backend = CliSchedulingBackend {
                db: db.clone(),
                mode: config.actions.scheduling.mode.clone(),
            };
            action_dispatcher = action_dispatcher.with_scheduling_backend(Arc::new(backend));
        }

        if config.actions.messaging.enabled {
            if config.actions.messaging.channels.is_empty() {
                tracing::warn!(
                    "actions.messaging.enabled=true but no channel webhook mappings are configured"
                );
            } else {
                let res = &config.actions.resilience;
                match WebhookMessageBackend::new(
                    &config.actions.messaging.channels,
                    config.actions.messaging.timeout_ms,
                    res,
                ) {
                    Ok(backend) => {
                        tracing::info!("Message backend configured");
                        action_dispatcher =
                            action_dispatcher.with_message_backend(Arc::new(backend));
                    }
                    Err(e) => tracing::warn!("Message backend init failed: {e}"),
                }
            }
        }

        let recall_engine = hippocampus::RecallEngine::with_defaults();

        let llm = cortex::llm::create_provider(&cortex::llm::ProviderConfig {
            provider: config.llm.provider.clone(),
            base_url: config.llm.base_url.clone(),
            api_key: None,
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens as i32,
        });

        let context_assembler = cortex::context::ContextAssembler::with_defaults();
        let session_id = episodic.create_session("cli")?;

        Ok(Self {
            _config: config.clone(),
            db,
            episodic,
            semantic,
            embedder,
            embedding_dim,
            recall_engine,
            llm,
            context_assembler,
            action_dispatcher,
            namespace: "personal".to_string(),
            conversation_history: Vec::new(),
            session_id,
        })
    }

    fn db(&self) -> &storage::SqlitePool {
        &self.db
    }

    fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Generate a vector embedding for text, falling back to a deterministic vector.
    async fn embed_text(&mut self, text: &str) -> Vec<f32> {
        let mut guard = self.embedder.lock().await;
        if let Some(ref mut embedder) = *guard {
            match embedder.embed(text).await {
                Ok(vec) => {
                    hippocampus::embedding::sanitize_embedding(vec, self.embedding_dim, text)
                }
                Err(e) => {
                    tracing::warn!("Embedding failed in CLI chat, using fallback vector: {e}");
                    hippocampus::embedding::deterministic_fallback_embedding(
                        text,
                        self.embedding_dim,
                    )
                }
            }
        } else {
            hippocampus::embedding::deterministic_fallback_embedding(text, self.embedding_dim)
        }
    }

    /// Phase 0: store user episode, route via thalamus, recall memories, assemble context.
    async fn prepare_context(&mut self, message: &str) -> anyhow::Result<PrepareResult> {
        let importance = hippocampus::ImportanceScorer::score(message, true);
        self.episodic.store_episode(
            &self.session_id,
            "user",
            message,
            importance,
            Some(&self.namespace),
            None,
        )?;

        let thalamus = thalamus::SignalRouter::new();
        let classification = thalamus
            .route(&thalamus::NormalizedMessage {
                content: message.to_string(),
                channel: "cli".to_string(),
                sender: "user".to_string(),
                timestamp: chrono::Utc::now(),
                message_id: None,
                metadata: std::collections::HashMap::new(),
            })
            .await;

        if let Some(action) = thalamus.intent_to_action(&classification.intent) {
            self.action_dispatcher.set_namespace(self.namespace.clone());
            let result = self.action_dispatcher.dispatch(&action).await;
            return if result.success {
                Ok(PrepareResult::ActionResult(result.output))
            } else {
                Ok(PrepareResult::ActionResult(format!(
                    "Error: {}",
                    result.error.unwrap_or_default()
                )))
            };
        }

        // Hybrid recall (BM25 + ANN via RecallEngine)
        let query_vector = self.embed_text(message).await;
        let memories = if let Some(semantic) = &self.semantic {
            match self
                .recall_engine
                .recall(
                    message,
                    query_vector,
                    &self.episodic,
                    semantic,
                    10,
                    Some(&self.namespace),
                    None,
                )
                .await
            {
                Ok(mems) => mems,
                Err(e) => {
                    tracing::warn!("Recall engine failed in CLI chat: {e}");
                    Vec::new()
                }
            }
        } else {
            self.episodic
                .search_bm25(message, 10, Some(&self.namespace), None)
                .unwrap_or_default()
                .into_iter()
                .map(|r| hippocampus::search::Memory {
                    id: r.episode_id,
                    content: r.content,
                    source: hippocampus::search::MemorySource::Episodic,
                    score: r.rank,
                    importance: 0.5,
                    timestamp: r.timestamp,
                    agent: r.agent,
                })
                .collect()
        };

        let messages =
            self.context_assembler
                .assemble(message, &memories, &self.conversation_history);

        Ok(PrepareResult::LlmReady(messages))
    }

    /// Store assistant response in episodic memory and conversation history.
    fn finalize_response(
        &mut self,
        user_message: &str,
        assistant_content: &str,
    ) -> anyhow::Result<()> {
        use cortex::llm::{Message, Role};

        self.episodic.store_episode(
            &self.session_id,
            "assistant",
            assistant_content,
            0.5,
            Some(&self.namespace),
            None,
        )?;

        self.conversation_history.push(Message {
            role: Role::User,
            content: user_message.to_string(),
        });
        self.conversation_history.push(Message {
            role: Role::Assistant,
            content: assistant_content.to_string(),
        });

        // Keep last 20 exchanges
        if self.conversation_history.len() > 40 {
            self.conversation_history = self.conversation_history.split_off(20);
        }

        Ok(())
    }

    /// Convenience wrapper: prepare_context → generate → finalize (non-streaming).
    #[allow(dead_code)]
    async fn process_message(&mut self, message: &str) -> anyhow::Result<String> {
        match self.prepare_context(message).await? {
            PrepareResult::ActionResult(text) => Ok(text),
            PrepareResult::LlmReady(messages) => {
                let response = self.llm.generate(&messages).await?;
                self.finalize_response(message, &response.content)?;
                Ok(response.content)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_promotion_idempotency_guard() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_str().unwrap().to_string();
        let processor = signal::SignalProcessor::new(config).await.unwrap();

        let session_id = processor.episodic().create_session("test").unwrap();
        let episode_id = processor
            .episodic()
            .store_episode(&session_id, "user", "project uses bun", 0.9, Some("work"), None)
            .unwrap();

        let candidates = vec![hippocampus::PromotionCandidate {
            episode_id,
            namespace: "work".to_string(),
            content: "project uses bun".to_string(),
            importance: 0.9,
            reinforcement_count: 3,
        }];

        let first = promote_candidates(&processor, &candidates).await;
        let second = promote_candidates(&processor, &candidates).await;

        assert_eq!(first, 1, "first promotion should persist");
        assert_eq!(second, 0, "second promotion should be skipped");
        assert_eq!(processor.list_facts(Some("work")).len(), 1);
    }

    #[tokio::test]
    async fn test_scheduling_backend_persists_intent() {
        let db = storage::SqlitePool::open_memory().unwrap();
        let backend = CliSchedulingBackend {
            db: db.clone(),
            mode: brain_core::config::SchedulingMode::PersistOnly,
        };

        let outcome = cortex::actions::SchedulingBackend::schedule(
            &backend,
            "ship release",
            Some("0 9 * * 1-5"),
            "work",
        )
        .await
        .unwrap();

        assert_eq!(outcome.status, "scheduled");
        let intents = db.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].id, outcome.schedule_id);
        assert_eq!(intents[0].description, "ship release");
    }

    #[tokio::test]
    async fn test_brain_session_schedule_intent_dispatches_and_persists() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.scheduling.enabled = true;
        config.actions.web_search.enabled = false;
        config.actions.messaging.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        session.namespace = "work".to_string();

        let result = session
            .prepare_context("remind me to ship release notes")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("schedule_task ok"));
                assert!(text.contains("namespace=work"));
            }
            _ => panic!("expected action dispatch result"),
        }

        let db = storage::SqlitePool::open(&config.sqlite_path()).unwrap();
        let intents = db.list_scheduled_intents(Some("work")).unwrap();
        assert_eq!(intents.len(), 1);
    }

    #[tokio::test]
    async fn test_brain_session_web_search_custom_without_endpoint_returns_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = brain_core::config::WebSearchProvider::Custom;
        config.actions.web_search.endpoint.clear();
        config.actions.messaging.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("search for rust async")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }

    #[tokio::test]
    async fn test_brain_session_tavily_without_api_key_returns_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = brain_core::config::WebSearchProvider::Tavily;
        config.actions.web_search.api_key.clear();
        config.actions.messaging.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("search for rust async")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }

    #[tokio::test]
    async fn test_brain_session_send_message_enabled_without_channel_mapping_explicit_error() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = brain_core::BrainConfig::default();
        config.brain.data_dir = temp.path().to_string_lossy().to_string();
        config.actions.messaging.enabled = true;
        config.actions.messaging.channels.clear();
        config.actions.web_search.enabled = false;
        config.actions.scheduling.enabled = false;

        let mut session = BrainSession::new(&config).await.unwrap();
        let result = session
            .prepare_context("send via ops to alice saying deploy now")
            .await
            .unwrap();

        match result {
            PrepareResult::ActionResult(text) => {
                assert!(text.contains("backend not configured"));
            }
            _ => panic!("expected action dispatch result"),
        }
    }

    #[test]
    fn test_circuit_breaker_closed_by_default() {
        let cb = CircuitBreaker::new("test", 3, 60);
        assert!(!cb.is_open());
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new("test", 3, 60);
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "should still be closed below threshold");
        cb.record_failure();
        assert!(cb.is_open(), "should be open at threshold");
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let cb = CircuitBreaker::new("test", 3, 60);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert!(!cb.is_open());
        // Failures start from zero again
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open(), "should be closed — counter was reset");
    }

    #[test]
    fn test_circuit_breaker_half_open_after_cooldown() {
        let cb = CircuitBreaker::new("test", 2, 0); // 0-second cooldown = immediate half-open
        cb.record_failure();
        cb.record_failure();
        // With 0s cooldown, the circuit should allow a probe immediately
        assert!(!cb.is_open(), "should be half-open after zero cooldown");
    }

    #[test]
    fn test_render_message_template_default() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "deploy done",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value = serde_json::from_str(&rendered)
            .expect("default template should produce valid JSON");
        assert_eq!(parsed["channel"], "alerts");
        assert_eq!(parsed["recipient"], "alice");
        assert_eq!(parsed["content"], "deploy done");
        assert_eq!(parsed["namespace"], "work");
        assert_eq!(parsed["timestamp"], "2026-03-08T12:00:00Z");
    }

    #[test]
    fn test_render_message_template_custom_slack() {
        let template = r#"{"text": "[{{channel}}] {{content}}"}"#;
        let rendered = render_message_template(
            template, "ops", "bob", "server is down", "personal", "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value = serde_json::from_str(&rendered)
            .expect("custom template should produce valid JSON");
        assert_eq!(parsed["text"], "[ops] server is down");
    }

    #[test]
    fn test_render_message_template_escapes_quotes() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            r#"He said "hello""#,
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value = serde_json::from_str(&rendered)
            .expect("escaped content should produce valid JSON");
        assert_eq!(parsed["content"], r#"He said "hello""#);
    }

    #[test]
    fn test_render_message_template_escapes_newlines() {
        let rendered = render_message_template(
            DEFAULT_MESSAGE_BODY,
            "alerts",
            "alice",
            "line1\nline2",
            "work",
            "2026-03-08T12:00:00Z",
        );
        let parsed: serde_json::Value = serde_json::from_str(&rendered)
            .expect("newline content should produce valid JSON");
        assert_eq!(parsed["content"], "line1\nline2");
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape(r#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(json_escape("a\nb"), r#"a\nb"#);
        assert_eq!(json_escape("back\\slash"), r#"back\\slash"#);
    }
}
