use std::sync::Arc;

use clap::{Parser, Subcommand};
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::ExecutableCommand;
use rustyline::DefaultEditor;
use std::io::stdout;

/// Brain OS -- Central AI Operating System
#[derive(Parser)]
#[command(name = "brain", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Brain (creates ~/.brain/ and default config)
    Init {
        /// Overwrite existing config file
        #[arg(long)]
        force: bool,
        /// Enable encryption at rest (AES-256-GCM). Generates a salt and
        /// prompts for a passphrase. Requires `encryption.enabled: true`
        /// in config to activate on subsequent runs.
        #[arg(long)]
        encrypt: bool,
    },

    /// Start an interactive chat session
    Chat {
        /// Optional initial message (non-interactive mode)
        message: Option<String>,
    },

    /// Show system status
    Status,

    /// Start all Brain services as a background daemon.
    ///
    /// All adapters (HTTP, WebSocket, gRPC, MCP HTTP) start on their configured
    /// ports and continue running after the terminal closes. Logs go to
    /// ~/.brain/logs/brain.log. Use `brain stop` to shut down.
    Start,

    /// Stop the running Brain daemon.
    Stop,

    /// Start Brain services in the foreground (development / debugging).
    ///
    /// With no flags all four adapters start concurrently.
    /// Use flags to start only specific adapters.
    ///
    /// Examples:
    ///   brain serve                  # all adapters
    ///   brain serve --http           # HTTP only
    ///   brain serve --http --ws      # HTTP + WebSocket
    Serve {
        /// Start the HTTP REST API adapter
        #[arg(long)]
        http: bool,
        /// Start the WebSocket adapter
        #[arg(long)]
        ws: bool,
        /// Start the gRPC adapter
        #[arg(long)]
        grpc: bool,
        /// Start the MCP HTTP adapter
        #[arg(long)]
        mcp: bool,
        /// Host to bind all adapters to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },

    /// Start the MCP stdio server.
    ///
    /// Used when an MCP client spawns Brain as a subprocess and communicates
    /// over stdin/stdout. This runs in the foreground by design.
    Mcp,

    /// Export all memory (facts + episodes) to a JSON backup file.
    ///
    /// Examples:
    ///   brain export                      # print JSON to stdout
    ///   brain export --output backup.json # write to file
    Export {
        /// Output file path (default: stdout)
        #[arg(long, short)]
        output: Option<String>,
    },

    /// Import memory from a JSON backup file produced by `brain export`.
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

    /// Manage Brain as a system service (auto-start on login).
    ///
    /// On macOS:   installs a launchd agent in ~/Library/LaunchAgents/.
    /// On Linux:   installs a systemd user service in ~/.config/systemd/user/.
    /// On Windows: registers a Task Scheduler task (no admin required).
    ///
    /// Examples:
    ///   brain service install    # register + enable auto-start
    ///   brain service uninstall  # remove the service registration
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },
}

#[derive(Subcommand)]
enum ServiceAction {
    /// Register Brain as a login service and start it immediately.
    ///
    /// macOS: launchd plist with KeepAlive. Linux: systemd user unit.
    /// Windows: Task Scheduler task at ONLOGON (no admin required).
    Install,
    /// Remove the Brain login service registration and stop auto-start.
    Uninstall,
}

// ─── Daemon helpers ───────────────────────────────────────────────────────────

fn pid_path(config: &core::BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("brain.pid")
}

fn read_pid(config: &core::BrainConfig) -> Option<u32> {
    std::fs::read_to_string(pid_path(config))
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn write_pid(config: &core::BrainConfig, pid: u32) -> anyhow::Result<()> {
    std::fs::write(pid_path(config), pid.to_string())?;
    Ok(())
}

fn remove_pid(config: &core::BrainConfig) {
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
fn spawn_daemon(log_path: &std::path::Path) -> anyhow::Result<u32> {
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

fn salt_path(config: &core::BrainConfig) -> std::path::PathBuf {
    config.data_dir().join("db/salt")
}

fn load_salt(config: &core::BrainConfig) -> Option<[u8; 16]> {
    let bytes = std::fs::read(salt_path(config)).ok()?;
    if bytes.len() == 16 {
        let mut arr = [0u8; 16];
        arr.copy_from_slice(&bytes);
        Some(arr)
    } else {
        None
    }
}

fn write_salt(config: &core::BrainConfig, salt: &[u8; 16]) -> anyhow::Result<()> {
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
fn resolve_encryptor(config: &core::BrainConfig) -> anyhow::Result<Option<storage::Encryptor>> {
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

    let config = core::BrainConfig::load().unwrap_or_else(|e| {
        tracing::warn!("Failed to load config, using defaults: {e}");
        core::BrainConfig::default()
    });

    config.ensure_data_dirs()?;

    match cli.command {
        // ── init ──────────────────────────────────────────────────────────────
        Commands::Init { force, encrypt } => {
            let data_dir = config.data_dir();
            println!("Initializing Brain...");
            println!("  Data dir:  {}", data_dir.display());

            match core::BrainConfig::write_default_config(force)? {
                Some(path) => println!("  Config:    {} (created)", path.display()),
                None => println!(
                    "  Config:    {} (already exists, use --force to overwrite)",
                    core::BrainConfig::user_config_path().display()
                ),
            }

            let subdirs = ["db", "ruvector", "models", "logs", "exports"];
            for sub in &subdirs {
                println!("  Dir:       {}", data_dir.join(sub).display());
            }

            println!(
                "\n  Embedding:  {} (pull with `ollama pull {}`)",
                config.embedding.model,
                config.embedding.model
            );

            // ── Encryption setup ──────────────────────────────────────────────
            if encrypt {
                let salt = storage::Encryptor::generate_salt();
                write_salt(&config, &salt)?;
                println!("\n  Encryption: salt generated → {}", salt_path(&config).display());
                println!("  Next steps:");
                println!("    1. Set 'encryption.enabled: true' in your config.");
                println!("    2. Set BRAIN_PASSPHRASE env var for the daemon, or");
                println!("       Brain will prompt you for a passphrase on startup.");
            }

            println!(
                "\nBrain initialized. Edit {} to customize.",
                core::BrainConfig::user_config_path().display()
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
                    println!("Brain is already running (PID {}).", pid);
                    println!(
                        "  Logs → {}",
                        config.data_dir().join("logs/brain.log").display()
                    );
                    println!("Run `brain stop` to stop it first.");
                    return Ok(());
                }
                // Stale PID file — clean it up and continue
                remove_pid(&config);
            }

            let log_path = config.data_dir().join("logs/brain.log");
            let pid = spawn_daemon(&log_path)?;
            write_pid(&config, pid)?;

            println!("Brain started (PID {}).", pid);
            println!("  HTTP  → http://127.0.0.1:{}", config.adapters.http.port);
            println!("  WS    → ws://127.0.0.1:{}", config.adapters.ws.port);
            println!("  MCP   → http://127.0.0.1:{}", config.adapters.mcp.port);
            println!("  gRPC  → 127.0.0.1:{}", config.adapters.grpc.port);
            println!("  Logs  → {}", log_path.display());
            println!("\nRun `brain stop` to shut down.");
        }

        // ── stop (daemon) ─────────────────────────────────────────────────────
        Commands::Stop => {
            match read_pid(&config) {
                Some(pid) if is_process_running(pid) => {
                    stop_process(pid)?;
                    remove_pid(&config);
                    println!("Brain stopped (PID {}).", pid);
                }
                Some(_) => {
                    remove_pid(&config);
                    println!("Brain was not running (stale PID file removed).");
                }
                None => {
                    println!("Brain is not running.");
                }
            }
        }

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
            let processor =
                Arc::new(signal::SignalProcessor::new_with_encryptor(config.clone(), encryptor).await?);

            println!("Starting Brain OS...");

            let mut set = tokio::task::JoinSet::new();

            if run_all || http {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.http.port;
                println!("  HTTP  → http://{}:{}", h, port);
                set.spawn(async move { httpadapter::serve(p, &h, port).await });
            }

            if run_all || ws {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.ws.port;
                println!("  WS    → ws://{}:{}", h, port);
                set.spawn(async move { wsadapter::serve(p, &h, port).await });
            }

            if run_all || grpc {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.grpc.port;
                println!("  gRPC  → {}:{}", h, port);
                set.spawn(async move { grpcadapter::serve(p, &h, port).await });
            }

            if run_all || mcp {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.mcp.port;
                println!("  MCP   → http://{}:{}", h, port);
                set.spawn(async move { mcp::serve_http(p, &h, port).await });
            }

            // ── Proactivity / habit engine background task ────────────────────
            // Runs the HabitEngine on a schedule (default: every 60 min).
            // When a recurring pattern is detected at the current time slot
            // and all rate limits are met, a proactive message is logged and
            // written to ~/.brain/logs/proactive.log.
            if config.proactivity.enabled {
                let p = processor.clone();
                let habit_cfg = ganglia::HabitConfig {
                    max_per_day: config.proactivity.max_per_day,
                    min_interval_minutes: config.proactivity.min_interval_minutes,
                    quiet_start: config.proactivity.quiet_hours.start.clone(),
                    quiet_end: config.proactivity.quiet_hours.end.clone(),
                    ..Default::default()
                };
                let log_path = config.data_dir().join("logs/proactive.log");
                set.spawn(async move {
                    let engine = ganglia::HabitEngine::new(
                        p.episodic().pool().clone(),
                        habit_cfg.clone(),
                    );
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
                                // Append to proactive log file
                                let line = format!(
                                    "[{}] {}\n",
                                    msg.created_at.to_rfc3339(),
                                    msg.content
                                );
                                let _ = std::fs::OpenOptions::new()
                                    .create(true)
                                    .append(true)
                                    .open(&log_path)
                                    .and_then(|mut f| {
                                        use std::io::Write;
                                        f.write_all(line.as_bytes())
                                    });
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

            // ── Memory consolidation background task ──────────────────────────
            // Runs the forgetting-curve pruner on a schedule (default: every 24h).
            // Aborted cleanly alongside adapters on shutdown.
            if config.memory.consolidation.enabled {
                let p = processor.clone();
                let interval_hours = config.memory.consolidation.interval_hours;
                let prune_threshold = config.memory.consolidation.forgetting_threshold;
                set.spawn(async move {
                    let consolidator = hippocampus::Consolidator::new(
                        hippocampus::ConsolidationConfig {
                            prune_threshold,
                            ..Default::default()
                        },
                    );
                    // Skip the first tick so consolidation doesn't run at startup.
                    let mut ticker = tokio::time::interval(
                        tokio::time::Duration::from_secs(interval_hours as u64 * 3600),
                    );
                    ticker.tick().await;
                    loop {
                        ticker.tick().await;
                        match consolidator.consolidate(p.episodic()) {
                            Ok(r) => tracing::info!(
                                pruned = r.episodes_pruned,
                                promoted = r.episodes_promoted,
                                remaining = r.episodes_remaining,
                                "Memory consolidation complete"
                            ),
                            Err(e) => tracing::warn!("Memory consolidation error: {e}"),
                        }
                    }
                });
                tracing::info!(
                    interval_hours,
                    "Memory consolidation scheduled"
                );
            }


            println!("\nPress Ctrl+C to stop.\n");

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
            tracing::info!("Brain OS stopped cleanly");
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
    }

    Ok(())
}

// ─── Status ───────────────────────────────────────────────────────────────────

async fn show_status(config: &core::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Status");
    println!("  Version:    {}", env!("CARGO_PKG_VERSION"));
    println!("  Data dir:   {}", config.data_dir().display());

    // Daemon state
    match read_pid(config) {
        Some(pid) if is_process_running(pid) => {
            println!("  Daemon:     running (PID {})", pid);
        }
        Some(_) => {
            println!("  Daemon:     stopped (stale PID file)");
        }
        None => {
            println!("  Daemon:     stopped  (run `brain start` to start)");
        }
    }

    println!(
        "  LLM:        {} ({})",
        config.llm.model, config.llm.provider
    );
    println!(
        "  Embedding:  {} ({}d)",
        config.embedding.model, config.embedding.dimensions
    );
    println!(
        "  Encryption: {}",
        if config.encryption.enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("  SQLite:     {}", config.sqlite_path().display());
    println!("  RuVector:   {}", config.ruvector_path().display());
    println!(
        "  Config:     {}",
        core::BrainConfig::user_config_path().display()
    );

    println!("\n  Adapters:");
    let h = &config.adapters.http;
    println!(
        "    HTTP      : port {} ({})",
        h.port,
        if h.enabled { "enabled" } else { "disabled" }
    );
    let w = &config.adapters.ws;
    println!(
        "    WebSocket : port {} ({})",
        w.port,
        if w.enabled { "enabled" } else { "disabled" }
    );
    let m = &config.adapters.mcp;
    println!(
        "    MCP       : port {} ({})",
        m.port,
        if m.enabled { "enabled" } else { "disabled" }
    );
    let g = &config.adapters.grpc;
    println!(
        "    gRPC      : port {} ({})",
        g.port,
        if g.enabled { "enabled" } else { "disabled" }
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
        "  LLM Health: {}",
        if llm_healthy { "connected" } else { "disconnected" }
    );

    // Database stats
    match storage::SqlitePool::open(&config.sqlite_path()) {
        Ok(pool) => match pool.table_stats() {
            Ok(stats) => {
                println!("\n  Database Tables:");
                for (table, count) in stats {
                    println!("    {}: {} rows", table, count);
                }
            }
            Err(e) => println!("\n  Database: error reading stats — {}", e),
        },
        Err(e) => println!("\n  Database: error opening — {}", e),
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

        println!("Brain service installed (launchd).");
        println!("  Plist:  {}", plist_path.display());
        println!("  Log:    {}/brain.log", log_dir.display());
        println!("  Brain will now start automatically on every login.");
        println!("  To stop auto-start: brain service uninstall");
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
Description=Brain OS — Personal AI Memory System
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
            println!("Brain service file written to {}.", service_path.display());
            println!("  Run manually:");
            println!("    systemctl --user daemon-reload");
            println!("    systemctl --user enable --now brain.service");
        } else {
            println!("Brain service installed (systemd user).");
            println!("  Unit:   {}", service_path.display());
            println!("  Log:    {}/brain.log", log_dir.display());
            println!("  Brain will now start automatically on every login.");
            println!("  To stop auto-start: brain service uninstall");
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
                "/Create",
                "/TN", task_name,
                "/TR", &cmd,
                "/SC", "ONLOGON",
                "/RL", "HIGHEST",   // run with the highest privileges the user has
                "/F",               // overwrite if already exists
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

        println!("Brain service installed (Windows Task Scheduler).");
        println!("  Task:   {task_name}");
        println!("  Brain will now start automatically on every login.");
        println!("  To stop auto-start: brain service uninstall");
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
            println!("Brain service is not installed (no plist found).");
            return Ok(());
        }

        let _ = std::process::Command::new("launchctl")
            .args(["unload", "-w", plist_path.to_str().unwrap()])
            .output();

        std::fs::remove_file(&plist_path)?;
        println!("Brain service uninstalled.");
        println!("  Removed: {}", plist_path.display());
        println!("  Brain will no longer start automatically on login.");
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
            println!("Brain service is not installed (no unit file found).");
            return Ok(());
        }

        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", "brain.service"])
            .output();

        std::fs::remove_file(&service_path)?;

        let _ = std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .output();

        println!("Brain service uninstalled.");
        println!("  Removed: {}", service_path.display());
        println!("  Brain will no longer start automatically on login.");
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
                println!("Brain service is not installed (no task found).");
                return Ok(());
            }
            anyhow::bail!("schtasks /Delete failed: {stderr}");
        }

        println!("Brain service uninstalled.");
        println!("  Task '{task_name}' removed from Windows Task Scheduler.");
        println!("  Brain will no longer start automatically on login.");
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
    role: String,
    content: String,
    timestamp: String,
    importance: f64,
    reinforcement_count: i32,
}

fn cmd_export(config: &core::BrainConfig, output: Option<&str>) -> anyhow::Result<()> {
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
                    e.role, e.content, e.timestamp,
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
                    role: row.get(3)?,
                    content: row.get(4)?,
                    timestamp: row.get(5)?,
                    importance: row.get(6)?,
                    reinforcement_count: row.get(7)?,
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

async fn cmd_import(config: &core::BrainConfig, file: &str, dry_run: bool) -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Cannot read {file}: {e}"))?;
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
    let mut sessions: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
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
                    f.id, f.namespace, f.category, f.subject,
                    f.predicate, f.object, f.confidence, f.source_episode_id
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
                    (id, session_id, role, content, timestamp,
                     importance, reinforcement_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO NOTHING",
                rusqlite::params![
                    e.id, e.session_id, e.role, e.content,
                    e.timestamp, e.importance, e.reinforcement_count
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

// ─── Chat ─────────────────────────────────────────────────────────────────────

async fn chat_non_interactive(config: &core::BrainConfig, message: &str) -> anyhow::Result<()> {
    let mut brain = BrainSession::new(config).await?;
    println!("{}", brain.process_message(message).await?);
    Ok(())
}

async fn chat_interactive(config: &core::BrainConfig) -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════╗");
    println!(
        "║  Brain v{}                          ║",
        env!("CARGO_PKG_VERSION")
    );
    println!("║  A personal AI that remembers you     ║");
    println!("╚═══════════════════════════════════════╝");
    println!();
    println!("  Model: {}", config.llm.model);
    println!("  Data:  {}", config.data_dir().display());
    println!();
    println!("Commands: /status  /clear  /quit");
    println!();

    let mut brain = BrainSession::new(config).await?;
    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);

    loop {
        match rl.readline("You: ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                let _ = rl.add_history_entry(input);

                match input {
                    "/quit" | "/exit" => {
                        println!("Goodbye!");
                        break;
                    }
                    "/status" => {
                        show_status(config).await?;
                        continue;
                    }
                    "/clear" => {
                        brain.clear_history();
                        println!("Conversation history cleared.");
                        continue;
                    }
                    _ => {}
                }

                match brain.process_message(input).await {
                    Ok(response) => {
                        let mut out = stdout();
                        out.execute(SetForegroundColor(Color::Green))?;
                        out.execute(Print("Brain: "))?;
                        out.execute(ResetColor)?;
                        println!("{}", response);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye!");
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

#[allow(dead_code)]
struct BrainSession {
    _config: core::BrainConfig,
    _db: storage::SqlitePool,
    episodic: hippocampus::EpisodicStore,
    semantic: Option<hippocampus::SemanticStore>,
    embedder: Option<hippocampus::Embedder>,
    embedding_dim: usize,
    recall_engine: hippocampus::RecallEngine,
    llm: Box<dyn cortex::llm::LlmProvider>,
    context_assembler: cortex::context::ContextAssembler,
    conversation_history: Vec<cortex::llm::Message>,
    session_id: String,
}

impl BrainSession {
    async fn new(config: &core::BrainConfig) -> anyhow::Result<Self> {
        let db = storage::SqlitePool::open(&config.sqlite_path())?;
        let episodic = hippocampus::EpisodicStore::new(db.clone());

        let semantic = if let Ok(ruv) =
            storage::RuVectorStore::open(&config.ruvector_path(), config.embedding.dimensions as usize).await
        {
            ruv.ensure_tables().await.ok();
            Some(hippocampus::SemanticStore::new(db.clone(), ruv))
        } else {
            None
        };

        // Create embedder (same logic as SignalProcessor)
        let embedding_dim = config.embedding.dimensions as usize;
        let embedder = match config.llm.provider.as_str() {
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
        };

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
            _db: db,
            episodic,
            semantic,
            embedder,
            embedding_dim,
            recall_engine,
            llm,
            context_assembler,
            conversation_history: Vec::new(),
            session_id,
        })
    }

    fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    /// Generate a vector embedding for text, falling back to a zero vector.
    async fn embed_text(&mut self, text: &str) -> Vec<f32> {
        if let Some(ref mut embedder) = self.embedder {
            match embedder.embed(text).await {
                Ok(vec) => vec,
                Err(e) => {
                    tracing::warn!("Embedding failed in CLI chat, using zero vector: {e}");
                    vec![0.0_f32; self.embedding_dim]
                }
            }
        } else {
            vec![0.0_f32; self.embedding_dim]
        }
    }

    async fn process_message(&mut self, message: &str) -> anyhow::Result<String> {
        use cortex::llm::{Message, Role};

        let importance = hippocampus::ImportanceScorer::score(message, true);
        self.episodic
            .store_episode(&self.session_id, "user", message, importance)?;

        let thalamus = thalamus::SignalRouter::new();
        let classification = thalamus.route(&thalamus::NormalizedMessage {
            content: message.to_string(),
            channel: "cli".to_string(),
            sender: "user".to_string(),
            timestamp: chrono::Utc::now(),
            message_id: None,
            metadata: std::collections::HashMap::new(),
        });

        if let Some(action) = thalamus.intent_to_action(&classification.intent) {
            let result = cortex::actions::ActionDispatcher::with_defaults()
                .dispatch(&action)
                .await;
            return if result.success {
                Ok(result.output)
            } else {
                Ok(format!("Error: {}", result.error.unwrap_or_default()))
            };
        }

        // Hybrid recall (BM25 + ANN via RecallEngine) — same path as SignalProcessor
        let query_vector = self.embed_text(message).await;
        let memories = if let Some(semantic) = &self.semantic {
            match self
                .recall_engine
                .recall(message, query_vector, &self.episodic, semantic, 10, None)
                .await
            {
                Ok(mems) => mems,
                Err(e) => {
                    tracing::warn!("Recall engine failed in CLI chat: {e}");
                    Vec::new()
                }
            }
        } else {
            // Semantic store unavailable — fall back to BM25 only
            self.episodic
                .search_bm25(message, 10)
                .unwrap_or_default()
                .into_iter()
                .map(|r| hippocampus::search::Memory {
                    id: r.episode_id,
                    content: r.content,
                    source: hippocampus::search::MemorySource::Episodic,
                    score: r.rank,
                    importance: 0.5,
                    timestamp: r.timestamp,
                })
                .collect()
        };

        let messages = self
            .context_assembler
            .assemble(message, &memories, &self.conversation_history);

        let response = self.llm.generate(&messages).await?;

        self.episodic
            .store_episode(&self.session_id, "assistant", &response.content, 0.5)?;

        self.conversation_history.push(Message {
            role: Role::User,
            content: message.to_string(),
        });
        self.conversation_history.push(Message {
            role: Role::Assistant,
            content: response.content.clone(),
        });

        // Keep last 20 exchanges
        if self.conversation_history.len() > 40 {
            self.conversation_history = self.conversation_history.split_off(20);
        }

        Ok(response.content)
    }
}
