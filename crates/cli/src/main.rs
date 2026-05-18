mod bootstrap;
mod bridge;
mod chat;
mod daemon;
mod deps;
mod doctor;
mod encryption;
mod errors;
mod export;
mod serve;
mod service;
mod status;
mod tail;
mod vault;

use crate::doctor::check_ollama_models;

use clap::{Parser, Subcommand};

/// Brain OS — your AI's long-term memory
#[derive(Parser)]
#[command(name = "brain", version, about, long_about = None)]
struct Cli {
    /// Show full error details (technical error chain)
    #[arg(long, short, global = true)]
    verbose: bool,

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
        #[cfg(feature = "encryption")]
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

    /// Diagnose the local environment — Ollama, models, ports, data dir.
    ///
    /// Run this when `brain start` or `brain chat` is misbehaving. Prints
    /// pass/fail per check and exits non-zero on failures so it's safe
    /// to use in scripts.
    Doctor,

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
        /// Activate the Terminal Bridge gRPC synapse (PTY sessions for agents)
        #[arg(long)]
        terminal: bool,
        /// Host to bind all synapses to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Allow startup with no API keys configured. Without this flag
        /// the daemon refuses to boot when `access.api_keys` is empty,
        /// because empty keys silently fall back to "open mode" and
        /// every `/v1/*` endpoint becomes anonymous. Set this only on
        /// trusted local boxes — never on a host reachable over the
        /// network.
        #[arg(long)]
        no_auth: bool,
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
        action: service::ServiceAction,
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
        action: deps::DepsAction,
    },

    /// Bridge to an external gateway for bidirectional messaging.
    ///
    /// Connects to an external WebSocket gateway and relays messages
    /// to/from Brain's WebSocket synapse. This enables proactive
    /// notifications to be delivered to external transports.
    ///
    /// The bridge also receives proactive notifications from Brain and
    /// pushes them to the connected gateway in real-time.
    ///
    /// Examples:
    ///   brain bridge ws://localhost:8080/bot                  # connect to gateway
    ///   brain bridge wss://gateway.example.com/ws             # over TLS
    ///   brain bridge ws://localhost:8080 --api-key YOUR_KEY   # with auth
    Bridge {
        /// WebSocket URL of the external gateway to connect to
        url: String,

        /// Brain API key for authentication (defaults to config key)
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Tail the observability bus — print BrainEvents from the running daemon.
    ///
    /// Subscribes to `GET /v1/events` (SSE) and emits each `brain_event`
    /// payload as a JSON line on stdout. Useful in headless / SSH sessions
    /// or when the UI is down.
    ///
    /// Examples:
    ///   brain tail
    ///   brain tail --kind signal_received
    ///   brain tail --tool-id mcp:fs:read --since 2026-05-14T00:00:00Z
    Tail {
        /// BrainEvent variant discriminant (e.g. signal_received, tool_call_started).
        #[arg(long)]
        kind: Option<String>,
        /// Filter to a specific tool_id (matches tool-bound events only).
        #[arg(long = "tool-id")]
        tool_id: Option<String>,
        /// Principal filter — forward-compatible; bus events do not yet carry one.
        #[arg(long)]
        principal: Option<String>,
        /// RFC3339 timestamp; only events with ts >= since are forwarded.
        #[arg(long)]
        since: Option<String>,
    },

    /// Manage the credential vault — store, retrieve, list, delete secrets.
    ///
    /// Raw values are never passed on argv or logged. `set` reads from stdin;
    /// `get` hides the value unless `--reveal` is passed.
    ///
    /// Examples:
    ///   brain vault init                        # picks backend; sets verifier if file
    ///   echo -n "$TOKEN" | brain vault set github token --shape env:GITHUB_TOKEN
    ///   brain vault get github token            # metadata only
    ///   brain vault get github token --reveal   # prints the value
    ///   brain vault list --tool github
    ///   brain vault status
    Vault {
        #[command(subcommand)]
        action: vault::VaultAction,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli).await {
        // Already parsed, so verbose is available
        let verbose = std::env::args().any(|a| a == "--verbose" || a == "-v");
        eprintln!("{}", errors::format_error(&err, verbose));
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    // Tracing routing:
    //   - `brain mcp` stdout IS the JSON-RPC channel → tracing must go to
    //     stderr with ANSI off so it never corrupts the stream.
    //   - All other commands also route tracing to stderr so human-readable
    //     stdout (`init`, `status`, `chat`, …) stays clean.
    //
    // Default filter:
    //   - RUST_LOG wins if set
    //   - --verbose / -v → brain=info
    //   - `serve` / `mcp` (long-running services) → brain=info
    //   - everything else → warn (no INFO leaking into stdout-adjacent UX)
    let default_filter =
        if cli.verbose || matches!(cli.command, Commands::Serve { .. } | Commands::Mcp) {
            "brain=info"
        } else {
            "warn"
        };
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| default_filter.into());

    let builder = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr);
    if matches!(cli.command, Commands::Mcp) {
        builder.with_ansi(false).init();
    } else {
        builder.init();
    }

    // Config parse errors are fatal: silently falling back to defaults would
    // boot the daemon with no API keys, no principals, and no standing
    // approvals — i.e. "open mode" without the operator's knowledge.
    // A missing config file is NOT an error here (loader returns Ok with
    // embedded defaults); only malformed YAML / invalid enum values reach this.
    let config = brain_core::BrainConfig::load()
        .map_err(|e| anyhow::anyhow!("failed to load config: {e}"))?;

    config.ensure_data_dirs()?;

    match cli.command {
        // ── init ──────────────────────────────────────────────────────────────
        Commands::Init {
            force,
            #[cfg(feature = "encryption")]
            encrypt,
        } => {
            let data_dir = config.data_dir();
            println!("Forming neural pathways...");
            println!("  Cortex (data dir):  {}", data_dir.display());

            let generated_key = match brain_core::BrainConfig::write_default_config(force)? {
                Some((path, key)) => {
                    println!("  Genome (config):    {} (written)", path.display());
                    Some(key)
                }
                None => {
                    println!(
                        "  Genome (config):    {} (exists, --force to overwrite)",
                        brain_core::BrainConfig::user_config_path().display()
                    );
                    None
                }
            };

            let subdirs = ["db", "ruvector", "models", "logs", "exports"];
            for sub in &subdirs {
                println!("  Region:             {}", data_dir.join(sub).display());
            }

            // Probe Ollama and only warn about the embedding model when it's
            // actually missing. Avoids the previous "(pull with `ollama
            // pull`)" hint firing even when the model was already installed.
            check_ollama_models(&config).await;

            #[cfg(feature = "encryption")]
            if encrypt {
                let salt = storage::Encryptor::generate_salt();
                encryption::write_salt(&config, &salt)?;

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
                    encryption::salt_path(&config).display()
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
                chat::chat_non_interactive(&config, &msg).await?;
            } else {
                chat::chat_interactive(&config).await?;
            }
        }

        // ── status ────────────────────────────────────────────────────────────
        Commands::Status => {
            status::show_status(&config).await?;
        }

        // ── doctor ────────────────────────────────────────────────────────────
        Commands::Doctor => {
            doctor::cmd_doctor(&config).await?;
        }

        // ── start (daemon) ────────────────────────────────────────────────────
        Commands::Start => {
            match config.validate() {
                Err(hard_err) => anyhow::bail!("Configuration error: {}", hard_err),
                Ok(warnings) => {
                    for w in &warnings {
                        eprintln!("WARNING: {w}");
                    }
                }
            }

            // Check for running daemon via HTTP health probe — the only reliable
            // indicator. PID files can drift on macOS with process_group(0).
            if let Some(url) = bootstrap::detect_running_daemon(&config).await {
                println!("Brain is already awake ({url}).");
                println!(
                    "  Logs → {}",
                    config.data_dir().join("logs/brain.log").display()
                );
                println!("Run `brain stop` to put it to sleep first.");
                return Ok(());
            }

            // If a login service is installed, start it instead of spawning directly.
            // The service manager (launchd/systemd) will handle the process lifecycle.
            #[cfg(target_os = "macos")]
            {
                use std::path::Path;
                let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
                if let Some(home) = home {
                    let plist = home
                        .join("Library")
                        .join("LaunchAgents")
                        .join("com.brain.plist");
                    if Path::exists(&plist) {
                        let plist_str = plist.to_str().unwrap_or("");

                        // Check if the service is loaded (managed by launchd)
                        let list_out = std::process::Command::new("launchctl")
                            .arg("list")
                            .output()
                            .ok();
                        let service_loaded = list_out
                            .map(|o| String::from_utf8_lossy(&o.stdout).contains("com.brain"))
                            .unwrap_or(false);

                        if service_loaded {
                            // Service is loaded — check if daemon is actually responding
                            if bootstrap::detect_running_daemon(&config).await.is_some() {
                                println!("Brain is already awake (launchd service).");
                                println!(
                                    "  Logs → {}",
                                    config.data_dir().join("logs/brain.log").display()
                                );
                                println!("Run `brain stop` to put it to sleep first.");
                                return Ok(());
                            }
                        }

                        // Service plist exists but daemon not running — load and start it.
                        // Uses `load -w` so it works even if the service was unloaded by `brain stop`.
                        let _ = std::process::Command::new("launchctl")
                            .args(["load", "-w", plist_str])
                            .output();

                        // Wait for service to come alive
                        for _ in 0..10 {
                            std::thread::sleep(std::time::Duration::from_millis(500));
                            if bootstrap::detect_running_daemon(&config).await.is_some() {
                                println!("Brain started via launchd service.");
                                println!(
                                    "  Logs → {}",
                                    config.data_dir().join("logs/brain.log").display()
                                );
                                return Ok(());
                            }
                        }

                        // Service failed to start — fall through to direct spawn
                        tracing::warn!(
                            "launchd service failed to start — falling back to direct spawn"
                        );
                    }
                }
            }

            // No service installed — spawn daemon directly.
            // First, kill any stale process holding the ports.
            if let Some(pid) = daemon::read_pid(&config) {
                if daemon::is_process_running(pid) {
                    tracing::warn!(pid, "Stale daemon PID found — killing");
                    let _ = daemon::stop_process(pid);
                    // Wait for the daemon to release ports (max 5s).
                    for _ in 0..10 {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        if bootstrap::detect_running_daemon(&config).await.is_none() {
                            break;
                        }
                    }
                }
                daemon::remove_pid(&config);
            }

            #[cfg(feature = "encryption")]
            let passphrase = if config.encryption.enabled {
                if let Ok(p) = std::env::var("BRAIN_PASSPHRASE") {
                    Some(p)
                } else {
                    let salt = encryption::load_salt(&config).ok_or_else(|| {
                        anyhow::anyhow!(
                            "Encryption is enabled but no salt file found.\n\
                             Run `brain init --encrypt` to generate one."
                        )
                    })?;
                    let p = rpassword::prompt_password("Brain passphrase: ")
                        .map_err(|e| anyhow::anyhow!("Failed to read passphrase: {e}"))?;
                    storage::Encryptor::from_passphrase(&p, &salt)
                        .map_err(|e| anyhow::anyhow!("Key derivation failed: {e}"))?;
                    Some(p)
                }
            } else {
                None
            };
            #[cfg(not(feature = "encryption"))]
            let passphrase: Option<String> = None;

            let log_path = config.data_dir().join("logs/brain.log");
            let pid = daemon::spawn_daemon(&log_path, passphrase.as_deref())?;
            daemon::write_pid(&config, pid)?;

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
        Commands::Stop => {
            // Check if daemon is running BEFORE stopping the service, so we can
            // report accurately whether we stopped something or it was already down.
            let was_running = daemon::is_daemon_running(&config).await;

            // Always stop the service manager first (if installed) to prevent respawn.
            // This unloads/disables the service so it doesn't restart the daemon.
            if daemon::is_service_installed() {
                daemon::stop_service();
                // Give the service manager time to propagate the stop
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }

            match daemon::read_pid(&config) {
                Some(pid) if daemon::is_process_running(pid) => {
                    daemon::stop_process(pid)?;
                    // Wait for process to exit (max 5s).
                    for _ in 0..10 {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        if !daemon::is_process_running(pid) {
                            break;
                        }
                    }
                    daemon::remove_pid(&config);
                    println!("Brain is asleep (PID {}).", pid);
                }
                Some(_) => {
                    daemon::remove_pid(&config);
                    println!("Brain was already asleep (stale PID file cleaned up).");
                }
                None => {
                    // No PID file — daemon may have been started by service manager.
                    // Check HTTP to determine if it was running.
                    if was_running {
                        // It was running — wait for the service stop to take effect.
                        for _ in 0..10 {
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                            if !daemon::is_daemon_running(&config).await {
                                break;
                            }
                        }
                        if daemon::is_daemon_running(&config).await {
                            println!("Brain stop requested but daemon did not exit cleanly.");
                        } else {
                            println!("Brain is asleep.");
                        }
                    } else {
                        println!("Brain is already asleep.");
                    }
                }
            }
        }

        // ── serve (foreground) ────────────────────────────────────────────────
        Commands::Serve {
            http,
            ws,
            grpc,
            mcp,
            terminal,
            host,
            no_auth,
        } => {
            serve::cmd_serve(&config, http, ws, grpc, mcp, terminal, host, no_auth).await?;
        }

        // ── mcp stdio ─────────────────────────────────────────────────────────
        Commands::Mcp => {
            // Always proxy through the daemon's MCP HTTP transport.
            // This ensures a single shared SignalProcessor — no ruvector lock
            // contention, no memory isolation, no passphrase prompts.
            //
            // Retry daemon detection a few times — the daemon might still be
            // booting when an MCP client spawns `brain mcp`.
            // Detect the daemon via the HTTP adapter's health endpoint…
            let _daemon_url = bootstrap::require_daemon(&config).await?;
            // …but proxy to the MCP adapter's own HTTP port (separate from the REST API).
            let mcp_host = &config.adapters.http.host;
            let mcp_port = config.adapters.mcp.port;
            let mcp_url = format!("http://{mcp_host}:{mcp_port}/mcp");
            tracing::info!(url = %mcp_url, "Daemon detected — proxying MCP stdio through HTTP");
            bootstrap::proxy_mcp_stdio(&mcp_url, &config).await?;
        }

        // ── export ────────────────────────────────────────────────────────────
        Commands::Export { output } => {
            export::cmd_export(&config, output.as_deref()).await?;
        }

        // ── import ────────────────────────────────────────────────────────────
        Commands::Import { file, dry_run } => {
            export::cmd_import(&config, &file, dry_run).await?;
        }

        // ── service ───────────────────────────────────────────────────────────
        Commands::Service { action } => match action {
            service::ServiceAction::Install => service::cmd_service_install().await?,
            service::ServiceAction::Uninstall => service::cmd_service_uninstall()?,
        },

        // ── deps ─────────────────────────────────────────────────────────────
        Commands::Deps { action } => {
            deps::cmd_deps(action)?;
        }

        // ── bridge ─────────────────────────────────────────────────────────
        Commands::Bridge { url, api_key } => {
            bridge::cmd_bridge(&config, &url, api_key.as_deref()).await?;
        }

        // ── vault ───────────────────────────────────────────────────────────
        Commands::Vault { action } => {
            vault::cmd_vault(&config, action).await?;
        }

        // ── tail ────────────────────────────────────────────────────────────
        Commands::Tail {
            kind,
            tool_id,
            principal,
            since,
        } => {
            tail::cmd_tail(
                &config,
                tail::TailFilter {
                    kind,
                    tool_id,
                    principal,
                    since,
                },
            )
            .await?;
        }
    }

    Ok(())
}
