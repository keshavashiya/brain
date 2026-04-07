mod bootstrap;
mod bridge;
mod chat;
mod daemon;
mod deps;
mod encryption;
mod export;
mod schedules;
mod serve;
mod service;
mod status;

use clap::{Parser, Subcommand};

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
    /// Connects to an external WebSocket gateway (e.g., Slack, Telegram, Discord bot)
    /// and relays messages to/from Brain's WebSocket synapse. This enables proactive
    /// notifications to be delivered to external platforms.
    ///
    /// The bridge also receives proactive notifications from Brain and pushes them
    /// to the connected gateway in real-time.
    ///
    /// Examples:
    ///   brain bridge ws://localhost:8080/bot          # connect to gateway
    ///   brain bridge wss://slack.bot.com/ws           # connect to Slack (with TLS)
    ///   brain bridge ws://localhost:8080 --api-key YOUR_KEY  # with auth
    Bridge {
        /// WebSocket URL of the external gateway to connect to
        url: String,

        /// Brain API key for authentication (defaults to config key)
        #[arg(long)]
        api_key: Option<String>,
    },

    /// Manage scheduled intents
    Schedules {
        #[command(subcommand)]
        action: schedules::SchedulesAction,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // For `brain mcp` (stdio transport), stdout IS the JSON-RPC channel.
    // Tracing must go to stderr with ANSI disabled so it never corrupts the stream.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "brain=info".into());

    if matches!(cli.command, Commands::Mcp) {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_writer(std::io::stderr)
            .with_ansi(false)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

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

            if let Some(pid) = daemon::read_pid(&config) {
                if daemon::is_process_running(pid) {
                    println!("Brain is already awake (PID {}).", pid);
                    println!(
                        "  Logs → {}",
                        config.data_dir().join("logs/brain.log").display()
                    );
                    println!("Run `brain stop` to put it to sleep first.");
                    return Ok(());
                }
                daemon::remove_pid(&config);
            }

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
        Commands::Stop => match daemon::read_pid(&config) {
            Some(pid) if daemon::is_process_running(pid) => {
                daemon::stop_process(pid)?;
                daemon::remove_pid(&config);
                println!("Brain is asleep (PID {}).", pid);
            }
            Some(_) => {
                daemon::remove_pid(&config);
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
            serve::cmd_serve(&config, http, ws, grpc, mcp, host).await?;
        }

        // ── mcp stdio ─────────────────────────────────────────────────────────
        Commands::Mcp => {
            // Always proxy through the daemon's MCP HTTP transport.
            // This ensures a single shared SignalProcessor — no ruvector lock
            // contention, no memory isolation, no passphrase prompts.
            //
            // Retry daemon detection a few times — the daemon might still be
            // booting when an MCP client spawns `brain mcp`.
            let daemon_url = bootstrap::require_daemon(&config).await?;
            let mcp_url = format!("{}/mcp", daemon_url);
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
            service::ServiceAction::Install => service::cmd_service_install()?,
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

        // ── schedules ───────────────────────────────────────────────────────
        Commands::Schedules { action } => {
            schedules::cmd_schedules(&config, action).await?;
        }
    }

    Ok(())
}
