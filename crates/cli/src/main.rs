mod baseline;
mod bootstrap;
mod bridge;
mod capabilities;
mod chat;
mod command_catalog;
mod config;
mod daemon;
mod deps;
mod doctor;
mod encryption;
mod errors;
mod export;
mod init;
mod logging;
mod logs;
mod serve;
mod service;
mod status;
mod tail;
mod update;
mod vault;

use clap::{Parser, Subcommand};

/// Brain OS — your AI's long-term memory
#[derive(Parser)]
#[command(name = "brain", version, about, long_about = None)]
pub(crate) struct Cli {
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
    Doctor {
        /// Also run store-level probes (SQLite schema, vector store, embedder
        /// round-trip, audit chain, row counts) — opens the data stores, so
        /// the vector-store probe is skipped when a daemon holds the lock.
        #[arg(long)]
        deep: bool,
    },

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
    ///   brain tail --correlation <signal-uuid>   # one turn, end-to-end
    Tail {
        /// BrainEvent variant discriminant (e.g. signal_received, tool_call_started).
        #[arg(long)]
        kind: Option<String>,
        /// Filter to a specific tool_id (matches tool-bound events only).
        #[arg(long = "tool-id")]
        tool_id: Option<String>,
        /// Correlation id — reconstruct a single signal flow (every event in
        /// one turn shares the originating signal's id).
        #[arg(long)]
        correlation: Option<String>,
        /// Principal filter — forward-compatible; bus events do not yet carry one.
        #[arg(long)]
        principal: Option<String>,
        /// RFC3339 timestamp; only events with ts >= since are forwarded.
        #[arg(long)]
        since: Option<String>,
    },

    /// Read the brain's logs — scan for recurring error/warning patterns.
    ///
    /// A deterministic pattern pass (level counts + recurring signatures) always
    /// runs offline; an optional LLM summary narrates the digest on top.
    ///
    /// Examples:
    ///   brain logs analyze                     # own daemon log, last 1h
    ///   brain logs analyze --since 30m         # narrower window
    ///   brain logs analyze --source system     # OS log (sandboxed log show/journalctl)
    ///   brain logs analyze --no-llm            # deterministic digest only
    Logs {
        #[command(subcommand)]
        action: logs::LogsAction,
    },

    /// Capture system baselines and detect drift from them.
    ///
    /// A baseline is an offline snapshot of stable system facts (version,
    /// platform, LLM/embedding wiring, adapters, action toggles, the native
    /// capability inventory, security surface), stored as a versioned JSON
    /// file. `diff` reports what changed since a baseline.
    ///
    /// Examples:
    ///   brain baseline capture --label "post-install"  # snapshot now
    ///   brain baseline list                            # stored versions
    ///   brain baseline diff                            # latest vs live state
    ///   brain baseline diff --from 1 --to 3            # two stored snapshots
    Baseline {
        #[command(subcommand)]
        action: baseline::BaselineAction,
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

    /// Inspect, validate, and discover the runtime config.
    ///
    /// Examples:
    ///   brain config validate                 # validate the resolved config
    ///   brain config validate --file new.yaml # dry-run a file before installing
    ///   brain config show                     # print the resolved effective config
    ///   brain config show --defaults          # print the embedded default (schema-by-example)
    ///   brain config path                     # print ~/.brain/config.yaml (BRAIN_CONFIG-aware)
    Config {
        #[command(subcommand)]
        action: config::ConfigAction,
    },

    /// List the live capability manifest from the running daemon.
    ///
    /// Shows every tool the kernel can dispatch to — native backends,
    /// terminal, and mounted MCP servers — grouped by source and tagged
    /// with its safety tier, plus the registered delegate agents. This is
    /// the same manifest the reasoner sees.
    Capabilities,

    /// Regenerate — check for a newer release and self-update via the installer.
    ///
    /// Queries the GitHub Releases API for the latest version and, if it is
    /// newer than the running binary, runs the official one-line installer
    /// (which handles arch detection + checksum verification) pinned to that
    /// tag. This is the only command that touches the network for updates;
    /// nothing checks passively.
    ///
    /// Examples:
    ///   brain update          # check, then install on confirmation
    ///   brain update --check  # report availability only, install nothing
    ///   brain update --yes    # install without the confirmation prompt
    ///   brain update --tag v0.5.0  # install a specific release (reinstall/pin)
    Update {
        /// Report whether a newer release exists, then exit without installing.
        #[arg(long)]
        check: bool,
        /// Skip the confirmation prompt before installing.
        #[arg(long, short)]
        yes: bool,
        /// Install this exact release tag instead of the latest.
        #[arg(long)]
        tag: Option<String>,
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
    // Config parse errors are fatal: silently falling back to defaults would
    // boot the daemon with no API keys, no principals, and no standing
    // approvals — i.e. "open mode" without the operator's knowledge.
    // A missing config file is NOT an error here (loader returns Ok with
    // embedded defaults); only malformed YAML / invalid enum values reach this.
    //
    // Loaded before tracing init so the logging policy (`[logging]`) can drive
    // the subscriber. Config-load failures use `eprintln!` (no tracing yet).
    //
    // Forward-migrate an older config file *before* loading it, so renamed or
    // moved keys survive into the typed struct instead of being silently
    // dropped by figment. A migration hiccup is non-fatal: the original file
    // is snapshotted first, and any real corruption surfaces at load below.
    match brain::BrainConfig::migrate_user_config_if_needed() {
        Ok(Some(outcome)) => {
            eprintln!(
                "config: migrated v{} → v{} ({} change(s))",
                outcome.from_version,
                outcome.to_version,
                outcome.changes.len()
            );
            for change in &outcome.changes {
                eprintln!("  - {change}");
            }
            if let Some(backup) = &outcome.backup_path {
                eprintln!("  backup: {}", backup.display());
            }
            for key in &outcome.unknown_keys {
                eprintln!(
                    "config: warning: unrecognized key '{key}' \
                     (typo, or removed in this version — left as-is)"
                );
            }
        }
        Ok(None) => {}
        Err(e) => eprintln!("config: migration skipped ({e})"),
    }

    let config =
        brain::BrainConfig::load().map_err(|e| anyhow::anyhow!("failed to load config: {e}"))?;

    config.ensure_data_dirs()?;

    // Install tracing per the logging policy. The returned guard flushes the
    // non-blocking file appender on drop — hold it for the whole process.
    let _log_guard = logging::init(&cli, &config);

    match cli.command {
        Commands::Init {
            force,
            #[cfg(feature = "encryption")]
            encrypt,
        } => {
            init::cmd_init(
                &config,
                force,
                #[cfg(feature = "encryption")]
                encrypt,
            )
            .await?
        }
        Commands::Chat { message } => match message {
            Some(msg) => chat::chat_non_interactive(&config, &msg).await?,
            None => chat::chat_interactive(&config).await?,
        },
        Commands::Status => status::show_status(&config).await?,
        Commands::Doctor { deep } => doctor::cmd_doctor(&config, deep).await?,
        Commands::Start => daemon::cmd_start(&config).await?,
        Commands::Stop => daemon::cmd_stop(&config).await?,
        Commands::Serve {
            http,
            ws,
            grpc,
            mcp,
            terminal,
            host,
        } => serve::cmd_serve(&config, http, ws, grpc, mcp, terminal, host).await?,
        Commands::Mcp => cmd_mcp(&config).await?,
        Commands::Export { output } => export::cmd_export(&config, output.as_deref()).await?,
        Commands::Import { file, dry_run } => export::cmd_import(&config, &file, dry_run).await?,
        Commands::Service { action } => match action {
            service::ServiceAction::Install => service::cmd_service_install().await?,
            service::ServiceAction::Uninstall => service::cmd_service_uninstall()?,
        },
        Commands::Deps { action } => deps::cmd_deps(action)?,
        Commands::Bridge { url, api_key } => {
            bridge::cmd_bridge(&config, &url, api_key.as_deref()).await?
        }
        Commands::Vault { action } => vault::cmd_vault(&config, action).await?,
        Commands::Config { action } => config::cmd_config(&config, action)?,
        Commands::Capabilities => capabilities::cmd_capabilities(&config).await?,
        Commands::Update { check, yes, tag } => update::cmd_update(check, yes, tag).await?,
        Commands::Tail {
            kind,
            tool_id,
            correlation,
            principal,
            since,
        } => {
            tail::cmd_tail(
                &config,
                tail::TailFilter {
                    kind,
                    tool_id,
                    correlation,
                    principal,
                    since,
                },
            )
            .await?
        }
        Commands::Logs { action } => logs::cmd_logs(&config, action).await?,
        Commands::Baseline { action } => baseline::cmd_baseline(&config, action).await?,
    }

    Ok(())
}

/// `brain mcp` — proxy stdio JSON-RPC through the daemon's MCP HTTP transport.
///
/// Routing everything through the daemon keeps a single shared `SignalProcessor`
/// (no ruvector lock contention, no memory isolation, no double passphrase
/// prompts when an MCP client spawns us as a subprocess).
async fn cmd_mcp(config: &brain::BrainConfig) -> anyhow::Result<()> {
    // Detect the daemon via the HTTP adapter's /health endpoint…
    let _daemon_url = bootstrap::require_daemon(config).await?;
    // …but proxy to the MCP adapter's own HTTP port (separate from REST).
    let mcp_host = &config.adapters.http.host;
    let mcp_port = config.adapters.mcp.port;
    let mcp_url = format!("http://{mcp_host}:{mcp_port}/mcp");
    tracing::info!(url = %mcp_url, "Daemon detected — proxying MCP stdio through HTTP");
    bootstrap::proxy_mcp_stdio(&mcp_url, config).await
}
