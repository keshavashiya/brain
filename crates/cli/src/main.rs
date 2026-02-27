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
/// Uses `kill -0` on Unix which sends no signal but validates the PID.
fn is_process_running(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Send SIGTERM to a process.
fn stop_process(pid: u32) -> anyhow::Result<()> {
    let status = std::process::Command::new("kill")
        .arg(pid.to_string())
        .status()?;
    if !status.success() {
        anyhow::bail!("Failed to send SIGTERM to PID {}", pid);
    }
    Ok(())
}

/// Spawn `brain serve` as a detached background process and return its PID.
///
/// stdout and stderr are redirected to the log file.
/// On Unix the child is placed in its own process group so it survives
/// terminal close / SIGHUP.
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

    // Detach from the terminal's process group so Brain survives terminal close
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = cmd.spawn()?;
    Ok(child.id())
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
        Commands::Init { force } => {
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
            let run_all = !http && !ws && !grpc && !mcp;
            let processor = Arc::new(signal::SignalProcessor::new(config.clone()).await?);

            println!("Starting Brain OS...");

            let mut set = tokio::task::JoinSet::new();

            if run_all || http {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.http.port;
                println!("  HTTP  → http://{}:{}", h, port);
                set.spawn(async move { adapters_http::serve(p, &h, port).await });
            }

            if run_all || ws {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.ws.port;
                println!("  WS    → ws://{}:{}", h, port);
                set.spawn(async move { adapters_ws::serve(p, &h, port).await });
            }

            if run_all || grpc {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.grpc.port;
                println!("  gRPC  → {}:{}", h, port);
                set.spawn(async move { adapters_grpc::serve(p, &h, port).await });
            }

            if run_all || mcp {
                let p = processor.clone();
                let h = host.clone();
                let port = config.adapters.mcp.port;
                println!("  MCP   → http://{}:{}", h, port);
                set.spawn(async move { mcp::serve_http(p, &h, port).await });
            }

            println!("\nPress Ctrl+C to stop.\n");

            while let Some(result) = set.join_next().await {
                match result {
                    Ok(Err(e)) => eprintln!("Adapter error: {e}"),
                    Err(e) => eprintln!("Task panicked: {e}"),
                    Ok(Ok(())) => {}
                }
            }
        }

        // ── mcp stdio ─────────────────────────────────────────────────────────
        Commands::Mcp => {
            let processor = signal::SignalProcessor::new(config.clone()).await?;
            mcp::serve_stdio(processor).await?;
        }
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
            storage::RuVectorStore::open(&config.ruvector_path()).await
        {
            ruv.ensure_tables().await.ok();
            Some(hippocampus::SemanticStore::new(db.clone(), ruv))
        } else {
            None
        };

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
            llm,
            context_assembler,
            conversation_history: Vec::new(),
            session_id,
        })
    }

    fn clear_history(&mut self) {
        self.conversation_history.clear();
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

        // BM25 recall from episodic memory
        let mut memories = Vec::new();
        if self.semantic.is_some() {
            if let Ok(hits) = self.episodic.search_bm25(message, 10) {
                for h in hits {
                    memories.push(hippocampus::search::Memory {
                        id: h.episode_id,
                        content: h.content,
                        source: hippocampus::search::MemorySource::Episodic,
                        score: h.rank,
                        importance: 0.5,
                        timestamp: String::new(),
                    });
                }
            }
        }

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
