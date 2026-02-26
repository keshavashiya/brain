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
    /// Manage the background daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
}

#[derive(Subcommand)]
enum DaemonAction {
    /// Start the background daemon
    Start,
    /// Stop the background daemon
    Stop,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "brain=info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Load configuration (always works -- defaults are embedded in binary)
    let config = core::BrainConfig::load().unwrap_or_else(|e| {
        tracing::warn!("Failed to load config, using defaults: {e}");
        core::BrainConfig::default()
    });

    // Ensure data directories exist on every invocation
    config.ensure_data_dirs()?;

    match cli.command {
        Commands::Init { force } => {
            let data_dir = config.data_dir();
            println!("Initializing Brain...");
            println!("  Data dir:  {}", data_dir.display());

            // Write config file
            match core::BrainConfig::write_default_config(force)? {
                Some(path) => println!("  Config:    {} (created)", path.display()),
                None => println!("  Config:    {} (already exists, use --force to overwrite)",
                    core::BrainConfig::user_config_path().display()),
            }

            // List created directories
            let subdirs = ["db", "ruvector", "models", "logs", "exports"];
            for sub in &subdirs {
                println!("  Dir:       {}", data_dir.join(sub).display());
            }

            println!("\nBrain initialized. Edit {} to customize.",
                core::BrainConfig::user_config_path().display());
        }
        Commands::Chat { message } => {
            if let Some(msg) = message {
                // Non-interactive mode
                chat_non_interactive(&config, &msg).await?;
            } else {
                // Interactive REPL mode
                chat_interactive(&config).await?;
            }
        }
        Commands::Status => {
            show_status(&config).await?;
        }
        Commands::Daemon { action } => match action {
            DaemonAction::Start => {
                println!("Starting Brain daemon...");
                // TODO: Implement daemon
            }
            DaemonAction::Stop => {
                println!("Stopping Brain daemon...");
                // TODO: Implement daemon stop
            }
        },
    }

    Ok(())
}

async fn show_status(config: &core::BrainConfig) -> anyhow::Result<()> {
    println!("Brain Status");
    println!("  Version:    {}", env!("CARGO_PKG_VERSION"));
    println!("  Data dir:   {}", config.data_dir().display());
    println!("  LLM:        {} ({})", config.llm.model, config.llm.provider);
    println!("  Embedding:  {} ({}d)", config.embedding.model, config.embedding.dimensions);
    println!("  Encryption: {}", if config.encryption.enabled { "enabled" } else { "disabled" });
    println!("  SQLite:     {}", config.sqlite_path().display());
    println!("  RuVector:   {}", config.ruvector_path().display());
    println!("  Config:     {}", core::BrainConfig::user_config_path().display());

    // Check LLM health
    let llm_config = cortex::llm::ProviderConfig {
        provider: config.llm.provider.clone(),
        base_url: config.llm.base_url.clone(),
        api_key: None,
        model: config.llm.model.clone(),
        temperature: config.llm.temperature,
        max_tokens: config.llm.max_tokens as i32,
    };
    let provider = cortex::llm::create_provider(&llm_config);
    let llm_healthy = provider.health_check().await;
    println!("  LLM Health: {}", if llm_healthy { "connected" } else { "disconnected" });

    // Check database
    let db_path = config.sqlite_path();
    match storage::SqlitePool::open(&db_path) {
        Ok(pool) => {
            match pool.table_stats() {
                Ok(stats) => {
                    println!("\n  Database Tables:");
                    for (table, count) in stats {
                        println!("    {}: {} rows", table, count);
                    }
                }
                Err(e) => println!("\n  Database: error reading stats - {}", e),
            }
        }
        Err(e) => println!("\n  Database: error opening - {}", e),
    }

    Ok(())
}

async fn chat_non_interactive(config: &core::BrainConfig, message: &str) -> anyhow::Result<()> {
    // Initialize brain components
    let mut brain = BrainSession::new(config).await?;

    // Process the message
    let response = brain.process_message(message).await?;

    // Print response
    println!("{}", response);

    Ok(())
}

async fn chat_interactive(config: &core::BrainConfig) -> anyhow::Result<()> {
    // Print welcome message
    println!("╔═══════════════════════════════════════╗");
    println!("║  Brain v{}                          ║", env!("CARGO_PKG_VERSION"));
    println!("║  A personal AI that remembers you     ║");
    println!("╚═══════════════════════════════════════╝");
    println!();
    println!("  Model: {}", config.llm.model);
    println!("  Data:  {}", config.data_dir().display());
    println!();
    println!("Commands:");
    println!("  /status  - Show system status");
    println!("  /clear   - Clear conversation history");
    println!("  /quit    - Exit");
    println!();

    // Initialize brain session
    let mut brain = BrainSession::new(config).await?;

    // Create rustyline editor
    let mut rl = DefaultEditor::new()?;
    let history_path = config.data_dir().join("history.txt");
    let _ = rl.load_history(&history_path);

    // REPL loop
    loop {
        // Read input
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                if input.is_empty() {
                    continue;
                }

                // Add to history
                let _ = rl.add_history_entry(input);

                // Handle commands
                if input == "/quit" || input == "/exit" {
                    println!("Goodbye!");
                    break;
                }

                if input == "/status" {
                    show_status(config).await?;
                    continue;
                }

                if input == "/clear" {
                    brain.clear_history();
                    println!("Conversation history cleared.");
                    continue;
                }

                // Process message
                match brain.process_message(input).await {
                    Ok(response) => {
                        // Print assistant response with styling
                        let mut stdout = stdout();
                        stdout.execute(SetForegroundColor(Color::Green))?;
                        stdout.execute(Print("Brain: "))?;
                        stdout.execute(ResetColor)?;
                        println!("{}", response);
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Interrupted");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_path);

    Ok(())
}

/// A brain session handles the chat pipeline.
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
        // Open database
        let db = storage::SqlitePool::open(&config.sqlite_path())?;

        // Create stores
        let episodic = hippocampus::EpisodicStore::new(db.clone());

        // Create semantic store (requires RuVector - TODO: refactor for RuVector)
        let ruvector_path = config.ruvector_path();
        let semantic = if let Ok(lance) = storage::LanceStore::open(&ruvector_path).await {
            lance.ensure_tables().await.ok();
            Some(hippocampus::SemanticStore::new(db.clone(), lance))
        } else {
            None
        };

        // Create LLM provider
        let llm_config = cortex::llm::ProviderConfig {
            provider: config.llm.provider.clone(),
            base_url: config.llm.base_url.clone(),
            api_key: None,
            model: config.llm.model.clone(),
            temperature: config.llm.temperature,
            max_tokens: config.llm.max_tokens as i32,
        };
        let llm = cortex::llm::create_provider(&llm_config);

        // Create context assembler
        let context_assembler = cortex::context::ContextAssembler::with_defaults();

        // Create a new session
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

        // Store user message in episodic memory
        let importance = hippocampus::ImportanceScorer::score(message, true);
        self.episodic.store_episode(&self.session_id, "user", message, importance)?;

        // Route the message through thalamus
        let thalamus = thalamus::SignalRouter::new();
        let normalized_msg = thalamus::NormalizedMessage {
            content: message.to_string(),
            channel: "cli".to_string(),
            sender: "user".to_string(),
            timestamp: chrono::Utc::now(),
            message_id: None,
            metadata: std::collections::HashMap::new(),
        };
        let classification = thalamus.route(&normalized_msg);

        // Handle action intents
        if let Some(action) = thalamus.intent_to_action(&classification.intent) {
            let action_dispatcher = cortex::actions::ActionDispatcher::with_defaults();
            let result = action_dispatcher.dispatch(&action).await;

            if result.success {
                return Ok(result.output);
            } else {
                return Ok(format!("Error: {}", result.error.unwrap_or_default()));
            }
        }

        // For chat intents, process through LLM
        // 1. Retrieve relevant memories
        let mut memories = Vec::new();
        if let Some(ref _semantic) = self.semantic {
            // Get query embedding (simplified - would use actual embedder)
            // For now, just use BM25 search on episodic memory
            if let Ok(bm25_results) = self.episodic.search_bm25(message, 10) {
                for result in bm25_results {
                    memories.push(hippocampus::search::Memory {
                        id: result.episode_id,
                        content: result.content,
                        source: hippocampus::search::MemorySource::Episodic,
                        score: result.rank,
                        importance: 0.5,
                        timestamp: String::new(),
                    });
                }
            }
        }

        // 2. Assemble context
        let messages = self.context_assembler.assemble(
            message,
            &memories,
            &self.conversation_history,
        );

        // 3. Generate LLM response
        let response = self.llm.generate(&messages).await?;

        // 4. Store assistant response in episodic memory
        self.episodic.store_episode(
            &self.session_id,
            "assistant",
            &response.content,
            0.5,
        )?;

        // 5. Update conversation history
        self.conversation_history.push(Message {
            role: Role::User,
            content: message.to_string(),
        });
        self.conversation_history.push(Message {
            role: Role::Assistant,
            content: response.content.clone(),
        });

        // Keep history manageable (last 20 exchanges)
        if self.conversation_history.len() > 40 {
            self.conversation_history = self.conversation_history.split_off(20);
        }

        Ok(response.content)
    }
}
