//! Action dispatch — tool execution.
//!
//! Dispatches tool calls from LLM: command execution (sandboxed),
//! web search, scheduling, memory operations, and message sending.

use std::sync::Arc;

use thiserror::Error;

mod tooling;
mod validation;

#[cfg(test)]
mod tests;

pub use tooling::{get_available_tools, ToolDefinition};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from action execution.
#[derive(Debug, Error)]
pub enum ActionError {
    #[error("Command not allowed: {0}")]
    CommandNotAllowed(String),

    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Timeout")]
    Timeout,

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ─── Action Types ───────────────────────────────────────────────────────────

/// Available actions/tools.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Execute a shell command (sandboxed).
    ExecuteCommand { command: String, args: Vec<String> },
    /// Search the web.
    WebSearch { query: String },
    /// Schedule a task.
    ScheduleTask {
        description: String,
        cron: Option<String>,
    },
    /// Store a fact in semantic memory.
    StoreFact {
        subject: String,
        predicate: String,
        object: String,
    },
    /// Recall from memory.
    Recall { query: String },
    /// Send a message to an external endpoint (via protocol adapters).
    SendMessage {
        channel: String,
        recipient: String,
        content: String,
    },
}

/// Result of an action execution.
#[derive(Debug, Clone)]
pub struct ActionResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

/// Normalized memory fact used by action backends.
#[derive(Debug, Clone)]
pub struct MemoryFact {
    pub namespace: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Optional backend that provides real memory read/write operations.
#[async_trait::async_trait]
pub trait MemoryBackend: Send + Sync {
    async fn store_fact(
        &self,
        namespace: &str,
        category: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<String, ActionError>;

    async fn recall(
        &self,
        query: &str,
        top_k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<MemoryFact>, ActionError>;
}

/// Structured web-search hit returned by WebSearchBackend.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Optional backend for web search actions.
#[async_trait::async_trait]
pub trait WebSearchBackend: Send + Sync {
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, ActionError>;
}

/// Structured scheduling outcome returned by SchedulingBackend.
#[derive(Debug, Clone)]
pub struct ScheduleOutcome {
    pub schedule_id: String,
    pub status: String,
}

/// Optional backend for scheduling actions.
#[async_trait::async_trait]
pub trait SchedulingBackend: Send + Sync {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<ScheduleOutcome, ActionError>;
}

/// Structured message-delivery outcome returned by MessageBackend.
#[derive(Debug, Clone)]
pub struct MessageOutcome {
    pub delivery_id: String,
    pub status: String,
}

/// Optional backend for outbound message actions.
#[async_trait::async_trait]
pub trait MessageBackend: Send + Sync {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<MessageOutcome, ActionError>;
}

impl ActionResult {
    /// Create a successful result.
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    /// Create a failed result.
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }
}

// ─── Action Dispatcher ──────────────────────────────────────────────────────

/// Configuration for action execution.
#[derive(Debug, Clone)]
pub struct ActionConfig {
    /// Allowed commands for execution.
    pub command_allowlist: Vec<String>,
    /// Timeout for command execution (seconds).
    pub command_timeout_secs: u64,
    /// Enable web search.
    pub enable_web_search: bool,
    /// Enable scheduling.
    pub enable_scheduling: bool,
    /// Enable channel sends.
    pub enable_channel_send: bool,
    /// Default number of hits to request from the web search backend.
    pub web_search_top_k: usize,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            command_allowlist: vec![
                "ls".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "git".to_string(),
                "cargo".to_string(),
                "rustc".to_string(),
                "pwd".to_string(),
            ],
            command_timeout_secs: 30,
            enable_web_search: true,
            enable_scheduling: false,
            enable_channel_send: false,
            web_search_top_k: 5,
        }
    }
}

/// Dispatches actions/tools.
pub struct ActionDispatcher {
    config: ActionConfig,
    memory_backend: Option<Arc<dyn MemoryBackend>>,
    web_search_backend: Option<Arc<dyn WebSearchBackend>>,
    scheduling_backend: Option<Arc<dyn SchedulingBackend>>,
    message_backend: Option<Arc<dyn MessageBackend>>,
    namespace: String,
}

impl ActionDispatcher {
    /// Create a new dispatcher.
    pub fn new(config: ActionConfig) -> Self {
        Self {
            config,
            memory_backend: None,
            web_search_backend: None,
            scheduling_backend: None,
            message_backend: None,
            namespace: "personal".to_string(),
        }
    }

    /// Create a new dispatcher with a memory backend attached.
    pub fn with_memory_backend(
        config: ActionConfig,
        memory_backend: Arc<dyn MemoryBackend>,
    ) -> Self {
        Self::new(config).with_memory(memory_backend)
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ActionConfig::default())
    }

    /// Attach a memory backend.
    pub fn with_memory(mut self, memory_backend: Arc<dyn MemoryBackend>) -> Self {
        self.memory_backend = Some(memory_backend);
        self
    }

    /// Attach a web-search backend.
    pub fn with_web_search_backend(mut self, backend: Arc<dyn WebSearchBackend>) -> Self {
        self.web_search_backend = Some(backend);
        self
    }

    /// Attach a scheduling backend.
    pub fn with_scheduling_backend(mut self, backend: Arc<dyn SchedulingBackend>) -> Self {
        self.scheduling_backend = Some(backend);
        self
    }

    /// Attach a message backend.
    pub fn with_message_backend(mut self, backend: Arc<dyn MessageBackend>) -> Self {
        self.message_backend = Some(backend);
        self
    }

    /// Set the default namespace used by action backends.
    pub fn set_namespace(&mut self, namespace: impl Into<String>) {
        self.namespace = namespace.into();
    }

    fn active_namespace(&self) -> &str {
        let trimmed = self.namespace.trim();
        if trimmed.is_empty() {
            "personal"
        } else {
            trimmed
        }
    }

    /// Execute an action.
    pub async fn dispatch(&self, action: &Action) -> ActionResult {
        match action {
            Action::ExecuteCommand { command, args } => self.execute_command(command, args).await,
            Action::WebSearch { query } => self.web_search(query).await,
            Action::ScheduleTask { description, cron } => {
                self.schedule_task(description, cron.as_deref()).await
            }
            Action::StoreFact {
                subject,
                predicate,
                object,
            } => self.store_fact(subject, predicate, object).await,
            Action::Recall { query } => self.recall(query).await,
            Action::SendMessage {
                channel,
                recipient,
                content,
            } => self.send_message(channel, recipient, content).await,
        }
    }

    /// Execute a sandboxed command.
    async fn execute_command(&self, command: &str, args: &[String]) -> ActionResult {
        // Check allowlist
        if !self.config.command_allowlist.iter().any(|c| c == command) {
            return ActionResult::failure(format!("Command '{command}' is not in the allowlist"));
        }

        // Validate arguments against deny-lists
        if let Err(reason) = validation::validate_args(command, args) {
            return ActionResult::failure(format!("Blocked: {}", reason));
        }

        // Build command
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Execute with timeout
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(self.config.command_timeout_secs),
            cmd.output(),
        )
        .await
        {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    ActionResult::success(stdout.to_string())
                } else {
                    ActionResult::failure(format!(
                        "Exit code: {:?}\nstderr: {}",
                        output.status.code(),
                        stderr
                    ))
                }
            }
            Ok(Err(e)) => ActionResult::failure(format!("Failed to execute: {}", e)),
            Err(_) => ActionResult::failure("Command timed out"),
        }
    }

    /// Search the web.
    async fn web_search(&self, query: &str) -> ActionResult {
        if !self.config.enable_web_search {
            return ActionResult::failure("Web search is disabled by config");
        }
        let Some(backend) = &self.web_search_backend else {
            return ActionResult::failure("Web search backend not configured");
        };
        let top_k = self.config.web_search_top_k.max(1);
        match backend.search(query, top_k).await {
            Ok(hits) => {
                if hits.is_empty() {
                    return ActionResult::success(format!(
                        "web_search ok query=\"{}\" top_k={} hits=0",
                        query, top_k
                    ));
                }
                let lines = hits
                    .iter()
                    .enumerate()
                    .map(|(i, hit)| {
                        format!("{}. {} ({}) - {}", i + 1, hit.title, hit.url, hit.snippet)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ActionResult::success(format!(
                    "web_search ok query=\"{}\" top_k={} hits={}\n{}",
                    query,
                    top_k,
                    hits.len(),
                    lines
                ))
            }
            Err(e) => ActionResult::failure(format!("Web search failed: {e}")),
        }
    }

    /// Schedule a task.
    async fn schedule_task(&self, description: &str, cron: Option<&str>) -> ActionResult {
        if !self.config.enable_scheduling {
            return ActionResult::failure("Scheduling is disabled by config");
        }
        let Some(backend) = &self.scheduling_backend else {
            return ActionResult::failure("Scheduling backend not configured");
        };
        let namespace = self.active_namespace();
        match backend.schedule(description, cron, namespace).await {
            Ok(outcome) => ActionResult::success(format!(
                "schedule_task ok id={} status={} namespace={} cron={} description=\"{}\"",
                outcome.schedule_id,
                outcome.status,
                namespace,
                cron.unwrap_or("none"),
                description
            )),
            Err(e) => ActionResult::failure(format!("Schedule task failed: {e}")),
        }
    }

    /// Store a fact in semantic memory.
    async fn store_fact(&self, subject: &str, predicate: &str, object: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };
        let namespace = self.active_namespace();

        match memory
            .store_fact(namespace, "action", subject, predicate, object)
            .await
        {
            Ok(id) => ActionResult::success(format!(
                "Fact stored [{}] [{}]: {} {} {}",
                id, namespace, subject, predicate, object
            )),
            Err(e) => ActionResult::failure(format!("Failed to store fact: {e}")),
        }
    }

    /// Recall from memory.
    async fn recall(&self, query: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };
        let namespace = self.active_namespace();

        match memory.recall(query, 10, Some(namespace)).await {
            Ok(results) if results.is_empty() => ActionResult::success("No matching facts found."),
            Ok(results) => {
                let lines = results
                    .iter()
                    .map(|r| {
                        format!(
                            "[{}] {} {} {} (confidence: {:.2})",
                            r.namespace, r.subject, r.predicate, r.object, r.confidence
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                ActionResult::success(format!("Found {} fact(s):\n{}", results.len(), lines))
            }
            Err(e) => ActionResult::failure(format!("Recall failed: {e}")),
        }
    }

    /// Send a message via channel.
    async fn send_message(&self, channel: &str, recipient: &str, content: &str) -> ActionResult {
        if !self.config.enable_channel_send {
            return ActionResult::failure("Channel sending is disabled by config");
        }
        let Some(backend) = &self.message_backend else {
            return ActionResult::failure("Message backend not configured");
        };
        let namespace = self.active_namespace();
        match backend.send(channel, recipient, content, namespace).await {
            Ok(outcome) => ActionResult::success(format!(
                "send_message ok id={} status={} channel={} recipient={} namespace={}",
                outcome.delivery_id, outcome.status, channel, recipient, namespace
            )),
            Err(e) => ActionResult::failure(format!("Send message failed: {e}")),
        }
    }
}
