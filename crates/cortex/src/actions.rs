//! Action dispatch — tool execution.
//!
//! Dispatches tool calls from LLM: command execution (sandboxed),
//! web search, scheduling, memory operations, and message sending.

use std::sync::Arc;

use thiserror::Error;

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
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            command_allowlist: vec![
                "ls".to_string(),
                "cat".to_string(),
                "grep".to_string(),
                "find".to_string(),
                "git".to_string(),
                "cargo".to_string(),
                "rustc".to_string(),
                "pwd".to_string(),
                "echo".to_string(),
                "head".to_string(),
                "tail".to_string(),
            ],
            command_timeout_secs: 30,
            enable_web_search: true,
            enable_scheduling: false,
            enable_channel_send: false,
        }
    }
}

/// Dispatches actions/tools.
pub struct ActionDispatcher {
    config: ActionConfig,
    memory_backend: Option<Arc<dyn MemoryBackend>>,
}

impl ActionDispatcher {
    /// Create a new dispatcher.
    pub fn new(config: ActionConfig) -> Self {
        Self {
            config,
            memory_backend: None,
        }
    }

    /// Create a new dispatcher with a memory backend attached.
    pub fn with_memory_backend(
        config: ActionConfig,
        memory_backend: Arc<dyn MemoryBackend>,
    ) -> Self {
        Self {
            config,
            memory_backend: Some(memory_backend),
        }
    }

    /// Create with default config.
    pub fn with_defaults() -> Self {
        Self::new(ActionConfig::default())
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
        if !self.config.command_allowlist.contains(&command.to_string()) {
            return ActionResult::failure(format!("Command '{}' is not in the allowlist", command));
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
            return ActionResult::failure("Web search is disabled");
        }

        // Placeholder: In a real implementation, this would use Brave/SearXNG API
        // For now, return a mock result
        tracing::info!("Web search query: {}", query);
        ActionResult::success(format!(
            "Web search for '{}' would be performed here. (Not implemented in v0.1)",
            query
        ))
    }

    /// Schedule a task.
    async fn schedule_task(&self, description: &str, cron: Option<&str>) -> ActionResult {
        if !self.config.enable_scheduling {
            return ActionResult::failure("Scheduling is disabled");
        }

        tracing::info!("Schedule task: {} (cron: {:?})", description, cron);
        ActionResult::success(format!(
            "Task '{}' scheduled (cron: {:?}). (Not fully implemented in v0.1)",
            description, cron
        ))
    }

    /// Store a fact in semantic memory.
    async fn store_fact(&self, subject: &str, predicate: &str, object: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };

        match memory
            .store_fact("personal", "action", subject, predicate, object)
            .await
        {
            Ok(id) => ActionResult::success(format!(
                "Fact stored [{}]: {} {} {}",
                id, subject, predicate, object
            )),
            Err(e) => ActionResult::failure(format!("Failed to store fact: {e}")),
        }
    }

    /// Recall from memory.
    async fn recall(&self, query: &str) -> ActionResult {
        let Some(memory) = &self.memory_backend else {
            return ActionResult::failure("Memory backend not available");
        };

        match memory.recall(query, 10, Some("personal")).await {
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
            return ActionResult::failure("Channel sending is disabled");
        }

        tracing::info!("Send message via {} to {}: {}", channel, recipient, content);
        ActionResult::success(format!(
            "Message queued to send via {} to {}",
            channel, recipient
        ))
    }
}

// ─── Tool Definition for LLM ────────────────────────────────────────────────

/// Tool definition for LLM function calling.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Get available tools as LLM function definitions.
pub fn get_available_tools() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "execute_command".to_string(),
            description: "Execute a sandboxed shell command".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command arguments"
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web for information".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "store_fact".to_string(),
            description: "Store a fact in memory".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"}
                },
                "required": ["subject", "predicate", "object"]
            }),
        },
        ToolDefinition {
            name: "recall".to_string(),
            description: "Search memory for relevant information".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for"
                    }
                },
                "required": ["query"]
            }),
        },
    ]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct MockMemoryBackend {
        facts: Mutex<Vec<MemoryFact>>,
    }

    #[async_trait::async_trait]
    impl MemoryBackend for MockMemoryBackend {
        async fn store_fact(
            &self,
            namespace: &str,
            _category: &str,
            subject: &str,
            predicate: &str,
            object: &str,
        ) -> Result<String, ActionError> {
            self.facts.lock().unwrap().push(MemoryFact {
                namespace: namespace.to_string(),
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                object: object.to_string(),
                confidence: 1.0,
            });
            Ok("fact-1".to_string())
        }

        async fn recall(
            &self,
            query: &str,
            _top_k: usize,
            namespace: Option<&str>,
        ) -> Result<Vec<MemoryFact>, ActionError> {
            let facts = self.facts.lock().unwrap();
            Ok(facts
                .iter()
                .filter(|f| {
                    namespace.is_none_or(|ns| f.namespace == ns)
                        && (f.subject.contains(query)
                            || f.predicate.contains(query)
                            || f.object.contains(query))
                })
                .cloned()
                .collect())
        }
    }

    #[test]
    fn test_action_result_success() {
        let result = ActionResult::success("output");
        assert!(result.success);
        assert_eq!(result.output, "output");
        assert!(result.error.is_none());
    }

    #[test]
    fn test_action_result_failure() {
        let result = ActionResult::failure("error");
        assert!(!result.success);
        assert_eq!(result.error, Some("error".to_string()));
    }

    #[test]
    fn test_action_config_default() {
        let config = ActionConfig::default();
        assert!(config.command_allowlist.contains(&"ls".to_string()));
        assert_eq!(config.command_timeout_secs, 30);
        assert!(config.enable_web_search);
    }

    #[tokio::test]
    async fn test_execute_allowed_command() {
        let dispatcher = ActionDispatcher::with_defaults();
        let action = Action::ExecuteCommand {
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
        };

        let result = dispatcher.dispatch(&action).await;
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_disallowed_command() {
        let dispatcher = ActionDispatcher::with_defaults();
        let action = Action::ExecuteCommand {
            command: "rm".to_string(),
            args: vec!["-rf".to_string(), "/".to_string()],
        };

        let result = dispatcher.dispatch(&action).await;
        assert!(!result.success);
        assert!(result
            .error
            .as_ref()
            .unwrap()
            .contains("not in the allowlist"));
    }

    #[test]
    fn test_get_available_tools() {
        let tools = get_available_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "execute_command"));
        assert!(tools.iter().any(|t| t.name == "web_search"));
    }

    #[tokio::test]
    async fn test_store_fact_with_memory_backend() {
        let backend = Arc::new(MockMemoryBackend {
            facts: Mutex::new(Vec::new()),
        });
        let dispatcher = ActionDispatcher::with_memory_backend(ActionConfig::default(), backend);

        let action = Action::StoreFact {
            subject: "user".to_string(),
            predicate: "likes".to_string(),
            object: "Rust".to_string(),
        };
        let result = dispatcher.dispatch(&action).await;
        assert!(result.success);
        assert!(result.output.contains("Fact stored"));
    }

    #[tokio::test]
    async fn test_recall_with_memory_backend() {
        let backend = Arc::new(MockMemoryBackend {
            facts: Mutex::new(Vec::new()),
        });
        let dispatcher =
            ActionDispatcher::with_memory_backend(ActionConfig::default(), backend.clone());

        let store = Action::StoreFact {
            subject: "user".to_string(),
            predicate: "likes".to_string(),
            object: "Rust".to_string(),
        };
        let _ = dispatcher.dispatch(&store).await;

        let recall = Action::Recall {
            query: "Rust".to_string(),
        };
        let result = dispatcher.dispatch(&recall).await;
        assert!(result.success);
        assert!(result.output.contains("Found 1 fact"));
    }

    #[tokio::test]
    async fn test_memory_actions_fail_without_backend() {
        let dispatcher = ActionDispatcher::with_defaults();
        let action = Action::Recall {
            query: "anything".to_string(),
        };
        let result = dispatcher.dispatch(&action).await;
        assert!(!result.success);
        assert!(result
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("Memory backend not available"));
    }
}
