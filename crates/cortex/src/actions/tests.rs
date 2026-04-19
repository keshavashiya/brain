use std::sync::{Arc, Mutex};

use super::*;

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

struct MockWebSearchBackend;

#[async_trait::async_trait]
impl WebSearchBackend for MockWebSearchBackend {
    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>, ActionError> {
        Ok((0..top_k)
            .map(|i| SearchHit {
                title: format!("{query} hit {}", i + 1),
                url: format!("https://example.com/{i}"),
                snippet: "snippet".to_string(),
            })
            .collect())
    }
}

struct MockSchedulingBackend {
    calls: Mutex<Vec<(String, Option<String>, String)>>,
}

#[async_trait::async_trait]
impl SchedulingBackend for MockSchedulingBackend {
    async fn schedule(
        &self,
        description: &str,
        cron: Option<&str>,
        namespace: &str,
    ) -> Result<ScheduleOutcome, ActionError> {
        self.calls.lock().expect("calls lock").push((
            description.to_string(),
            cron.map(|c| c.to_string()),
            namespace.to_string(),
        ));
        Ok(ScheduleOutcome {
            schedule_id: "sched-1".to_string(),
            status: "scheduled".to_string(),
        })
    }
}

struct MockMessageBackend {
    calls: Mutex<Vec<(String, String, String, String)>>,
}

#[async_trait::async_trait]
impl MessageBackend for MockMessageBackend {
    async fn send(
        &self,
        channel: &str,
        recipient: &str,
        content: &str,
        namespace: &str,
    ) -> Result<MessageOutcome, ActionError> {
        self.calls.lock().expect("calls lock").push((
            channel.to_string(),
            recipient.to_string(),
            content.to_string(),
            namespace.to_string(),
        ));
        Ok(MessageOutcome {
            delivery_id: "msg-1".to_string(),
            status: "accepted".to_string(),
        })
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
    assert_eq!(config.web_search_top_k, 5);
}

#[tokio::test]
async fn test_execute_allowed_command() {
    let dispatcher = ActionDispatcher::with_defaults();
    let action = Action::ExecuteCommand {
        command: "ls".to_string(),
        args: vec!["-la".to_string()],
    };

    let result = dispatcher.dispatch(&action).await;
    assert!(result.success);
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
    let mut dispatcher =
        ActionDispatcher::with_memory_backend(ActionConfig::default(), backend.clone());

    dispatcher.set_namespace("work");
    let store = Action::StoreFact {
        subject: "user".to_string(),
        predicate: "likes".to_string(),
        object: "Rust".to_string(),
    };
    let _ = dispatcher.dispatch(&store).await;

    dispatcher.set_namespace("personal");
    let store_personal = Action::StoreFact {
        subject: "user".to_string(),
        predicate: "likes".to_string(),
        object: "Go".to_string(),
    };
    let _ = dispatcher.dispatch(&store_personal).await;

    dispatcher.set_namespace("work");
    let recall = Action::Recall {
        query: "Rust".to_string(),
    };
    let result = dispatcher.dispatch(&recall).await;
    assert!(result.success);
    assert!(result.output.contains("Found 1 fact"));
    assert!(result.output.contains("[work]"));
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

#[tokio::test]
async fn test_web_search_disabled() {
    let cfg = ActionConfig {
        enable_web_search: false,
        ..ActionConfig::default()
    };
    let dispatcher = ActionDispatcher::new(cfg);
    let result = dispatcher
        .dispatch(&Action::WebSearch {
            query: "rust".to_string(),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("disabled by config"));
}

#[tokio::test]
async fn test_web_search_backend_not_configured() {
    let dispatcher = ActionDispatcher::with_defaults();
    let result = dispatcher
        .dispatch(&Action::WebSearch {
            query: "rust".to_string(),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("backend not configured"));
}

#[tokio::test]
async fn test_web_search_success_with_backend() {
    let dispatcher =
        ActionDispatcher::with_defaults().with_web_search_backend(Arc::new(MockWebSearchBackend));
    let result = dispatcher
        .dispatch(&Action::WebSearch {
            query: "rust".to_string(),
        })
        .await;
    assert!(result.success);
    assert!(result.output.contains("web_search ok"));
    assert!(result.output.contains("hits=5"));
}

#[tokio::test]
async fn test_schedule_task_backend_matrix() {
    let mut disabled = ActionConfig {
        enable_scheduling: false,
        ..ActionConfig::default()
    };
    let dispatcher = ActionDispatcher::new(disabled.clone());
    let result = dispatcher
        .dispatch(&Action::ScheduleTask {
            description: "ship release".to_string(),
            cron: Some("0 10 * * 1".to_string()),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("disabled by config"));

    disabled.enable_scheduling = true;
    let unconfigured = ActionDispatcher::new(disabled.clone());
    let result = unconfigured
        .dispatch(&Action::ScheduleTask {
            description: "ship release".to_string(),
            cron: Some("0 10 * * 1".to_string()),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("backend not configured"));

    let backend = Arc::new(MockSchedulingBackend {
        calls: Mutex::new(Vec::new()),
    });
    let backend_trait: Arc<dyn SchedulingBackend> = backend.clone();
    let mut configured = ActionDispatcher::new(disabled).with_scheduling_backend(backend_trait);
    configured.set_namespace("work");
    let result = configured
        .dispatch(&Action::ScheduleTask {
            description: "ship release".to_string(),
            cron: Some("0 10 * * 1".to_string()),
        })
        .await;
    assert!(result.success);
    let calls = backend.calls.lock().expect("calls lock");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].2, "work");
}

#[tokio::test]
async fn test_send_message_backend_matrix() {
    let mut disabled = ActionConfig {
        enable_channel_send: false,
        ..ActionConfig::default()
    };
    let dispatcher = ActionDispatcher::new(disabled.clone());
    let result = dispatcher
        .dispatch(&Action::SendMessage {
            channel: "ops".to_string(),
            recipient: "alice".to_string(),
            content: "deploy now".to_string(),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("disabled by config"));

    disabled.enable_channel_send = true;
    let unconfigured = ActionDispatcher::new(disabled.clone());
    let result = unconfigured
        .dispatch(&Action::SendMessage {
            channel: "ops".to_string(),
            recipient: "alice".to_string(),
            content: "deploy now".to_string(),
        })
        .await;
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap_or_default()
        .contains("backend not configured"));

    let backend = Arc::new(MockMessageBackend {
        calls: Mutex::new(Vec::new()),
    });
    let backend_trait: Arc<dyn MessageBackend> = backend.clone();
    let mut configured = ActionDispatcher::new(disabled).with_message_backend(backend_trait);
    configured.set_namespace("project-x");
    let result = configured
        .dispatch(&Action::SendMessage {
            channel: "ops".to_string(),
            recipient: "alice".to_string(),
            content: "deploy now".to_string(),
        })
        .await;
    assert!(result.success);
    let calls = backend.calls.lock().expect("calls lock");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].3, "project-x");
}

#[test]
fn test_validate_args_blocks_shell_metacharacters() {
    assert!(super::validation::validate_args("ls", &["&&".to_string(), "rm".to_string()]).is_err());
    assert!(
        super::validation::validate_args("echo", &["hello".to_string(), "|".to_string()]).is_err()
    );
    assert!(
        super::validation::validate_args("cat", &[">".to_string(), "/etc/passwd".to_string()])
            .is_err()
    );
}

#[test]
fn test_validate_args_blocks_exec_pattern() {
    assert!(
        super::validation::validate_args("find", &["/".to_string(), "-exec".to_string()]).is_err()
    );
    assert!(
        super::validation::validate_args("find", &["/tmp".to_string(), "-delete".to_string()])
            .is_err()
    );
}

#[test]
fn test_validate_args_blocks_dangerous_git_subcommands() {
    assert!(super::validation::validate_args("git", &["push".to_string()]).is_err());
    assert!(super::validation::validate_args("git", &["reset".to_string()]).is_err());
    assert!(super::validation::validate_args("git", &["clean".to_string()]).is_err());
}

#[test]
fn test_validate_args_allows_safe_git_subcommands() {
    assert!(super::validation::validate_args("git", &["status".to_string()]).is_ok());
    assert!(super::validation::validate_args("git", &["log".to_string()]).is_ok());
    assert!(super::validation::validate_args("git", &["diff".to_string()]).is_ok());
}

#[test]
fn test_validate_args_cargo_subcommand_allowlist() {
    assert!(super::validation::validate_args("cargo", &["build".to_string()]).is_ok());
    assert!(super::validation::validate_args("cargo", &["test".to_string()]).is_ok());
    assert!(super::validation::validate_args("cargo", &["install".to_string()]).is_err());
    assert!(super::validation::validate_args("cargo", &["publish".to_string()]).is_err());
}

#[test]
fn test_validate_args_blocks_absolute_paths_outside_home() {
    assert!(super::validation::validate_args("ls", &["/etc/shadow".to_string()]).is_err());
    assert!(super::validation::validate_args("cat", &["/var/log/syslog".to_string()]).is_err());
}

#[test]
fn test_validate_args_allows_tmp_paths() {
    let result = super::validation::validate_args("ls", &["/tmp/foo".to_string()]);
    assert!(result.is_ok() || result.unwrap_err().contains("HOME not set"));
}

#[test]
fn test_validate_args_allows_relative_paths() {
    assert!(super::validation::validate_args("ls", &["./src".to_string()]).is_ok());
    assert!(super::validation::validate_args("cat", &["Cargo.toml".to_string()]).is_ok());
}

#[tokio::test]
async fn test_execute_command_blocked_args() {
    let dispatcher = ActionDispatcher::with_defaults();
    let action = Action::ExecuteCommand {
        command: "find".to_string(),
        args: vec![
            "/".to_string(),
            "-exec".to_string(),
            "rm".to_string(),
            "-rf".to_string(),
            "{}".to_string(),
        ],
    };
    let result = dispatcher.dispatch(&action).await;
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("Blocked"));
}
