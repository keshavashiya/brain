//! AgentRegistry — the orchestrator's lookup table for delegates.

use std::collections::HashMap;
use std::sync::Arc;

use crate::traits::{AgentDelegate, AgentError};

/// Holds every known delegate keyed by `name()`. Additional aliases can
/// be registered to route requests like `"claude"` to the canonical
/// `"claude-code"` entry.
#[derive(Default)]
pub struct AgentRegistry {
    delegates: HashMap<String, Arc<dyn AgentDelegate>>,
    aliases: HashMap<String, String>,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a delegate under its declared `name()`. Last write wins.
    pub fn register(&mut self, delegate: Arc<dyn AgentDelegate>) {
        let name = delegate.name().to_string();
        self.delegates.insert(name, delegate);
    }

    /// Add an alias: `alias -> canonical_name`. If `canonical_name` isn't
    /// registered yet, the alias is still stored — resolved lazily.
    pub fn alias(&mut self, alias: impl Into<String>, canonical: impl Into<String>) {
        self.aliases.insert(alias.into(), canonical.into());
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn AgentDelegate>, AgentError> {
        let resolved = self.aliases.get(name).map(String::as_str).unwrap_or(name);
        self.delegates
            .get(resolved)
            .cloned()
            .ok_or_else(|| AgentError::NotFound(name.to_string()))
    }

    pub fn contains(&self, name: &str) -> bool {
        let resolved = self.aliases.get(name).map(String::as_str).unwrap_or(name);
        self.delegates.contains_key(resolved)
    }

    /// Ordered list of canonical delegate names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.delegates.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn is_empty(&self) -> bool {
        self.delegates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{AgentCapabilities, AgentResult, AgentTask, AgentTaskStatus};
    use async_trait::async_trait;
    use chrono::Utc;

    struct MockAgent {
        name: String,
    }

    #[async_trait]
    impl AgentDelegate for MockAgent {
        fn name(&self) -> &str {
            &self.name
        }
        fn capabilities(&self) -> AgentCapabilities {
            AgentCapabilities::default()
        }
        async fn delegate(&self, task: AgentTask) -> Result<AgentResult, AgentError> {
            let now = Utc::now();
            Ok(AgentResult {
                task_id: task.id,
                status: AgentTaskStatus::Succeeded,
                summary: format!("{} ran: {}", self.name, task.description),
                artifacts: vec![],
                stdout: String::new(),
                stderr: String::new(),
                exit_code: Some(0),
                started_at: now,
                completed_at: now,
            })
        }
    }

    #[test]
    fn register_and_get() {
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(MockAgent {
            name: "mock".to_string(),
        }));
        let d = reg.get("mock").unwrap();
        assert_eq!(d.name(), "mock");
    }

    #[test]
    fn alias_resolves() {
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(MockAgent {
            name: "claude-code".to_string(),
        }));
        reg.alias("claude", "claude-code");
        assert!(reg.contains("claude"));
        assert_eq!(reg.get("claude").unwrap().name(), "claude-code");
    }

    #[test]
    fn missing_delegate_errors() {
        let reg = AgentRegistry::new();
        assert!(matches!(reg.get("nope"), Err(AgentError::NotFound(_))));
    }

    #[test]
    fn list_returns_sorted_names() {
        let mut reg = AgentRegistry::new();
        reg.register(Arc::new(MockAgent {
            name: "b".to_string(),
        }));
        reg.register(Arc::new(MockAgent {
            name: "a".to_string(),
        }));
        assert_eq!(reg.list(), vec!["a".to_string(), "b".to_string()]);
    }
}
