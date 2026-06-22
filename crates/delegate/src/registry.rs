//! AgentRegistry — the orchestrator's lookup table for delegates.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::definition::AgentDefinition;
use crate::discovery::{DiscoveredBinary, DiscoveryStatus};
use crate::subprocess::{SubprocessAgentConfig, SubprocessAgentDelegate};
use crate::traits::{AgentCapabilities, AgentDelegate, AgentError};

/// Operator-configured override for a fingerprinted agent. All fields
/// are optional; unset ones fall through to discovery defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentOverride {
    /// Override the binary path found on `$PATH`.
    pub binary: Option<PathBuf>,
    /// Exclude this agent from the registry entirely.
    #[serde(default)]
    pub disabled: bool,
    /// Replace the fingerprint's default capabilities.
    pub capabilities: Option<AgentCapabilities>,
    /// Replace the invocation args. `{prompt}` and `{task_id}` are
    /// substituted at spawn time.
    pub args: Option<Vec<String>>,
    /// Force stdin vs. argv prompt delivery.
    pub prompt_via_stdin: Option<bool>,
}

fn default_true() -> bool {
    true
}

/// Full delegate config sourced from user settings. Empty default is
/// fine — pure discovery with no overrides.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DelegateOverrides {
    /// Toggle auto-discovery; custom entries still register.
    #[serde(default = "default_true")]
    pub auto_discovery: bool,
    /// Keyed by canonical agent id.
    #[serde(default)]
    pub overrides: HashMap<String, AgentOverride>,
    /// Explicit-path agent definitions (the old `CustomAgentSpec` case):
    /// registered directly, bypassing the `$PATH` scan.
    #[serde(default)]
    pub custom: Vec<AgentDefinition>,
}

impl DelegateOverrides {
    pub fn new() -> Self {
        Self {
            auto_discovery: true,
            ..Self::default()
        }
    }
}

/// What the registry knows about a candidate agent — populated by
/// [`AgentRegistry::populate_from_discovery`] so callers (e.g.
/// `brain doctor`, `QueryAgents` intent) can explain availability.
#[derive(Debug, Clone)]
pub enum RegistryAgentStatus {
    Registered {
        binary: PathBuf,
        version: Option<String>,
        source: AgentSource,
    },
    DisabledByConfig,
    Unavailable {
        binary: PathBuf,
        reason: String,
    },
}

/// Where a registered agent came from. Distinguishes auto-discovered
/// entries from hand-registered / custom ones so the operator surface
/// can be honest about "why is this here".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSource {
    Discovered,
    Custom,
    Manual,
}

/// Holds every known delegate keyed by `name()`. Additional aliases can
/// be registered to route shorthand request names to canonical entries.
#[derive(Default)]
pub struct AgentRegistry {
    delegates: HashMap<String, Arc<dyn AgentDelegate>>,
    aliases: HashMap<String, String>,
    agent_status: HashMap<String, RegistryAgentStatus>,
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

    /// Register a delegate that came from hand-written `agents.delegates[]`
    /// config. Records `agent_status` as [`AgentSource::Manual`] so the
    /// operator surface (`QueryAgents`, doctor) sees the entry instead of
    /// silently diverging from what the orchestrator will actually run.
    pub fn register_manual(
        &mut self,
        delegate: Arc<dyn AgentDelegate>,
        binary: PathBuf,
        version: Option<String>,
    ) {
        let name = delegate.name().to_string();
        self.agent_status.insert(
            name.clone(),
            RegistryAgentStatus::Registered {
                binary,
                version,
                source: AgentSource::Manual,
            },
        );
        self.delegates.insert(name, delegate);
    }

    /// Build a `SubprocessAgentDelegate` from an explicit-path
    /// [`AgentDefinition`] and register it under `def.id` with the given
    /// source attribution. Returns the binary path so callers can reuse it
    /// for status/log messages.
    ///
    /// This is the canonical entry point for both the YAML
    /// `agents.delegates[]` flow (source = [`AgentSource::Manual`]) and the
    /// programmatic [`DelegateOverrides::custom`] flow
    /// (source = [`AgentSource::Custom`]). Centralising the
    /// `SubprocessAgentConfig` construction here keeps the two paths from
    /// drifting — every new field on [`AgentDefinition`] is honoured by both.
    ///
    /// The binary is `def.binary` when set, else the first `binary_names`
    /// entry (resolved via `$PATH` at spawn), else `def.id` as a last resort.
    pub fn register_subprocess_spec(
        &mut self,
        def: &AgentDefinition,
        source: AgentSource,
    ) -> PathBuf {
        let binary: PathBuf = def
            .binary
            .clone()
            .or_else(|| def.binary_names.first().map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(&def.id));
        let cfg = SubprocessAgentConfig {
            name: def.id.clone(),
            binary: binary.to_string_lossy().into_owned(),
            args: def.args.clone(),
            workdir: def.workdir.clone(),
            capabilities: def.capabilities.clone(),
            prompt_via_stdin: def.prompt_via_stdin,
            version_args: def.version_args.clone(),
        };
        let delegate: Arc<dyn AgentDelegate> = Arc::new(SubprocessAgentDelegate::new(cfg));
        self.agent_status.insert(
            def.id.clone(),
            RegistryAgentStatus::Registered {
                binary: binary.clone(),
                version: None,
                source,
            },
        );
        if let Some(alias) = &def.alias {
            self.aliases.insert(alias.clone(), def.id.clone());
        }
        self.delegates.insert(def.id.clone(), delegate);
        binary
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

    /// Known state for an agent id — including ones we saw but did not
    /// register (disabled, probe-failed). Returns `None` for ids the
    /// discovery pass never encountered.
    pub fn agent_status(&self, id: &str) -> Option<&RegistryAgentStatus> {
        self.agent_status.get(id)
    }

    /// Every agent id we've learned about, whether registered or not.
    pub fn known_agents(&self) -> Vec<(&str, &RegistryAgentStatus)> {
        let mut v: Vec<_> = self
            .agent_status
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }

    /// Consume a discovery pass + user overrides and register runnable
    /// delegates for every `Available` agent that isn't disabled.
    ///
    /// Unavailable or disabled agents are *not* registered but their
    /// reason is retained in `agent_status` so the operator surface can
    /// answer "why isn't X available". Existing manual registrations
    /// are left untouched.
    pub fn populate_from_discovery(
        &mut self,
        discovered: Vec<DiscoveredBinary>,
        overrides: &DelegateOverrides,
    ) {
        if overrides.auto_discovery {
            for d in discovered {
                let agent_id = d.definition.id.clone();
                let ov = overrides
                    .overrides
                    .get(&agent_id)
                    .cloned()
                    .unwrap_or_default();
                if ov.disabled {
                    self.agent_status
                        .insert(agent_id, RegistryAgentStatus::DisabledByConfig);
                    continue;
                }
                match &d.status {
                    DiscoveryStatus::Available => {
                        let binary = ov.binary.clone().unwrap_or_else(|| d.path.clone());
                        let caps = ov
                            .capabilities
                            .clone()
                            .unwrap_or_else(|| d.definition.capabilities.clone());
                        let delegate = build_from_template(&binary, &d.definition, &ov, caps);
                        self.agent_status.insert(
                            agent_id.clone(),
                            RegistryAgentStatus::Registered {
                                binary,
                                version: d.version.clone(),
                                source: AgentSource::Discovered,
                            },
                        );
                        // Preserve legacy/shorthand routing names (e.g.
                        // `claude-code` → `claude`) declared in the seed.
                        if let Some(alias) = &d.definition.alias {
                            self.aliases.insert(alias.clone(), agent_id.clone());
                        }
                        self.register(delegate);
                    }
                    DiscoveryStatus::Unavailable(reason) => {
                        self.agent_status.insert(
                            agent_id,
                            RegistryAgentStatus::Unavailable {
                                binary: d.path.clone(),
                                reason: reason.clone(),
                            },
                        );
                    }
                }
            }
        }

        for spec in &overrides.custom {
            self.register_subprocess_spec(spec, AgentSource::Custom);
        }
    }
}

fn build_from_template(
    binary: &Path,
    def: &AgentDefinition,
    ov: &AgentOverride,
    capabilities: AgentCapabilities,
) -> Arc<dyn AgentDelegate> {
    let args = ov.args.clone().unwrap_or_else(|| def.args.clone());
    let prompt_via_stdin = ov.prompt_via_stdin.unwrap_or(def.prompt_via_stdin);
    let cfg = SubprocessAgentConfig {
        name: def.id.clone(),
        binary: binary.to_string_lossy().into_owned(),
        args,
        workdir: def.workdir.clone(),
        capabilities,
        prompt_via_stdin,
        version_args: def.version_args.clone(),
    };
    Arc::new(SubprocessAgentDelegate::new(cfg))
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

    fn discovered(agent_id: &str, status: DiscoveryStatus) -> DiscoveredBinary {
        DiscoveredBinary {
            definition: AgentDefinition {
                id: agent_id.to_string(),
                alias: None,
                binary_names: vec![agent_id.to_string()],
                binary: None,
                version_args: vec!["--version".to_string()],
                args: Vec::new(),
                prompt_via_stdin: true,
                capabilities: AgentCapabilities::default(),
                workdir: None,
            },
            binary_name: agent_id.to_string(),
            path: PathBuf::from(format!("/usr/local/bin/{agent_id}")),
            version: Some(format!("{agent_id} 1.0")),
            status,
        }
    }

    /// Explicit-path definition for the custom/manual registration tests.
    fn custom_def(id: &str, binary: &str) -> AgentDefinition {
        AgentDefinition {
            id: id.to_string(),
            alias: None,
            binary_names: Vec::new(),
            binary: Some(PathBuf::from(binary)),
            version_args: vec!["--version".to_string()],
            args: Vec::new(),
            prompt_via_stdin: true,
            capabilities: AgentCapabilities::default(),
            workdir: None,
        }
    }

    #[test]
    fn populate_registers_available_agents() {
        let mut reg = AgentRegistry::new();
        let discovered = vec![
            discovered("codex", DiscoveryStatus::Available),
            discovered("aider", DiscoveryStatus::Available),
        ];
        reg.populate_from_discovery(discovered, &DelegateOverrides::new());
        assert!(reg.contains("codex"));
        assert!(reg.contains("aider"));
        assert!(matches!(
            reg.agent_status("codex"),
            Some(RegistryAgentStatus::Registered { .. })
        ));
    }

    #[test]
    fn populate_skips_unavailable_but_records_reason() {
        let mut reg = AgentRegistry::new();
        let discovered = vec![discovered(
            "broken",
            DiscoveryStatus::Unavailable("missing API key".into()),
        )];
        reg.populate_from_discovery(discovered, &DelegateOverrides::new());
        assert!(!reg.contains("broken"));
        match reg.agent_status("broken") {
            Some(RegistryAgentStatus::Unavailable { reason, .. }) => {
                assert!(reason.contains("missing API key"));
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn disabled_override_blocks_registration() {
        let mut reg = AgentRegistry::new();
        let mut overrides = DelegateOverrides::new();
        overrides.overrides.insert(
            "aider".to_string(),
            AgentOverride {
                disabled: true,
                ..Default::default()
            },
        );
        reg.populate_from_discovery(
            vec![discovered("aider", DiscoveryStatus::Available)],
            &overrides,
        );
        assert!(!reg.contains("aider"));
        assert!(matches!(
            reg.agent_status("aider"),
            Some(RegistryAgentStatus::DisabledByConfig)
        ));
    }

    #[test]
    fn auto_discovery_off_still_registers_custom() {
        let mut reg = AgentRegistry::new();
        let overrides = DelegateOverrides {
            auto_discovery: false,
            overrides: HashMap::new(),
            custom: vec![custom_def("mine", "/usr/bin/true")],
        };
        reg.populate_from_discovery(
            vec![discovered("aider", DiscoveryStatus::Available)],
            &overrides,
        );
        assert!(!reg.contains("aider"));
        assert!(reg.contains("mine"));
        match reg.agent_status("mine") {
            Some(RegistryAgentStatus::Registered { source, .. }) => {
                assert_eq!(*source, AgentSource::Custom);
            }
            other => panic!("expected Registered, got {other:?}"),
        }
    }

    #[test]
    fn override_binary_path_takes_precedence() {
        let mut reg = AgentRegistry::new();
        let mut overrides = DelegateOverrides::new();
        overrides.overrides.insert(
            "aider".to_string(),
            AgentOverride {
                binary: Some(PathBuf::from("/opt/custom/aider")),
                ..Default::default()
            },
        );
        reg.populate_from_discovery(
            vec![discovered("aider", DiscoveryStatus::Available)],
            &overrides,
        );
        match reg.agent_status("aider") {
            Some(RegistryAgentStatus::Registered { binary, .. }) => {
                assert_eq!(binary, &PathBuf::from("/opt/custom/aider"));
            }
            other => panic!("expected Registered, got {other:?}"),
        }
    }

    #[test]
    fn register_manual_records_status_with_manual_source() {
        let mut reg = AgentRegistry::new();
        reg.register_manual(
            Arc::new(MockAgent {
                name: "hand-wired".to_string(),
            }),
            PathBuf::from("/opt/local/bin/hand-wired"),
            Some("1.2.3".to_string()),
        );
        assert!(reg.contains("hand-wired"));
        match reg.agent_status("hand-wired") {
            Some(RegistryAgentStatus::Registered {
                source,
                binary,
                version,
            }) => {
                assert_eq!(*source, AgentSource::Manual);
                assert_eq!(binary, &PathBuf::from("/opt/local/bin/hand-wired"));
                assert_eq!(version.as_deref(), Some("1.2.3"));
            }
            other => panic!("expected Registered, got {other:?}"),
        }
    }

    #[test]
    fn register_subprocess_spec_honours_alias_and_workdir() {
        let mut reg = AgentRegistry::new();
        let spec = AgentDefinition {
            id: "canon".to_string(),
            alias: Some("alt".to_string()),
            binary_names: Vec::new(),
            binary: Some(PathBuf::from("/usr/local/bin/canon")),
            version_args: vec!["--version".to_string()],
            args: vec!["--task".to_string(), "{task_id}".to_string()],
            prompt_via_stdin: false,
            capabilities: AgentCapabilities {
                tags: vec!["plan".to_string()],
                languages: vec!["rust".to_string()],
                max_concurrency: 3,
                needs_network: false,
            },
            workdir: Some(PathBuf::from("/var/work")),
        };

        let binary = reg.register_subprocess_spec(&spec, AgentSource::Manual);
        assert_eq!(binary, PathBuf::from("/usr/local/bin/canon"));
        assert!(reg.contains("canon"), "canonical id must resolve");
        assert!(
            reg.contains("alt"),
            "alias registered inside spec must resolve to the same delegate"
        );
        match reg.agent_status("canon") {
            Some(RegistryAgentStatus::Registered { source, binary, .. }) => {
                assert_eq!(*source, AgentSource::Manual);
                assert_eq!(binary, &PathBuf::from("/usr/local/bin/canon"));
            }
            other => panic!("expected Registered, got {other:?}"),
        }
    }

    #[test]
    fn register_subprocess_spec_can_record_custom_source() {
        let mut reg = AgentRegistry::new();
        let spec = custom_def("via_overrides", "/opt/x");
        reg.register_subprocess_spec(&spec, AgentSource::Custom);
        match reg.agent_status("via_overrides") {
            Some(RegistryAgentStatus::Registered { source, .. }) => {
                assert_eq!(*source, AgentSource::Custom);
            }
            other => panic!("expected Registered, got {other:?}"),
        }
    }
}
