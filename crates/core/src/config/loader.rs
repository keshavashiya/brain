use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use figment::{
    providers::{Env, Format, Yaml},
    Figment,
};

use super::*;

impl BrainConfig {
    /// Load configuration from all sources.
    ///
    /// Priority (highest wins):
    /// 1. Environment variables (`BRAIN_LLM__MODEL=...`)
    /// 2. User config (`~/.brain/config.yaml`)
    /// 3. Embedded defaults (compiled into binary)
    #[allow(clippy::result_large_err)]
    pub fn load() -> Result<Self, figment::Error> {
        Self::load_from(None)
    }

    /// Load configuration with an optional explicit config path.
    #[allow(clippy::result_large_err)]
    pub fn load_from(config_path: Option<&Path>) -> Result<Self, figment::Error> {
        // Layer 1: Embedded defaults (always available, no file needed)
        let mut figment = Figment::new().merge(Yaml::string(super::DEFAULT_CONFIG));

        // Layer 2: User config (~/.brain/config.yaml)
        let user_config = Self::user_config_path();
        if user_config.exists() {
            figment = figment.merge(Yaml::file(&user_config));
        }

        // Layer 3: Explicit config path (if provided)
        if let Some(path) = config_path {
            figment = figment.merge(Yaml::file(path));
        }

        // Layer 4: Environment variables (BRAIN_LLM__MODEL=...)
        figment = figment.merge(Env::prefixed("BRAIN_").split("__"));

        let mut cfg: Self = figment.extract()?;

        // Post-load: if the user set legacy llm.{base_url,model,api_key} via
        // env vars but `providers[]` is non-empty, the multi-provider path
        // would silently ignore the env override. Forward those overrides
        // onto providers[0] so `BRAIN_LLM__BASE_URL=...` does what users
        // expect regardless of how the YAML is structured.
        if !cfg.llm.providers.is_empty() {
            #[allow(deprecated)]
            {
                if std::env::var("BRAIN_LLM__BASE_URL").is_ok() {
                    cfg.llm.providers[0].base_url = cfg.llm.base_url.clone();
                }
                if std::env::var("BRAIN_LLM__MODEL").is_ok() {
                    cfg.llm.providers[0].model = cfg.llm.model.clone();
                }
                if std::env::var("BRAIN_LLM__API_KEY").is_ok() {
                    cfg.llm.providers[0].api_key = cfg.llm.api_key.clone();
                }
            }
        }

        Ok(cfg)
    }

    /// Resolve the data directory path, expanding `~` to the home directory.
    pub fn data_dir(&self) -> PathBuf {
        expand_tilde(&self.brain.data_dir)
    }

    /// Ensure the data directory and subdirectories exist.
    pub fn ensure_data_dirs(&self) -> std::io::Result<()> {
        let data_dir = self.data_dir();
        let dirs = [
            data_dir.clone(),
            data_dir.join("db"),
            data_dir.join("ruvector"),
            data_dir.join("models"),
            data_dir.join("logs"),
            data_dir.join("exports"),
        ];

        for dir in &dirs {
            std::fs::create_dir_all(dir)?;
        }

        Ok(())
    }

    /// Path to the SQLite database file.
    pub fn sqlite_path(&self) -> PathBuf {
        self.data_dir().join("db").join("brain.db")
    }

    /// Path to the RuVector directory.
    pub fn ruvector_path(&self) -> PathBuf {
        self.data_dir().join("ruvector")
    }

    /// Path to the models directory.
    pub fn models_path(&self) -> PathBuf {
        self.data_dir().join("models")
    }

    /// Check whether Brain has been initialized (data dir exists).
    pub fn is_initialized() -> bool {
        expand_tilde("~/.brain").exists()
    }

    /// Write the default config to `~/.brain/config.yaml`.
    ///
    /// Returns `(config_path, generated_api_key)`, or `None` if the file already
    /// exists and `force` is false.
    pub fn write_default_config(force: bool) -> std::io::Result<Option<(PathBuf, String)>> {
        let config_path = Self::user_config_path();

        if config_path.exists() && !force {
            return Ok(None);
        }

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let api_key = Self::generate_api_key();
        let config = super::DEFAULT_CONFIG.replace(
            "api_keys: []",
            &format!(
                "api_keys:\n    - key: \"{}\"\n      name: \"Default Key\"\n      permissions: [read, write]",
                api_key
            ),
        );

        std::fs::write(&config_path, config)?;
        Ok(Some((config_path, api_key)))
    }

    /// Generate a random 36-char API key with `brk_` prefix.
    fn generate_api_key() -> String {
        let mut buf = [0u8; 16];
        getrandom::getrandom(&mut buf).expect("failed to obtain random bytes from OS");
        let hex: String = buf.iter().map(|b| format!("{:02x}", b)).collect();
        format!("brk_{}", hex)
    }

    /// Path to user config file.
    ///
    /// `BRAIN_CONFIG` env var overrides the default `~/.brain/config.yaml`,
    /// useful for sandboxes, CI, and multi-config workflows.
    pub fn user_config_path() -> PathBuf {
        if let Ok(p) = std::env::var("BRAIN_CONFIG") {
            if !p.trim().is_empty() {
                return PathBuf::from(p);
            }
        }
        expand_tilde("~/.brain/config.yaml")
    }

    /// Get the embedded default config content.
    pub fn default_config_content() -> &'static str {
        super::DEFAULT_CONFIG
    }

    /// Validate configuration and return a list of warnings.
    pub fn validate(&self) -> Result<Vec<String>, String> {
        let mut warnings: Vec<String> = Vec::new();

        let mut ports: HashMap<u16, &str> = HashMap::new();
        let adapter_ports = [
            (self.adapters.http.port, "http"),
            (self.adapters.ws.port, "ws"),
            (self.adapters.mcp.port, "mcp"),
            (self.adapters.grpc.port, "grpc"),
        ];
        for (port, name) in &adapter_ports {
            if let Some(existing) = ports.insert(*port, name) {
                return Err(format!(
                    "Port conflict: adapters '{}' and '{}' both use port {}",
                    existing, name, port
                ));
            }
        }

        #[allow(deprecated)]
        let url = &self.llm.base_url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(format!(
                "Invalid LLM base_url '{}': must start with http:// or https://",
                url
            ));
        }

        let data_dir = self.data_dir();
        if data_dir.exists() {
            let probe = data_dir.join(".brain_write_probe");
            if std::fs::write(&probe, b"").is_err() {
                return Err(format!(
                    "Data directory '{}' is not writable",
                    data_dir.display()
                ));
            }
            let _ = std::fs::remove_file(&probe);
        }

        if self.access.api_keys.is_empty() {
            return Err("No API keys configured. Run `brain init` to generate a config with a secure API key, or configure 'access.api_keys' manually.".to_string());
        }

        if self.llm.temperature > 1.5 {
            warnings.push(format!(
                "LLM temperature {:.1} is very high — responses may be unpredictable.",
                self.llm.temperature
            ));
        }

        // Issue 40: `llm.provider` is #[deprecated]. When `providers[]`
        // is non-empty AND the legacy field is also set, the loader
        // backfills providers[0] from the legacy shape — but operators
        // typically don't realise the legacy field is no longer
        // authoritative for LLM routing. Surface the overlap loudly so
        // they migrate to `providers[]`.
        #[allow(deprecated)]
        let legacy_provider_set = !self.llm.provider.trim().is_empty();
        if !self.llm.providers.is_empty() && legacy_provider_set {
            warnings.push(
                "Legacy `llm.provider` is set alongside `llm.providers[]`. \
                 `llm.providers[]` is the authoritative routing surface; \
                 the legacy field is only kept for embedder selection \
                 and will be retired in a future release."
                    .to_string(),
            );
        }

        // Tier provider names must resolve to a `providers[]` entry. An
        // unmatched name fails closed at daemon startup (a tier meant to stay
        // local must never silently reroute onto the default chain), so reject
        // it here too rather than green-lighting a config that won't boot.
        if !self.llm.providers.is_empty() {
            for (tier, names) in [
                ("fast", &self.llm.tiers.fast),
                ("balanced", &self.llm.tiers.balanced),
                ("deep", &self.llm.tiers.deep),
            ] {
                for name in names {
                    if !self.llm.providers.iter().any(|p| &p.name == name) {
                        return Err(format!(
                            "llm.tiers.{tier} references unknown provider '{name}'; \
                             it must match an `llm.providers[].name` entry \
                             (this fails closed at daemon startup)."
                        ));
                    }
                }
            }
        }

        if self.memory.consolidation.enabled && self.memory.consolidation.interval_hours == 0 {
            warnings.push("Consolidation interval_hours is 0 — consolidation will run immediately on every daemon wake-up, which may impact performance.".to_string());
        }

        if self.actions.web_search.enabled {
            match self.actions.web_search.provider {
                WebSearchProvider::Custom if self.actions.web_search.endpoint.trim().is_empty() => {
                    warnings.push("Actions web_search provider is 'custom' but endpoint is empty; dispatches will fail with backend-not-configured.".to_string());
                }
                WebSearchProvider::Tavily if self.actions.web_search.api_key.trim().is_empty() => {
                    warnings.push("Actions web_search provider is 'tavily' but api_key is empty; dispatches will fail.".to_string());
                }
                _ => {}
            }
        }

        if self.actions.messaging.enabled {
            if self.actions.messaging.channels.is_empty() {
                if self.channel.transports.is_empty() && self.channel.relays.is_empty() {
                    warnings.push("Actions messaging is enabled but neither actions.messaging.channels, channel.transports, nor channel.relays are configured; dispatches will fail.".to_string());
                }
            } else {
                for (name, channel_cfg) in &self.actions.messaging.channels {
                    if channel_cfg.url.trim().is_empty() {
                        warnings.push(format!(
                            "actions.messaging.channels.{name}: url is empty; dispatches to this channel will fail."
                        ));
                    }
                }
            }
        }

        for (name, ms) in [
            ("web_search.timeout_ms", self.actions.web_search.timeout_ms),
            ("messaging.timeout_ms", self.actions.messaging.timeout_ms),
        ] {
            if ms == 0 {
                warnings.push(format!(
                    "actions.{name} is 0; will be clamped to 1ms at runtime."
                ));
            } else if ms > 30_000 {
                warnings.push(format!(
                    "actions.{name} is {}ms (>30s) — requests may block for a long time.",
                    ms
                ));
            }
        }

        let mut service_names: HashMap<&str, ()> = HashMap::new();
        for svc in &self.monitoring.services {
            if svc.name.trim().is_empty() {
                warnings.push(
                    "monitoring.services has an entry with an empty name; its alerts will be hard to attribute.".to_string(),
                );
            } else if service_names.insert(svc.name.as_str(), ()).is_some() {
                warnings.push(format!(
                    "monitoring.services has a duplicate name '{}'; both probe loops still run but their alerts are indistinguishable.",
                    svc.name
                ));
            }
            if svc.target.trim().is_empty() {
                warnings.push(format!(
                    "monitoring.services.{}: target is empty; this probe will always report the service as down.",
                    svc.name
                ));
            } else if matches!(svc.kind, ServiceCheckKind::Http)
                && !svc.target.starts_with("http://")
                && !svc.target.starts_with("https://")
            {
                warnings.push(format!(
                    "monitoring.services.{}: kind is 'http' but target '{}' is not an http(s) URL; probes will fail.",
                    svc.name, svc.target
                ));
            } else if matches!(svc.kind, ServiceCheckKind::Tcp) && !svc.target.contains(':') {
                warnings.push(format!(
                    "monitoring.services.{}: kind is 'tcp' but target '{}' is not 'host:port'; probes will fail.",
                    svc.name, svc.target
                ));
            }
            if svc.interval_secs == 0 {
                warnings.push(format!(
                    "monitoring.services.{}: interval_secs is 0 — it will be clamped to 1s at runtime, probing in a tight loop.",
                    svc.name
                ));
            }
        }

        let res = &self.actions.resilience;
        if res.max_retries > 10 {
            warnings.push(format!("actions.resilience.max_retries is {} (>10) — excessive retries may amplify failures.", res.max_retries));
        }
        if res.circuit_breaker_threshold == 0 {
            warnings.push("actions.resilience.circuit_breaker_threshold is 0; circuit breaker will never trip.".to_string());
        }

        // `actions.scheduling` is the *write* axis (create/persist scheduled
        // intents); `reflex.cron` is the *fire* axis. Enabling the former
        // without the latter persists intents that never fire — a silent
        // black hole. `brain doctor` flags this too, but a user who only ever
        // runs `brain start` would otherwise never hear about it.
        if self.actions.scheduling.enabled && !self.reflex.cron.enabled {
            warnings.push(
                "actions.scheduling is enabled but reflex.cron is disabled — scheduled \
                 intents will persist but never fire. Set reflex.cron.enabled = true."
                    .to_string(),
            );
        }

        Ok(warnings)
    }
}

impl Default for BrainConfig {
    // `llm.provider` is `#[deprecated]` (Issue 40) but still load-bearing as
    // the implicit single entry when `providers[]` is empty — populate it.
    #[allow(deprecated)]
    fn default() -> Self {
        Self {
            brain: GeneralConfig {
                version: env!("CARGO_PKG_VERSION").to_string(),
                data_dir: "~/.brain".to_string(),
            },
            storage: StorageConfig {
                ruvector_path: "~/.brain/ruvector/".to_string(),
                sqlite_path: "~/.brain/db/brain.db".to_string(),
                hnsw: HnswConfig {
                    ef_construction: 200,
                    max_elements: HnswConfig::default_max_elements(),
                    m: 16,
                    ef_search: 50,
                },
            },
            llm: LlmConfig {
                provider: "ollama".to_string(),
                model: "qwen2.5-coder:7b".to_string(),
                base_url: "http://localhost:11434".to_string(),
                temperature: 0.7,
                max_tokens: 4096,
                context_window: super::default_context_window(),
                api_key: String::new(),
                api_key_file: None,
                providers: Vec::new(),
                tiers: LlmTiersConfig::default(),
            },
            embedding: EmbeddingConfig {
                model: "nomic-embed-text".to_string(),
                dimensions: 768,
            },
            memory: MemoryConfig {
                semantic: SemanticConfig {
                    similarity_threshold: 0.65,
                    max_results: 20,
                },
                search: SearchConfig {
                    rrf_k: 60,
                    pre_fusion_limit: 50,
                    importance_weight: 0.3,
                    recency_weight: 0.2,
                    decay_rate: 0.01,
                },
                consolidation: ConsolidationConfig {
                    enabled: true,
                    interval_hours: 24,
                    forgetting_threshold: 0.05,
                },
                namespaces: Default::default(),
                trust: Default::default(),
            },
            encryption: EncryptionConfig { enabled: false },
            security: SecurityConfig {
                exec_allowlist: vec![
                    // Read-only inspection
                    "ls".into(),
                    "cat".into(),
                    "head".into(),
                    "tail".into(),
                    "wc".into(),
                    "file".into(),
                    "stat".into(),
                    // Text processing
                    "grep".into(),
                    "find".into(),
                    "sort".into(),
                    "uniq".into(),
                    "cut".into(),
                    "awk".into(),
                    "sed".into(),
                    // Shell discovery / pathing
                    "which".into(),
                    "command".into(),
                    "type".into(),
                    "test".into(),
                    "basename".into(),
                    "dirname".into(),
                    "realpath".into(),
                    // Output
                    "echo".into(),
                    "printf".into(),
                    "true".into(),
                    "false".into(),
                    // Toolchain
                    "git".into(),
                    "cargo".into(),
                    "rustc".into(),
                    "rustup".into(),
                    // Shell wrapper for the shell-mode execution tier
                    // (see SandboxCommand::shell). The per-binary
                    // allowlist is bypassed for commands wrapped by
                    // `sh -c` — rlimits + Seatbelt + timeout +
                    // forbidden_commands still apply.
                    "sh".into(),
                ],
                exec_timeout_seconds: 30,
                // Empty → handler defaults to `$HOME`. Configure
                // `security.allowed_paths: ["~/code", "~/work"]` to
                // restrict further.
                allowed_paths: Vec::new(),
            },
            actions: ActionsConfig {
                web_search: WebSearchActionConfig {
                    // On by default. DuckDuckGo HTML scraping is the
                    // zero-config built-in so first-run has working web
                    // search without Docker or an API key.
                    enabled: true,
                    provider: WebSearchProvider::DuckDuckGo,
                    endpoint: "http://localhost:8888".to_string(),
                    api_key: String::new(),
                    timeout_ms: 3_000,
                    default_top_k: 5,
                },
                scheduling: SchedulingActionConfig {
                    enabled: false,
                    mode: SchedulingMode::PersistOnly,
                },
                messaging: MessagingActionConfig {
                    enabled: false,
                    timeout_ms: 3_000,
                    channels: HashMap::new(),
                },
                resilience: ResilienceConfig::default(),
            },
            proactivity: ProactivityConfig {
                // Synced to `default.yaml`: programmatic `BrainConfig::default()`
                // must match the embedded YAML so the loader and the struct
                // produce the same shape (Issue 36).
                enabled: true,
                max_per_day: 2,
                min_interval_minutes: 60,
                quiet_hours: QuietHoursConfig {
                    start: "20:00".to_string(),
                    end: "10:00".to_string(),
                    timezone: "UTC".to_string(),
                },
                delivery: DeliveryConfig::default(),
                open_loop: OpenLoopDetectionConfig::default(),
            },
            adapters: AdaptersConfig {
                http: HttpAdapterConfig {
                    enabled: true,
                    host: "127.0.0.1".to_string(),
                    port: 19789,
                    cors: true,
                    sse_redact_previews: false,
                },
                ws: WebSocketAdapterConfig {
                    enabled: true,
                    port: 19790,
                },
                mcp: McpAdapterConfig {
                    enabled: true,
                    port: 19791,
                },
                grpc: GrpcAdapterConfig {
                    enabled: true,
                    port: 19792,
                },
                terminal: TerminalAdapterConfig::default_enabled(),
            },
            access: AccessConfig {
                api_keys: vec![ApiKeyConfig {
                    key: Self::generate_api_key(),
                    name: "Default Key".to_string(),
                    permissions: vec!["read".to_string(), "write".to_string()],
                    agent_id: None,
                }],
                rate_limit: ClientRateLimitConfig::default(),
            },
            channel: ChannelIntelligenceConfig::default(),
            agents: AgentsConfig::default(),
            confirm: ConfirmConfig::default(),
            identity: identity::IdentityConfig::default(),
            reflex: ReflexConfig::default(),
            logging: crate::config::LoggingConfig::default(),
            learning: crate::config::LearningConfig::default(),
            observability: crate::config::ObservabilityConfig::default(),
            monitoring: crate::config::MonitoringConfig::default(),
        }
    }
}

pub(crate) fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
