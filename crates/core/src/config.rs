//! Configuration management for Brain.
//!
//! Loads configuration from multiple sources with this priority (highest -> lowest):
//! 1. Environment variables (`BRAIN_` prefix, e.g. `BRAIN_LLM__MODEL`)
//! 2. User config file (`~/.brain/config.yaml`)
//! 3. Embedded defaults (compiled into the binary)

/// Default configuration embedded at compile time.
/// This means `brain` works anywhere without needing config files on disk.
const DEFAULT_CONFIG: &str = include_str!("../../../config/default.yaml");

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use figment::{
    providers::{Env, Format, Yaml},
    Figment,
};
use serde::{Deserialize, Serialize};

/// Top-level Brain configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    pub brain: GeneralConfig,
    pub storage: StorageConfig,
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub memory: MemoryConfig,
    pub encryption: EncryptionConfig,
    pub security: SecurityConfig,
    pub actions: ActionsConfig,
    pub proactivity: ProactivityConfig,
    pub adapters: AdaptersConfig,
    pub access: AccessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub version: String,
    pub data_dir: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub ruvector_path: String,
    pub sqlite_path: String,
    pub hnsw: HnswConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    pub ef_construction: u32,
    pub m: u32,
    pub ef_search: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub temperature: f64,
    pub max_tokens: u32,
    /// API key for the LLM provider (required for OpenAI, OpenRouter, etc.).
    /// Can also be set via `BRAIN_LLM__API_KEY` environment variable.
    #[serde(default)]
    pub api_key: String,
    /// Deprecated toggle: intent routing now uses LLM-first classification when
    /// an LLM provider is available. Kept for backwards-compatible config parsing.
    #[serde(default)]
    pub intent_llm_fallback: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model name (e.g. "nomic-embed-text" for Ollama,
    /// "text-embedding-3-small" for OpenAI). Must be available in
    /// the same service configured under `llm`.
    pub model: String,
    /// Output vector dimension — must exactly match the model's output size.
    /// Ollama nomic-embed-text → 768, OpenAI text-embedding-3-small → 1536.
    pub dimensions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub episodic: EpisodicConfig,
    pub semantic: SemanticConfig,
    pub search: SearchConfig,
    pub consolidation: ConsolidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicConfig {
    pub max_entries: u64,
    pub retention_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    pub similarity_threshold: f64,
    pub max_results: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub hybrid_weight: f64,
    pub rrf_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    pub enabled: bool,
    pub interval_hours: u32,
    pub forgetting_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub exec_allowlist: Vec<String>,
    pub exec_timeout_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionsConfig {
    pub web_search: WebSearchActionConfig,
    pub scheduling: SchedulingActionConfig,
    pub messaging: MessagingActionConfig,
    #[serde(default)]
    pub resilience: ResilienceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    pub max_retries: u32,
    pub retry_base_ms: u64,
    pub circuit_breaker_threshold: u32,
    pub circuit_breaker_cooldown_secs: u64,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_base_ms: 500,
            circuit_breaker_threshold: 5,
            circuit_breaker_cooldown_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchProvider {
    Searxng,
    Tavily,
    #[default]
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchActionConfig {
    pub enabled: bool,
    #[serde(default)]
    pub provider: WebSearchProvider,
    pub endpoint: String,
    #[serde(default)]
    pub api_key: String,
    pub timeout_ms: u64,
    pub default_top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingActionConfig {
    pub enabled: bool,
    pub mode: SchedulingMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SchedulingMode {
    PersistOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub url: String,
    #[serde(default)]
    pub body: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagingActionConfig {
    pub enabled: bool,
    pub timeout_ms: u64,
    #[serde(deserialize_with = "deserialize_channels", default)]
    pub channels: HashMap<String, ChannelConfig>,
}

/// Deserialize channels supporting both old format (string URL) and new format (ChannelConfig).
fn deserialize_channels<'de, D>(deserializer: D) -> Result<HashMap<String, ChannelConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ChannelEntry {
        Full(ChannelConfig),
        UrlOnly(String),
    }

    let raw: HashMap<String, ChannelEntry> = HashMap::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .map(|(k, v)| {
            let config = match v {
                ChannelEntry::Full(c) => c,
                ChannelEntry::UrlOnly(url) => ChannelConfig {
                    url,
                    body: String::new(),
                    headers: HashMap::new(),
                },
            };
            (k, config)
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactivityConfig {
    pub enabled: bool,
    pub max_per_day: u32,
    pub min_interval_minutes: u32,
    pub quiet_hours: QuietHoursConfig,
    #[serde(default)]
    pub delivery: DeliveryConfig,
    #[serde(default)]
    pub open_loop: OpenLoopDetectionConfig,
}

/// Configuration for open-loop (unresolved commitment) detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenLoopDetectionConfig {
    /// Enable open-loop detection.
    pub enabled: bool,
    /// How many hours back to scan for commitments.
    pub scan_window_hours: u32,
    /// Hours after a commitment before it's flagged as unresolved.
    pub resolution_window_hours: u32,
    /// Check interval in minutes.
    pub check_interval_minutes: u32,
}

impl Default for OpenLoopDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scan_window_hours: 72,
            resolution_window_hours: 24,
            check_interval_minutes: 120,
        }
    }
}

/// Configuration for proactive notification delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryConfig {
    /// Always write to outbox (drain on next interaction).
    pub outbox: bool,
    /// Push to live sessions via broadcast channel.
    pub broadcast: bool,
    /// Messaging channel keys (from actions.messaging.channels) to push proactive notifications.
    pub webhook_channels: Vec<String>,
    /// Maximum age (days) before undelivered outbox items are pruned.
    pub max_outbox_age_days: u32,
}

impl Default for DeliveryConfig {
    fn default() -> Self {
        Self {
            outbox: true,
            broadcast: true,
            webhook_channels: Vec::new(),
            max_outbox_age_days: 7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHoursConfig {
    pub start: String,
    pub end: String,
    #[serde(default = "default_timezone")]
    pub timezone: String,
}

fn default_timezone() -> String {
    "UTC".to_string()
}

/// A single API key entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyConfig {
    /// The raw API key string.
    pub key: String,
    /// Human-readable name for this key (for display/audit purposes).
    pub name: String,
    /// Granted permissions: `"read"` and/or `"write"`.
    pub permissions: Vec<String>,
}

impl ApiKeyConfig {
    /// Returns true if this key grants the requested permission.
    pub fn has_permission(&self, perm: &str) -> bool {
        self.permissions.iter().any(|p| p == perm)
    }
}

/// Access-control configuration (API keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessConfig {
    pub api_keys: Vec<ApiKeyConfig>,
}

impl AccessConfig {
    /// Returns true if `key` is valid and has the given `permission`.
    pub fn validate(&self, key: &str, permission: &str) -> bool {
        self.api_keys
            .iter()
            .any(|k| k.key == key && k.has_permission(permission))
    }

    /// Find a key entry by its raw key string.
    pub fn find_key(&self, key: &str) -> Option<&ApiKeyConfig> {
        self.api_keys.iter().find(|k| k.key == key)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptersConfig {
    pub http: HttpAdapterConfig,
    pub ws: WebSocketAdapterConfig,
    pub mcp: McpAdapterConfig,
    pub grpc: GrpcAdapterConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpAdapterConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub cors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAdapterConfig {
    pub enabled: bool,
    pub stdio: bool,
    pub http: bool,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

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
        let mut figment = Figment::new().merge(Yaml::string(DEFAULT_CONFIG));

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

        figment.extract()
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
            data_dir.join("db"),       // SQLite databases
            data_dir.join("ruvector"), // RuVector vector tables
            data_dir.join("models"),   // Reserved for future local models
            data_dir.join("logs"),     // Log files
            data_dir.join("exports"),  // Memory exports
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
    /// Returns the path written, or None if the file already exists
    /// and `force` is false.
    /// Write the default config to `~/.brain/config.yaml`.
    ///
    /// Returns `(config_path, generated_api_key)`, or `None` if the file already
    /// exists and `force` is false.
    pub fn write_default_config(force: bool) -> std::io::Result<Option<(PathBuf, String)>> {
        let config_path = Self::user_config_path();

        if config_path.exists() && !force {
            return Ok(None);
        }

        // Ensure parent directory exists
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Generate a random API key to replace the demo key
        let api_key = Self::generate_api_key();
        let config = DEFAULT_CONFIG.replace("demokey123", &api_key);

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
    pub fn user_config_path() -> PathBuf {
        expand_tilde("~/.brain/config.yaml")
    }

    /// Get the embedded default config content.
    pub fn default_config_content() -> &'static str {
        DEFAULT_CONFIG
    }

    /// Validate configuration and return a list of warnings.
    ///
    /// Returns `Err` for hard errors (invalid config that will prevent startup),
    /// and a `Vec<String>` of soft warnings for things that are unusual but
    /// won't prevent the process from running.
    pub fn validate(&self) -> Result<Vec<String>, String> {
        let mut warnings: Vec<String> = Vec::new();

        // ── Port conflict detection ───────────────────────────────────────────
        let mut ports: std::collections::HashMap<u16, &str> = std::collections::HashMap::new();
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

        // ── LLM URL format ────────────────────────────────────────────────────
        let url = &self.llm.base_url;
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(format!(
                "Invalid LLM base_url '{}': must start with http:// or https://",
                url
            ));
        }

        // ── Data directory writability ────────────────────────────────────────
        let data_dir = self.data_dir();
        if data_dir.exists() {
            // Check we can create a file inside it
            let probe = data_dir.join(".brain_write_probe");
            if std::fs::write(&probe, b"").is_err() {
                return Err(format!(
                    "Data directory '{}' is not writable",
                    data_dir.display()
                ));
            }
            let _ = std::fs::remove_file(&probe);
        }

        // ── Soft warnings ─────────────────────────────────────────────────────
        if self.access.api_keys.is_empty() {
            warnings.push("No API keys configured — all adapters will reject authenticated requests. Add at least one key under 'access.api_keys'.".to_string());
        } else if self.access.api_keys.iter().any(|k| k.key == "demokey123") {
            warnings.push("Demo API key 'demokey123' is still active. Replace it with a strong key in production.".to_string());
        }

        if self.llm.temperature > 1.5 {
            warnings.push(format!(
                "LLM temperature {:.1} is very high — responses may be unpredictable.",
                self.llm.temperature
            ));
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
                warnings.push("Actions messaging is enabled but actions.messaging.channels has no mappings; dispatches will fail for all channels.".to_string());
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

        // ── Timeout bounds ───────────────────────────────────────────────────
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

        // ── Resilience bounds ────────────────────────────────────────────────
        let res = &self.actions.resilience;
        if res.max_retries > 10 {
            warnings.push(format!("actions.resilience.max_retries is {} (>10) — excessive retries may amplify failures.", res.max_retries));
        }
        if res.circuit_breaker_threshold == 0 {
            warnings.push("actions.resilience.circuit_breaker_threshold is 0; circuit breaker will never trip.".to_string());
        }

        Ok(warnings)
    }
}

impl Default for BrainConfig {
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
                api_key: String::new(),
                intent_llm_fallback: false,
            },
            embedding: EmbeddingConfig {
                model: "nomic-embed-text".to_string(),
                dimensions: 768,
            },
            memory: MemoryConfig {
                episodic: EpisodicConfig {
                    max_entries: 100_000,
                    retention_days: 365,
                },
                semantic: SemanticConfig {
                    similarity_threshold: 0.65,
                    max_results: 20,
                },
                search: SearchConfig {
                    hybrid_weight: 0.7,
                    rrf_k: 60,
                },
                consolidation: ConsolidationConfig {
                    enabled: true,
                    interval_hours: 24,
                    forgetting_threshold: 0.05,
                },
            },
            encryption: EncryptionConfig { enabled: false }, // Deferred to v1.1
            security: SecurityConfig {
                exec_allowlist: vec![
                    "ls".into(),
                    "cat".into(),
                    "grep".into(),
                    "find".into(),
                    "git".into(),
                    "cargo".into(),
                    "rustc".into(),
                ],
                exec_timeout_seconds: 30,
            },
            actions: ActionsConfig {
                web_search: WebSearchActionConfig {
                    enabled: true,
                    provider: WebSearchProvider::Searxng,
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
                enabled: false,
                max_per_day: 5,
                min_interval_minutes: 60,
                quiet_hours: QuietHoursConfig {
                    start: "22:00".to_string(),
                    end: "08:00".to_string(),
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
                },
                ws: WebSocketAdapterConfig {
                    enabled: true,
                    port: 19790,
                },
                mcp: McpAdapterConfig {
                    enabled: true,
                    stdio: true,
                    http: true,
                    port: 19791,
                },
                grpc: GrpcAdapterConfig {
                    enabled: true,
                    port: 19792,
                },
            },
            access: AccessConfig {
                api_keys: vec![ApiKeyConfig {
                    key: "demokey123".to_string(),
                    name: "Demo Key".to_string(),
                    permissions: vec!["read".to_string(), "write".to_string()],
                }],
            },
        }
    }
}

/// Expand `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs_home() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

/// Get the user's home directory.
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BrainConfig::default();
        assert_eq!(config.brain.data_dir, "~/.brain");
        assert_eq!(config.llm.provider, "ollama");
        assert_eq!(config.embedding.dimensions, 768); // nomic-embed-text default
        assert!(!config.encryption.enabled); // Deferred to v1.1
        assert_eq!(
            config.actions.web_search.provider,
            WebSearchProvider::Searxng
        );
        assert_eq!(config.actions.scheduling.mode, SchedulingMode::PersistOnly);
        assert!(!config.proactivity.enabled);
        assert!(config.adapters.http.enabled);
    }

    #[test]
    fn test_expand_tilde() {
        let expanded = expand_tilde("~/.brain");
        assert!(!expanded.to_str().unwrap().starts_with('~'));
        assert!(expanded.to_str().unwrap().ends_with(".brain"));
    }

    #[test]
    fn test_data_dir_paths() {
        let config = BrainConfig::default();
        let data = config.data_dir();
        assert!(data.to_str().unwrap().ends_with(".brain"));
        assert!(config.sqlite_path().to_str().unwrap().ends_with("brain.db"));
        assert!(config
            .ruvector_path()
            .to_str()
            .unwrap()
            .ends_with("ruvector"));
    }

    #[test]
    fn test_load_from_defaults() {
        use figment::providers::Serialized;
        // Load using Serialized defaults (no file needed)
        let figment = Figment::new().merge(Serialized::defaults(BrainConfig::default()));
        let config: BrainConfig = figment.extract().unwrap();
        assert_eq!(config.llm.model, "qwen2.5-coder:7b");
        assert_eq!(config.memory.search.rrf_k, 60);
    }

    // ── validate() ────────────────────────────────────────────────────────────

    /// Helper: default config with no API keys (to keep warnings deterministic).
    fn writable_test_data_dir() -> String {
        std::env::temp_dir()
            .join("brain-core-tests")
            .to_string_lossy()
            .to_string()
    }

    /// Helper: default config with no API keys (to keep warnings deterministic).
    fn validated_config() -> BrainConfig {
        let mut c = BrainConfig::default();
        c.brain.data_dir = writable_test_data_dir();
        c.access.api_keys.clear();
        c
    }

    #[test]
    fn test_validate_default_has_demo_key_warning() {
        let mut config = BrainConfig::default();
        config.brain.data_dir = writable_test_data_dir();
        let warnings = config.validate().expect("default config should be valid");
        assert!(
            warnings.iter().any(|w| w.contains("demokey123")),
            "expected demo-key warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_no_api_keys_warning() {
        let config = validated_config();
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings.iter().any(|w| w.contains("No API keys")),
            "expected no-api-keys warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_port_conflict_is_hard_error() {
        let mut config = validated_config();
        // Make HTTP and WS share the same port
        config.adapters.ws.port = config.adapters.http.port;
        let err = config
            .validate()
            .expect_err("should fail with port conflict");
        assert!(
            err.contains("Port conflict"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_validate_bad_llm_url_is_hard_error() {
        let mut config = validated_config();
        config.llm.base_url = "ftp://invalid.example.com".to_string();
        let err = config.validate().expect_err("should fail with bad URL");
        assert!(
            err.contains("Invalid LLM base_url"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_validate_high_temperature_warning() {
        let mut config = validated_config();
        config.llm.temperature = 2.0;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings.iter().any(|w| w.contains("temperature")),
            "expected temperature warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_consolidation_interval_zero_warning() {
        let mut config = validated_config();
        config.memory.consolidation.enabled = true;
        config.memory.consolidation.interval_hours = 0;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings.iter().any(|w| w.contains("interval_hours")),
            "expected interval warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_actions_defaults_deserialize() {
        let config = BrainConfig::load().expect("embedded defaults should load");
        assert!(config.actions.web_search.enabled);
        assert_eq!(
            config.actions.web_search.provider,
            WebSearchProvider::Searxng
        );
        assert_eq!(config.actions.web_search.default_top_k, 5);
        assert_eq!(config.actions.scheduling.mode, SchedulingMode::PersistOnly);
        assert!(!config.actions.messaging.enabled);
    }

    #[test]
    fn test_validate_actions_warning_custom_without_endpoint() {
        let mut config = validated_config();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = WebSearchProvider::Custom;
        config.actions.web_search.endpoint.clear();
        config.actions.messaging.enabled = true;
        config.actions.messaging.channels.clear();
        let warnings = config.validate().expect("config should still be valid");
        assert!(warnings.iter().any(|w| w.contains("'custom'")));
        assert!(warnings.iter().any(|w| w.contains("messaging")));
    }

    #[test]
    fn test_validate_tavily_without_api_key_warning() {
        let mut config = validated_config();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = WebSearchProvider::Tavily;
        config.actions.web_search.api_key.clear();
        let warnings = config.validate().expect("config should still be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("'tavily'") && w.contains("api_key")),
            "expected tavily api_key warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_searxng_no_web_search_warning() {
        let mut config = validated_config();
        config.actions.web_search.enabled = true;
        config.actions.web_search.provider = WebSearchProvider::Searxng;
        let warnings = config.validate().expect("config should still be valid");
        assert!(
            !warnings.iter().any(|w| w.contains("web_search")),
            "SearXNG with default endpoint should not trigger web_search warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_http_and_https_urls_accepted() {
        let mut config = validated_config();
        config.llm.base_url = "https://api.example.com/v1".to_string();
        assert!(config.validate().is_ok());

        config.llm.base_url = "http://localhost:11434".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_all_unique_ports_ok() {
        let config = validated_config();
        // Default config has unique ports — should not error
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_timeout_zero_warning() {
        let mut config = validated_config();
        config.actions.web_search.timeout_ms = 0;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("timeout_ms") && w.contains("0")),
            "expected timeout_ms=0 warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_timeout_too_high_warning() {
        let mut config = validated_config();
        config.actions.messaging.timeout_ms = 60_000;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("timeout_ms") && w.contains("60000")),
            "expected high timeout warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_resilience_max_retries_warning() {
        let mut config = validated_config();
        config.actions.resilience.max_retries = 15;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("max_retries") && w.contains("15")),
            "expected max_retries warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_validate_resilience_threshold_zero_warning() {
        let mut config = validated_config();
        config.actions.resilience.circuit_breaker_threshold = 0;
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("circuit_breaker_threshold")),
            "expected circuit_breaker_threshold=0 warning, got: {:?}",
            warnings
        );
    }

    #[test]
    fn test_resilience_defaults() {
        let res = ResilienceConfig::default();
        assert_eq!(res.max_retries, 2);
        assert_eq!(res.retry_base_ms, 500);
        assert_eq!(res.circuit_breaker_threshold, 5);
        assert_eq!(res.circuit_breaker_cooldown_secs, 60);
    }

    #[test]
    fn test_channel_config_old_format_compat() {
        // Old format: channels map string → string (URL only)
        let yaml = r#"
            enabled: false
            timeout_ms: 3000
            channels:
              alerts: "https://example.com/hook"
              ops: "https://slack.example.com/webhook"
        "#;
        let cfg: MessagingActionConfig =
            serde_yaml::from_str(yaml).expect("old format should deserialize");
        assert_eq!(cfg.channels.len(), 2);
        assert_eq!(cfg.channels["alerts"].url, "https://example.com/hook");
        assert!(cfg.channels["alerts"].body.is_empty());
        assert!(cfg.channels["alerts"].headers.is_empty());
    }

    #[test]
    fn test_channel_config_new_format() {
        let yaml = r#"
            enabled: true
            timeout_ms: 3000
            channels:
              alerts:
                url: "https://hooks.slack.com/services/T/B/x"
                body: '{"text": "{{content}}"}'
                headers:
                  Authorization: "Bearer tok123"
        "#;
        let cfg: MessagingActionConfig =
            serde_yaml::from_str(yaml).expect("new format should deserialize");
        assert_eq!(cfg.channels.len(), 1);
        let ch = &cfg.channels["alerts"];
        assert_eq!(ch.url, "https://hooks.slack.com/services/T/B/x");
        assert_eq!(ch.body, r#"{"text": "{{content}}"}"#);
        assert_eq!(ch.headers["Authorization"], "Bearer tok123");
    }

    #[test]
    fn test_channel_config_mixed_format() {
        let yaml = r#"
            enabled: true
            timeout_ms: 3000
            channels:
              simple: "https://example.com/hook"
              custom:
                url: "https://discord.com/api/webhooks/123/abc"
                body: '{"content": "{{content}}"}'
        "#;
        let cfg: MessagingActionConfig =
            serde_yaml::from_str(yaml).expect("mixed format should deserialize");
        assert_eq!(cfg.channels.len(), 2);
        assert_eq!(cfg.channels["simple"].url, "https://example.com/hook");
        assert!(cfg.channels["simple"].body.is_empty());
        let custom = &cfg.channels["custom"];
        assert_eq!(custom.url, "https://discord.com/api/webhooks/123/abc");
        assert!(!custom.body.is_empty());
        assert!(custom.headers.is_empty());
    }

    #[test]
    fn test_validate_channel_empty_url_warning() {
        let mut config = validated_config();
        config.actions.messaging.enabled = true;
        config.actions.messaging.channels.insert(
            "bad".into(),
            ChannelConfig {
                url: "".into(),
                body: String::new(),
                headers: HashMap::new(),
            },
        );
        let warnings = config.validate().expect("should be valid");
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("channels.bad") && w.contains("url is empty")),
            "expected empty-url warning, got: {:?}",
            warnings
        );
    }
}
