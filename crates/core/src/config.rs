//! Configuration management for Brain.
//!
//! Loads configuration from multiple sources with this priority (highest -> lowest):
//! 1. Environment variables (`BRAIN_` prefix, e.g. `BRAIN_LLM__MODEL`)
//! 2. User config file (`~/.brain/config.yaml`)
//! 3. Embedded defaults (compiled into the binary)

/// Default configuration embedded at compile time.
/// This means `brain` works anywhere without needing config files on disk.
const DEFAULT_CONFIG: &str = include_str!("../default.yaml");

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    #[serde(default)]
    pub channel: ChannelIntelligenceConfig,
    #[serde(default)]
    pub agents: AgentsConfig,
    #[serde(default)]
    pub confirm: ConfirmConfig,
    /// Principal & identity configuration consumed by
    /// `identity::ConfigIdentityStore`. Default is empty — signals carry
    /// `Principal = None` and the identity gate is silently skipped.
    #[serde(default)]
    pub identity: identity::IdentityConfig,
    /// Reactive signal sources. Each subsection drives one reflex type;
    /// default is empty/disabled across the board, so a fresh install
    /// spawns no reflex tasks. `cmd_serve` reads this to construct
    /// `FsReflex` / `CronReflex` / `SysStateReflex` and bridge their
    /// streams into the pipeline via `signal::spawn_reflex`.
    #[serde(default)]
    pub reflex: ReflexConfig,
}

/// Top-level reactive-source configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReflexConfig {
    /// Filesystem watchers. One entry per `FsReflex` source. Empty list
    /// means no FS reflex is spawned.
    #[serde(default)]
    pub fs: Vec<FsReflexEntry>,
    /// Cron-style reflex bridging the scheduler. Disabled by default.
    #[serde(default)]
    pub cron: CronReflexEntry,
    /// System-state edge-trigger reflex. Disabled by default. Uses a
    /// noop sampler until a per-platform implementation is wired.
    #[serde(default)]
    pub sys: SysReflexEntry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FsReflexEntry {
    /// Stable name used in tracing + as the reflex's `name()`. Also
    /// embedded in the resulting `Provenance::Reflex { trigger }`.
    pub name: String,
    /// Filesystem paths to watch (absolute or `~`-relative).
    pub paths: Vec<String>,
    #[serde(default)]
    pub recursive: bool,
    /// Debounce window in milliseconds. Default 200ms when omitted.
    #[serde(default = "FsReflexEntry::default_debounce_ms")]
    pub debounce_ms: u64,
}

impl FsReflexEntry {
    pub fn default_debounce_ms() -> u64 {
        200
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CronReflexEntry {
    #[serde(default)]
    pub enabled: bool,
    /// Poll interval in seconds. Default 60s when omitted (matches the
    /// historical `cli::serve` ticker).
    #[serde(default = "CronReflexEntry::default_poll_seconds")]
    pub poll_interval_seconds: u64,
    /// Optional namespace filter — only intents in this namespace fire.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub namespace_filter: Option<String>,
}

impl CronReflexEntry {
    pub fn default_poll_seconds() -> u64 {
        60
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SysReflexEntry {
    #[serde(default)]
    pub enabled: bool,
    /// Sampler poll cadence in seconds. Default 30s.
    #[serde(default = "SysReflexEntry::default_poll_seconds")]
    pub poll_interval_seconds: u64,
    /// Edge-triggered rules to evaluate on each transition.
    #[serde(default)]
    pub rules: Vec<SysReflexRuleEntry>,
}

impl SysReflexEntry {
    pub fn default_poll_seconds() -> u64 {
        30
    }
}

/// YAML-bound mirror of `reflex::SysStateRule`. Kept here so `brain`
/// doesn't take a dependency on `reflex` (which depends on `brain`
/// transitively); `cmd_serve` converts each entry to a concrete
/// `SysStateRule` at spawn time.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SysReflexRuleEntry {
    /// Fires when battery percentage crosses below `threshold`.
    BatteryBelow { threshold: u8 },
    /// Fires when `on_ac` flips in either direction.
    OnAcChanged,
    /// Fires when network reachability flips between online and offline.
    NetworkChanged,
    /// Fires when session lock state flips.
    LockChanged,
}

/// Confirmation-engine configuration. Currently only declares standing
/// approvals — pre-granted (agent, verb) consent that bypasses the
/// human-confirm prompt. Empty defaults preserve pre-Phase-5 behavior.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfirmConfig {
    #[serde(default)]
    pub standing_approvals: Vec<StandingApprovalDecl>,
}

/// One standing-approval declaration. Loaded at startup into the
/// `StandingApprovalStore`; idempotent across launches (an existing
/// active grant for the same triple is left alone).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandingApprovalDecl {
    pub agent_id: String,
    pub verb_ns: String,
    pub verb_action: String,
    #[serde(default)]
    pub note: Option<String>,
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
    /// Maximum number of vectors a single HNSW table can hold. Threaded
    /// into the underlying ruvector database at `open` time (Issue 37);
    /// previously hardcoded at 10_000_000 inside the storage crate.
    ///
    /// HNSW pre-allocates the index graph for `max_elements` entries
    /// up-front, so this knob is a real memory cost — not just an
    /// upper bound. Personal-scale installs rarely need more than
    /// 100k facts/episodes; production / shared installs that need
    /// more should raise this explicitly in their config rather than
    /// pay for the headroom in every dev install. (Wave F, Issue 71.)
    #[serde(default = "HnswConfig::default_max_elements")]
    pub max_elements: u32,
}

impl HnswConfig {
    /// 100k vectors — covers the vast majority of personal-scale
    /// deployments without pre-allocating headroom for a million users
    /// of facts that nobody will ever store. Raise via
    /// `storage.hnsw.max_elements` when you actually need it.
    pub fn default_max_elements() -> u32 {
        100_000
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Legacy single-provider selector. Superseded by `providers[]`,
    /// which supports multi-provider failover and runtime health
    /// checks. Still honoured as the implicit single entry when
    /// `providers[]` is empty, and `Embedder::from_config` reads it
    /// to pick the embedding transport — so it can't be removed yet.
    /// New configs should leave this set to a reasonable default and
    /// drive everything from `providers[]` instead.
    #[deprecated(
        note = "Set `llm.providers[]` instead. Single-provider mode is still functional but no longer the recommended shape."
    )]
    pub provider: String,
    pub model: String,
    pub base_url: String,
    pub temperature: f64,
    pub max_tokens: u32,
    /// API key for the LLM provider (required for OpenAI, OpenRouter, etc.).
    /// Can also be set via `BRAIN_LLM__API_KEY` environment variable.
    #[serde(default)]
    pub api_key: String,
    /// Issue 125: path to a chmod-0600 file holding the API key. Preferred
    /// over `api_key` because the YAML config typically gets backed up,
    /// version-controlled, and replicated; a sibling file with restricted
    /// perms keeps the secret out of those flows. When both are set,
    /// `api_key_file` wins.
    #[serde(default)]
    pub api_key_file: Option<std::path::PathBuf>,
    /// Optional multi-provider entries. When non-empty, startup probes each
    /// entry's `/models` endpoint and selects the first reachable one whose
    /// `preferred_models` are live. When empty, the legacy single-provider
    /// fields above are used as-is.
    #[serde(default)]
    pub providers: Vec<ProviderEntry>,
}

/// One entry in `llm.providers` — a named destination that the cortex
/// will probe at startup. Only two transport kinds are recognised:
/// `ollama` (local) and `openai_compat` (any OpenAI-compatible endpoint).
/// A preset name (`groq`, `openrouter`, `deepseek`, `together`,
/// `gemini-compat`, `openai`) is also accepted as shorthand for
/// `openai_compat` with a prefilled `base_url`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEntry {
    /// Human-readable identifier (`"primary"`, `"groq-free"`, …).
    pub name: String,
    /// Transport kind or preset name.
    pub kind: String,
    /// Override the preset's base_url; required when `kind` is
    /// `openai_compat` without a preset.
    #[serde(default)]
    pub base_url: String,
    /// Bearer token for OpenAI-compatible providers.
    #[serde(default)]
    pub api_key: String,
    /// Issue 125: file-backed alternative to `api_key`. When set, the
    /// trimmed contents of the file are used as the bearer token.
    #[serde(default)]
    pub api_key_file: Option<std::path::PathBuf>,
    /// Fallback model used when no `preferred_models` entry is live.
    pub model: String,
    /// Priority-ordered models. The first one present in the live
    /// `list_models` response wins.
    #[serde(default)]
    pub preferred_models: Vec<String>,
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
    pub semantic: SemanticConfig,
    pub search: SearchConfig,
    pub consolidation: ConsolidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    pub similarity_threshold: f64,
    pub max_results: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub rrf_k: u32,
    /// Candidates fetched from each source (BM25, ANN) before RRF fusion.
    #[serde(default = "default_pre_fusion_limit")]
    pub pre_fusion_limit: u32,
    /// Weight for importance in final reranking (0.0–1.0).
    #[serde(default = "default_importance_weight")]
    pub importance_weight: f64,
    /// Weight for recency in final reranking (0.0–1.0).
    #[serde(default = "default_recency_weight")]
    pub recency_weight: f64,
    /// Decay rate for the forgetting curve (higher = faster forgetting).
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f64,
}

fn default_pre_fusion_limit() -> u32 {
    50
}
fn default_importance_weight() -> f64 {
    0.3
}
fn default_recency_weight() -> f64 {
    0.2
}
fn default_decay_rate() -> f64 {
    0.01
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
    /// Roots that read-only filesystem reads (chat-time path
    /// attachments and decompose path excerpts) are allowed to touch.
    /// Each entry may use `~` for the user's home and is canonicalized
    /// at use time. An empty list means "default to `$HOME`" — never
    /// "anywhere" — so a fresh install can't be coaxed into reading
    /// `/etc` or `/Users/<other>/...`.
    #[serde(default)]
    pub allowed_paths: Vec<String>,
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
    /// Built-in DuckDuckGo HTML scraper. Zero-config, no API key, no
    /// Docker — basic quality but always available.
    #[default]
    #[serde(alias = "duckduckgo", rename = "duckduckgo")]
    DuckDuckGo,
    Searxng,
    Tavily,
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

/// Channel intelligence configuration — bidirectional relay gateways
/// (custom WS agents) that integrate with the channel router and
/// confirmation correlator.
///
/// Distinct from `actions.messaging.channels`, which configures one-way
/// webhook pushes. Entries here open a long-lived WebSocket and can
/// carry user responses back into Brain.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelIntelligenceConfig {
    #[serde(default)]
    pub relays: Vec<RelayEntry>,
    /// Generic preset-driven transports (`http_polled`, `webhook_inbound`,
    /// `webhook_outbound`). Each entry names a preset id that ships
    /// embedded under `crates/channel/presets/` or lives at
    /// `~/.brain/presets/<id>.yaml`.
    #[serde(default)]
    pub transports: Vec<TransportEntry>,
}

/// A single preset-driven transport — which preset, what id, what
/// secrets to plug into the preset's templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportEntry {
    /// Stable id registered with the channel router (e.g. `"chat-main"`).
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Preset id — resolved via the channel crate's preset loader.
    pub preset: String,
    /// Memory namespace attributed to inbound messages on this transport.
    #[serde(default = "default_relay_namespace")]
    pub namespace: String,
    /// Credential substituted into `{credential}` in url/body templates
    /// (bot token, webhook URL, app id — whatever the preset expects).
    /// May be empty.
    #[serde(default)]
    pub credential: String,
    /// Optional signing secret used by `webhook_inbound` transports
    /// whose preset declares a `verifier` (HMAC shared key, Ed25519
    /// pubkey hex, ...).
    #[serde(default)]
    pub signing_secret: Option<String>,
}

/// One relay gateway entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayEntry {
    /// Stable id registered with the channel router (e.g. `"chat-main"`).
    pub id: String,
    /// Human-readable label used in CLI and audit entries.
    pub label: String,
    /// WebSocket URL of the gateway.
    pub url: String,
    /// Memory namespace attributed to messages arriving on this relay.
    #[serde(default = "default_relay_namespace")]
    pub namespace: String,
    /// Optional bearer token forwarded to the gateway (if supported).
    #[serde(default)]
    pub api_key: String,
    /// Reconnection tuning — initial backoff in milliseconds.
    #[serde(default = "default_relay_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    /// Reconnection tuning — max backoff in milliseconds.
    #[serde(default = "default_relay_max_backoff_ms")]
    pub max_backoff_ms: u64,
}

fn default_relay_namespace() -> String {
    "personal".to_string()
}
fn default_relay_initial_backoff_ms() -> u64 {
    1_000
}
fn default_relay_max_backoff_ms() -> u64 {
    60_000
}

/// Agent delegation configuration — specialist CLI/HTTP agents that
/// orchestrator-level `Implement` steps can hand off to.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentsConfig {
    /// Manually-registered delegates. Kept for advanced setups and
    /// backward compatibility; most users rely on `auto_discovery`.
    #[serde(default)]
    pub delegates: Vec<AgentEntry>,
    /// Ordered fallback agent names applied when a delegation fails on
    /// a retryable error. Names must match discovered ids or `delegates`
    /// entries.
    #[serde(default)]
    pub fallbacks: Vec<String>,
    /// Whether timeout failures should trigger fallback retries
    /// (default: true). Set to false for tasks where retry cost is
    /// prohibitive.
    #[serde(default = "default_retry_on_timeout")]
    pub retry_on_timeout: bool,
    /// Scan `$PATH` on startup and auto-register known CLI agents using
    /// the built-in fingerprint table. Default: true. Set to `false` to
    /// go fully manual via `delegates[]`.
    #[serde(default = "default_auto_discovery")]
    pub auto_discovery: bool,
    /// Per-agent overrides merged on top of discovery defaults. Keyed
    /// by the canonical agent id from the fingerprint table.
    #[serde(default)]
    pub discovery_overrides: std::collections::HashMap<String, AgentDiscoveryOverride>,
}

fn default_retry_on_timeout() -> bool {
    true
}

fn default_auto_discovery() -> bool {
    true
}

/// Tweak a single auto-discovered agent. All fields are optional —
/// unset ones keep the fingerprint default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentDiscoveryOverride {
    /// Force a specific binary path instead of the `$PATH` hit.
    #[serde(default)]
    pub binary: Option<String>,
    /// Exclude from the registry entirely.
    #[serde(default)]
    pub disabled: bool,
    /// Override the invocation args (supports `{prompt}` / `{task_id}`).
    #[serde(default)]
    pub args: Option<Vec<String>>,
    /// Force stdin vs. argv prompt delivery.
    #[serde(default)]
    pub prompt_via_stdin: Option<bool>,
}

/// One registered delegate. Currently only `kind = "subprocess"` is
/// supported — any CLI agent the orchestrator can spawn. Auto-discovery
/// covers most common agents without needing manual entries here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEntry {
    /// Registered name — this is what appears in `StepAction::Implement`.
    pub name: String,
    /// Adapter kind (`"subprocess"`).
    pub kind: String,
    /// Optional alias registered alongside `name`. Handy for routing
    /// shorthand request names to the canonical entry.
    #[serde(default)]
    pub alias: Option<String>,
    /// Binary to launch. Required for `subprocess`.
    #[serde(default)]
    pub binary: String,
    /// Args passed to the binary. Supports `{prompt}` and `{task_id}`
    /// substitution.
    #[serde(default)]
    pub args: Vec<String>,
    /// Default working directory for the delegate. Task-level workdir
    /// (set by the orchestrator) wins when present.
    #[serde(default)]
    pub workdir: Option<String>,
    /// Whether the prompt is written to the child's stdin rather than
    /// templated into `args`. Defaults to `true`. Ignored for
    /// argv-templated entries that don't read stdin.
    #[serde(default = "default_prompt_via_stdin")]
    pub prompt_via_stdin: bool,
    /// Declared capability tags (e.g. `["code-edit","rust"]`).
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_prompt_via_stdin() -> bool {
    true
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
    /// Granted permissions. Recognised scopes:
    /// - `"read"`   — read-only memory + signal/status endpoints
    /// - `"write"`  — submit signals, store/forget facts (Issue 127: does
    ///   NOT imply `read`; list both if needed)
    /// - `"export"` — bulk memory export (Issue 123)
    /// - `"admin"`  — implicit superset of every other scope (Issue 127)
    pub permissions: Vec<String>,
    /// Agent identity bound to this key. Used by adapters to resolve a
    /// `Principal` from the `identity:` config. Backwards-compatible
    /// default: `None` — adapters then send `Signal.principal = None`
    /// and the identity gate is skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}

impl ApiKeyConfig {
    /// Returns true if this key grants the requested permission.
    ///
    /// Issue 127: the `admin` permission implicitly grants every other
    /// scope (read, write, export). All other scopes are exact match —
    /// `write` does **not** imply `read`, so historically a key with
    /// `["write"]` could not call read endpoints, and that contract is
    /// preserved.
    pub fn has_permission(&self, perm: &str) -> bool {
        if self.permissions.iter().any(|p| p == "admin") {
            return true;
        }
        self.permissions.iter().any(|p| p == perm)
    }
}

/// Access-control configuration (API keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessConfig {
    pub api_keys: Vec<ApiKeyConfig>,
    /// Per-client (per-API-key) rate limiting applied across HTTP / WS /
    /// gRPC adapters. Disabled by default so a fresh install behaves like
    /// older versions; enable in `default.yaml` to throttle abusive
    /// clients without changing identity wiring.
    #[serde(default)]
    pub rate_limit: ClientRateLimitConfig,
}

impl AccessConfig {
    /// Find a key entry by its raw key string. Delegates to the constant-
    /// time helper in `auth` (Issue 62).
    pub fn find_key(&self, key: &str) -> Option<&ApiKeyConfig> {
        crate::auth::find_key_ct(&self.api_keys, key)
    }
}

/// Tuning surface for adapter-level rate limiting (Issue 51).
///
/// Defaults are conservative: 60 tokens/min with a burst of 20, so a
/// well-behaved client sees no impact while a tight loop is rejected
/// after the burst is drained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientRateLimitConfig {
    #[serde(default = "ClientRateLimitConfig::default_enabled")]
    pub enabled: bool,
    /// Token grant per `refill_interval_ms`. Steady-state rate is
    /// `tokens_per_refill / refill_interval_ms * 1000` per second.
    #[serde(default = "ClientRateLimitConfig::default_tokens_per_refill")]
    pub tokens_per_refill: u32,
    #[serde(default = "ClientRateLimitConfig::default_refill_interval_ms")]
    pub refill_interval_ms: u64,
    /// Maximum tokens the bucket holds — the burst ceiling.
    #[serde(default = "ClientRateLimitConfig::default_burst_capacity")]
    pub burst_capacity: u32,
}

impl Default for ClientRateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            tokens_per_refill: Self::default_tokens_per_refill(),
            refill_interval_ms: Self::default_refill_interval_ms(),
            burst_capacity: Self::default_burst_capacity(),
        }
    }
}

impl ClientRateLimitConfig {
    pub fn default_enabled() -> bool {
        true
    }
    pub fn default_tokens_per_refill() -> u32 {
        60
    }
    pub fn default_refill_interval_ms() -> u64 {
        60_000
    }
    pub fn default_burst_capacity() -> u32 {
        20
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptersConfig {
    pub http: HttpAdapterConfig,
    pub ws: WebSocketAdapterConfig,
    pub mcp: McpAdapterConfig,
    pub grpc: GrpcAdapterConfig,
    /// Terminal Bridge gRPC server — backs `Intent::OpenTerminalSession`
    /// and friends. Default enabled so AI agents can drive PTY sessions
    /// out of the box.
    #[serde(default = "TerminalAdapterConfig::default_enabled")]
    pub terminal: TerminalAdapterConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpAdapterConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub cors: bool,
    /// Issue 131: when true, the SSE `/v1/events` stream replaces
    /// content-bearing fields (LLM responses, notification bodies) with a
    /// `[redacted]` marker so an observer with `read` scope sees event
    /// shape and counts but no message text. Default `false` to preserve
    /// the existing local-dev behavior; flip on for shared deployments.
    #[serde(default)]
    pub sse_redact_previews: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

/// Terminal Bridge gRPC server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalAdapterConfig {
    pub enabled: bool,
    pub port: u16,
}

impl TerminalAdapterConfig {
    /// Default for `#[serde(default)]` on `AdaptersConfig.terminal` — keeps
    /// the bridge available out of the box for fresh installs whose YAML
    /// pre-dates this field.
    pub fn default_enabled() -> Self {
        Self {
            enabled: true,
            port: 19793,
        }
    }
}

impl Default for TerminalAdapterConfig {
    fn default() -> Self {
        Self::default_enabled()
    }
}

impl BrainConfig {}

mod loader;

#[cfg(test)]
mod tests;
