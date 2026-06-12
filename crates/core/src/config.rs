//! Configuration management for Brain.
//!
//! Loads configuration from multiple sources with this priority (highest -> lowest):
//! 1. Environment variables (`BRAIN_` prefix, e.g. `BRAIN_LLM__MODEL`)
//! 2. User config file (`~/.brain/config.yaml`)
//! 3. Embedded defaults (compiled into the binary)

/// Default configuration embedded at compile time.
/// This means `brain` works anywhere without needing config files on disk.
/// Also the single source of truth the product self-model (the `selfmodel`
/// crate) slices into config-schema grounding for the SOUL, handed in via
/// [`BrainConfig::default_config_content`].
pub(crate) const DEFAULT_CONFIG: &str = include_str!("../default.yaml");

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
    /// Logging policy — base level, per-subsystem overrides, output format,
    /// and daemon log-file rotation. Default is empty/`info` pretty with daily
    /// rotation; `RUST_LOG` still overrides the computed filter at runtime.
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Learned self-model knobs — currently the capability-fitness loop that
    /// records per-tool success/failure and feeds it back into tool ranking
    /// and the SOUL capability digest. Default is on with a 30-day half-life.
    #[serde(default)]
    pub learning: LearningConfig,
    /// Runtime resource-observability knobs — the resource sampler's cadence
    /// and the per-gauge ceilings that trip a `ResourcePressure` event.
    /// Default is a 30s sample with generous, fail-safe ceilings.
    #[serde(default)]
    pub observability: ObservabilityConfig,
    /// External-service health monitoring — a list of HTTP/TCP endpoints to
    /// probe on a cadence, alerting on up↔down transitions. Default is empty,
    /// so a fresh install probes nothing.
    #[serde(default)]
    pub monitoring: MonitoringConfig,
}

/// Learned self-model configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LearningConfig {
    #[serde(default)]
    pub capability_fitness: CapabilityFitnessConfig,
}

/// Capability-fitness learning: per-tool success/failure mass that decays
/// under the forgetting curve and nudges the chat tool-loop's advertised
/// ranking. See `cerebellum::CapabilityFitnessStore`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityFitnessConfig {
    /// Record outcomes, boost ranking, and surface "proven tools" in the
    /// digest. When false the store is inert (nothing recorded or surfaced).
    #[serde(default = "CapabilityFitnessConfig::default_enabled")]
    pub enabled: bool,
    /// Decay half-life in days: how long a success/failure observation keeps
    /// half its weight. Longer = slower forgetting.
    #[serde(default = "CapabilityFitnessConfig::default_half_life_days")]
    pub half_life_days: f64,
}

impl CapabilityFitnessConfig {
    fn default_enabled() -> bool {
        true
    }
    fn default_half_life_days() -> f64 {
        30.0
    }
    /// Half-life expressed in hours, as the fitness store consumes it.
    pub fn half_life_hours(&self) -> f64 {
        self.half_life_days * 24.0
    }
}

impl Default for CapabilityFitnessConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            half_life_days: Self::default_half_life_days(),
        }
    }
}

/// Runtime resource-observability configuration. Drives the resource sampler
/// that gauges process RSS, CPU, open SQLite connections, and `~/.brain` disk
/// usage, plus the thresholds at which a `ResourcePressure` event is emitted.
/// Default is a 30s sample cadence with generous, fail-safe ceilings, so a
/// fresh install never trips a pressure event under normal load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Seconds between resource samples. The sampler is a single bounded
    /// background task; lower = more responsive pressure detection at a
    /// slightly higher idle cost.
    #[serde(default = "ObservabilityConfig::default_resource_sample_secs")]
    pub resource_sample_secs: u64,
    /// Per-gauge ceilings above which a `ResourcePressure` event fires
    /// (edge-triggered, not per sample).
    #[serde(default)]
    pub thresholds: ResourceThresholds,
    /// Sampling for high-volume, low-information log lines (the resource
    /// sampler heartbeat, etc.).
    #[serde(default)]
    pub log_sampling: LogSamplingConfig,
}

impl ObservabilityConfig {
    fn default_resource_sample_secs() -> u64 {
        30
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            resource_sample_secs: Self::default_resource_sample_secs(),
            thresholds: ResourceThresholds::default(),
            log_sampling: LogSamplingConfig::default(),
        }
    }
}

/// Log-sampling policy: emit only 1 in N of designated high-volume log lines
/// so a hot loop doesn't drown the log. The metric/event behind each line is
/// still recorded every time — only the *log line* is throttled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogSamplingConfig {
    /// Emit 1 in N of high-volume log lines. `1` (the default) logs every
    /// line — sampling off. Raise it in production to thin periodic chatter.
    #[serde(default = "LogSamplingConfig::default_high_volume_1_in_n")]
    pub high_volume_1_in_n: u32,
}

impl LogSamplingConfig {
    fn default_high_volume_1_in_n() -> u32 {
        1
    }
}

impl Default for LogSamplingConfig {
    fn default() -> Self {
        Self {
            high_volume_1_in_n: Self::default_high_volume_1_in_n(),
        }
    }
}

/// Per-gauge pressure ceilings. A gauge crossing its ceiling emits a
/// `ResourcePressure` event; defaults are generous so normal operation is
/// silent. A `0` disables that gauge's threshold (it never fires).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// Resident-set-size ceiling, in mebibytes.
    #[serde(default = "ResourceThresholds::default_rss_mb")]
    pub rss_mb: u64,
    /// Process CPU-utilisation ceiling, in percent (single-core basis, so
    /// values above 100 are possible on a multi-core busy loop).
    #[serde(default = "ResourceThresholds::default_cpu_pct")]
    pub cpu_pct: f64,
    /// `~/.brain` data-directory disk-usage ceiling, in mebibytes.
    #[serde(default = "ResourceThresholds::default_disk_mb")]
    pub disk_mb: u64,
    /// Open-file-descriptor ceiling (count). Crossing it warns of a possible
    /// fd leak before the process hits its OS `RLIMIT_NOFILE` and starts
    /// failing to open files/sockets. Generous by default so normal operation
    /// is silent.
    #[serde(default = "ResourceThresholds::default_open_fds")]
    pub open_fds: u64,
}

impl ResourceThresholds {
    fn default_rss_mb() -> u64 {
        2048
    }
    fn default_cpu_pct() -> f64 {
        90.0
    }
    fn default_disk_mb() -> u64 {
        10_240
    }
    fn default_open_fds() -> u64 {
        1024
    }
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            rss_mb: Self::default_rss_mb(),
            cpu_pct: Self::default_cpu_pct(),
            disk_mb: Self::default_disk_mb(),
            open_fds: Self::default_open_fds(),
        }
    }
}

/// External-service health monitoring. Each [`ServiceCheck`] drives one bounded
/// background probe loop that periodically reaches a service (HTTP or raw TCP)
/// and, on an up↔down *transition*, surfaces a proactive notification through
/// the same router the resource sampler uses. Default is an empty list, so a
/// fresh install spawns no probes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Services to health-check. One bounded background loop is spawned per
    /// entry; an empty list (the default) spawns none.
    #[serde(default)]
    pub services: Vec<ServiceCheck>,
    /// Network-connectivity probing — the kernel's Online/Degraded/Offline
    /// state. See [`ConnectivityProbeConfig`].
    #[serde(default)]
    pub connectivity: ConnectivityProbeConfig,
}

/// Connectivity probing: derives the kernel's `Online / Degraded / Offline`
/// state by reaching the **already-configured remote endpoints** (remote LLM
/// providers) — never a third-party beacon, so enabling this adds no new
/// egress destination. With nothing remote configured (fully-local install)
/// no probe loop is spawned and the state stays `Online`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityProbeConfig {
    /// Probe at all. Disabling pins the kernel's view to `Online`.
    #[serde(default = "ConnectivityProbeConfig::default_enabled")]
    pub enabled: bool,
    /// Seconds between probe rounds.
    #[serde(default = "ConnectivityProbeConfig::default_interval_secs")]
    pub interval_secs: u64,
    /// Per-target TCP-connect timeout in seconds.
    #[serde(default = "ConnectivityProbeConfig::default_timeout_secs")]
    pub timeout_secs: u64,
    /// Explicit `host:port` probe targets. Empty (the default) derives the
    /// target set from the configured remote (non-loopback) LLM provider
    /// endpoints, keeping probe egress inside what the user already opted
    /// into reaching.
    #[serde(default)]
    pub targets: Vec<String>,
}

impl ConnectivityProbeConfig {
    fn default_enabled() -> bool {
        true
    }
    fn default_interval_secs() -> u64 {
        60
    }
    fn default_timeout_secs() -> u64 {
        5
    }
}

impl Default for ConnectivityProbeConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            interval_secs: Self::default_interval_secs(),
            timeout_secs: Self::default_timeout_secs(),
            targets: Vec::new(),
        }
    }
}

/// Probe protocol for a [`ServiceCheck`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceCheckKind {
    /// HTTP(S) GET — healthy when the response status matches `expect_status`
    /// (or is any 2xx when that is unset).
    #[default]
    Http,
    /// Raw TCP connect — healthy when the connection is accepted before the
    /// timeout. `target` is `host:port`.
    Tcp,
}

/// One external service to health-check. `target` is a URL for the `http` kind
/// or `host:port` for the `tcp` kind. Probes are edge-triggered: a notification
/// fires only when the service crosses between reachable and unreachable, never
/// once per interval while it stays in one state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCheck {
    /// Stable label used in the alert text and the notification's
    /// `triggered_by` (`service_health:<name>`).
    pub name: String,
    /// Probe protocol. Defaults to `http`.
    #[serde(default)]
    pub kind: ServiceCheckKind,
    /// URL (`http` kind) or `host:port` (`tcp` kind) to reach.
    pub target: String,
    /// Seconds between probes. Default 60.
    #[serde(default = "ServiceCheck::default_interval_secs")]
    pub interval_secs: u64,
    /// Per-probe timeout in seconds — a probe that does not complete in this
    /// window counts as unreachable. Default 10.
    #[serde(default = "ServiceCheck::default_timeout_secs")]
    pub timeout_secs: u64,
    /// HTTP only: the exact status code that counts as healthy. When unset,
    /// any 2xx response is healthy. Ignored for the `tcp` kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect_status: Option<u16>,
}

impl ServiceCheck {
    fn default_interval_secs() -> u64 {
        60
    }
    fn default_timeout_secs() -> u64 {
        10
    }
}

/// Logging configuration. Drives the `tracing` subscriber the CLI installs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Base level applied to the `brain` target when neither `RUST_LOG` nor a
    /// per-command default is in force: `trace|debug|info|warn|error`.
    #[serde(default = "LoggingConfig::default_level")]
    pub level: String,
    /// Per-subsystem level overrides, keyed by tracing target (crate name,
    /// e.g. `hippocampus`, `signal`). Each becomes an `EnvFilter` directive.
    #[serde(default)]
    pub targets: HashMap<String, String>,
    /// Output format: `pretty` (human) or `json` (structured/machine).
    #[serde(default)]
    pub format: LogFormat,
    /// Daemon log-file rotation cadence for `logs/brain.log`.
    #[serde(default)]
    pub rotation: LogRotation,
}

impl LoggingConfig {
    fn default_level() -> String {
        "info".to_string()
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: Self::default_level(),
            targets: HashMap::new(),
            format: LogFormat::default(),
            rotation: LogRotation::default(),
        }
    }
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

/// Daemon log-file rotation cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogRotation {
    #[default]
    Daily,
    Hourly,
    Never,
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
    /// Legacy single-provider model name. Superseded by per-entry
    /// `llm.providers[].model` + `preferred_models[]`. Still consulted
    /// when `providers[]` is empty.
    #[deprecated(
        note = "Set `llm.providers[].model` (and optionally `preferred_models`) instead."
    )]
    pub model: String,
    /// Legacy single-provider endpoint. Superseded by per-entry
    /// `llm.providers[].base_url`. Still consulted when `providers[]`
    /// is empty and by `Embedder::from_config` to pick the embedding
    /// transport.
    #[deprecated(
        note = "Set `llm.providers[].base_url` instead. Embedder transport selection still reads this field as a fallback."
    )]
    pub base_url: String,
    pub temperature: f64,
    pub max_tokens: u32,
    /// The active model's input context window, in tokens. Drives the
    /// prompt assembler's [`TokenBudget`](cortex) so a large-window model
    /// (e.g. 128k) reads far more file/attachment + memory content instead
    /// of being clipped to the conservative 8k default. Set this to your
    /// model's real context size. Defaults to 8192 (safe for most models)
    /// when omitted, preserving the historical budget.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// API key for the LLM provider (required for OpenAI, OpenRouter, etc.).
    /// Can also be set via `BRAIN_LLM__API_KEY` environment variable.
    /// Prefer `api_key_file` (chmod-0600) for secrets that shouldn't live
    /// in YAML, or move credentials to `llm.providers[].api_key_file`.
    #[deprecated(
        note = "Move credentials to `llm.providers[].api_key_file` (or `api_key_file` here) — the YAML field gets backed up and replicated."
    )]
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
    /// Task-tier routing over the provider pool. Each tier is an ordered
    /// list of `providers[].name` entries forming that tier's failover
    /// chain. Kernel chores (intent-classification fallback, importance,
    /// history compaction, web-search synthesis) ride `fast`; chat and
    /// task decomposition ride `deep`; everything unrouted rides
    /// `balanced`. An empty tier uses the default startup chain, so an
    /// unset `tiers` block changes nothing. Naming a *local* provider in
    /// `fast` is the residency local lane: kernel chores then provably
    /// never leave the machine even when chat uses a cloud provider.
    #[serde(default)]
    pub tiers: LlmTiersConfig,
}

/// `llm.tiers` — named provider chains per task tier. See
/// [`LlmConfig::tiers`]. A name that doesn't match any `providers[]`
/// entry is a startup error: a typo must never silently reroute a tier
/// meant to stay local onto the default (possibly remote) chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LlmTiersConfig {
    /// Cheap/latency-sensitive kernel work (classification fallback,
    /// importance, compaction, web synthesis, background nudges).
    #[serde(default)]
    pub fast: Vec<String>,
    /// Work not explicitly routed to `fast` or `deep`.
    #[serde(default)]
    pub balanced: Vec<String>,
    /// Quality-sensitive generation (chat, task decomposition).
    #[serde(default)]
    pub deep: Vec<String>,
}

impl LlmTiersConfig {
    /// True when no tier names any provider — the zero-config shape.
    pub fn is_unset(&self) -> bool {
        self.fast.is_empty() && self.balanced.is_empty() && self.deep.is_empty()
    }
}

/// Default context window when `llm.context_window` is omitted. 8192 is the
/// historical assembler budget — safe for nearly every model, and large-window
/// models opt into more by setting their real size.
pub(crate) fn default_context_window() -> usize {
    8192
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
    /// Per-namespace policy (`memory.namespaces.<name>`). An entry also
    /// governs `name/…` sub-namespaces unless a more specific entry
    /// exists. Namespaces without an entry default to `residency: any`.
    #[serde(default)]
    pub namespaces: std::collections::HashMap<String, crate::residency::NamespaceConfig>,
    /// Per-agent memory-trust weights (`memory.trust`) applied to recall
    /// scoring. Defaults to the identity (every weight 1.0).
    #[serde(default)]
    pub trust: crate::trust::MemoryTrustConfig,
}

impl MemoryConfig {
    /// Resolve the residency policy for a namespace (exact entry, then
    /// `/`-truncated ancestors — see [`crate::residency::resolve_residency`]).
    pub fn residency_of(&self, namespace: &str) -> crate::residency::Residency {
        crate::residency::resolve_residency(namespace, |s| {
            self.namespaces.get(s).map(|c| c.residency)
        })
    }

    /// Compile the per-namespace residency entries into a config-free
    /// [`crate::residency::ResidencyPolicy`] for subsystems that never
    /// see `BrainConfig`.
    pub fn residency_policy(&self) -> crate::residency::ResidencyPolicy {
        crate::residency::ResidencyPolicy::new(
            self.namespaces
                .iter()
                .map(|(n, c)| (n.clone(), c.residency))
                .collect(),
        )
    }

    /// Names of all namespaces configured `local_only`, for status and
    /// export surfaces.
    pub fn local_only_namespaces(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .namespaces
            .iter()
            .filter(|(_, c)| c.residency.is_local_only())
            .map(|(n, _)| n.as_str())
            .collect();
        names.sort_unstable();
        names
    }
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
    /// Replace the fingerprint's default capability declaration.
    /// Mirrors the runtime `AgentCapabilities` shape in `brainos-delegate`;
    /// when set, every listed field is forwarded onto the registry entry
    /// in place of the discovery default.
    #[serde(default)]
    pub capabilities: Option<CapabilitiesOverride>,
}

/// YAML-side mirror of `brainos_delegate::AgentCapabilities`. Lives here
/// to keep `brainos-core` free of a `brainos-delegate` dependency
/// (delegate already depends on us, so the reverse would be a cycle).
/// The CLI bootstrap layer converts this into the runtime type when
/// building the registry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitiesOverride {
    /// Free-form capability tags (`"code-edit"`, `"plan"`, `"research"`).
    #[serde(default)]
    pub tags: Vec<String>,
    /// Preferred languages/frameworks (`"rust"`, `"typescript"`).
    #[serde(default)]
    pub languages: Vec<String>,
    /// Maximum concurrent delegations the orchestrator will dispatch to
    /// this agent at once. Defaults to 1 (conservative).
    #[serde(default = "default_capability_concurrency")]
    pub max_concurrency: u32,
    /// Whether this delegate needs network — informs sandbox policy.
    #[serde(default)]
    pub needs_network: bool,
}

fn default_capability_concurrency() -> u32 {
    1
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
pub mod migrate;

#[cfg(test)]
mod tests;
