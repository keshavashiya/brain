//! # Brain Signal Processor
//!
//! Central hub that converts all input signals (CLI, HTTP, WebSocket, MCP, gRPC)
//! into a unified Signal type and routes them through the Brain pipeline.
//!
//! The SignalProcessor wires together:
//! - Thalamus (intent classification)
//! - Amygdala (importance scoring)
//! - Hippocampus (episodic + semantic memory)
//! - Cortex (LLM reasoning + context assembly)
//! - NotificationRouter (proactive delivery)

pub mod approval;
pub mod authz;
pub mod notification;
pub mod reflex_runner;
pub mod terminal_graph_mirror;
pub mod types;

mod attachment;
mod budget_guard;
mod bundles;
mod constructors;
mod exchange;
mod extract;
mod memory_subsystem;
mod pipeline;
mod quarantine;
mod recall;
mod render;
mod secrets;
mod streaming;
mod tier_budget;
mod wiring;

pub use budget_guard::{check_llm_input, record_llm_usage, BudgetGate};

pub use approval::ChannelApprovalNotifier;

// Re-export all public types so `use signal::X;` continues to work.
pub use types::*;

// ─── Signal Processor ─────────────────────────────────────────────────────────

/// Central processor that wires all Brain subsystems together.
///
/// One instance is shared across all adapters. Each incoming Signal is routed
/// through intent classification → importance scoring → memory → LLM → response.
pub struct SignalProcessor {
    // ── Always-on core wiring ────────────────────────────────────────────
    config: brain::BrainConfig,
    classifier: thalamus::IntentClassifier,
    importance: amygdala::ImportanceScorer,
    /// Memory subsystem: episodic + semantic stores, the embedding provider
    /// and its query cache, the recall engine, and the dual-memory reader.
    /// See [`memory_subsystem::MemorySubsystem`].
    memory: memory_subsystem::MemorySubsystem,
    /// The `deep` tier chain — quality-sensitive generation (chat turns,
    /// streaming, the tool loop). With `llm.tiers` unset all three tier
    /// fields wrap the same default chain, preserving single-chain
    /// behavior. Each chain is wrapped in a
    /// [`tier_budget::TierUsageRecorder`], so completed generations are
    /// also recorded under `tier:<name>` once a cost budget is wired.
    llm: std::sync::Arc<dyn cortex::LlmProvider>,
    /// The `fast` tier chain — cheap kernel chores: classifier fallback,
    /// importance, history compaction, web-search synthesis. Routing a
    /// local provider here is the residency "local lane": these chores
    /// then provably never leave the machine.
    llm_fast: std::sync::Arc<dyn cortex::LlmProvider>,
    /// The `balanced` tier chain — work not explicitly routed fast/deep.
    llm_balanced: std::sync::Arc<dyn cortex::LlmProvider>,
    /// Cost-budget slot shared with the tier recorders. Filled by
    /// `with_cost_budget`; empty means tier accounting is off.
    tier_budget: tier_budget::BudgetCell,
    context_assembler: cortex::context::ContextAssembler,
    /// LRU cache of compacted-history summaries. Keyed by a fast hash of the
    /// overflow turns being summarized, so a long chat doesn't re-summarize
    /// the same prefix every turn. Only the summary text is stored.
    history_summary_cache: std::sync::Mutex<lru::LruCache<u64, std::sync::Arc<str>>>,
    procedures: cerebellum::ProcedureStore,
    /// Learned capability self-model: per-tool success/failure mass recorded
    /// after each dispatch (see [`SignalProcessor::dispatch_tool_route`]),
    /// fed back as a bounded ranking nudge in the chat tool-loop and a
    /// "proven tools" line in the capability digest. Inert when
    /// `learning.capability_fitness.enabled = false`.
    fitness: cerebellum::CapabilityFitnessStore,
    /// Cross-subsystem metrics (embedding, consolidation, circuit breaker, intent).
    metrics: std::sync::Arc<metrics::SubsystemMetrics>,
    /// Runtime proactivity toggle. Initialised from
    /// `config.proactivity.enabled` and flipped by
    /// `handle_set_proactivity`. The CLI bootstrap hands a clone of this
    /// `Arc` to the ganglia habit-engine and open-loop background tasks,
    /// which check it on every tick and skip generation when it is
    /// `false`. Spawn-time still respects the startup config: if
    /// `proactivity.enabled` was `false` at boot, the tasks were never
    /// spawned and a runtime flip to `true` cannot resurrect them —
    /// that's the v1.0 work.
    proactivity_enabled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The kernel's network view (Online/Degraded/Offline). The serve
    /// loop's connectivity probe writes through a clone of this handle
    /// (see [`SignalProcessor::connectivity`]); the pipeline reads it per
    /// turn to route offline generation onto a local tier, degrade web
    /// search honestly, and stamp the capability digest. Defaults to
    /// Online, so processors without a probe loop (CLI one-shots, tests)
    /// behave exactly as before.
    connectivity: brain::Connectivity,

    // ── Capability bundles (opt-in via builder) ──────────────────────────
    /// Approval / accounting / sandbox gates. See [`bundles::SafetyBundle`].
    pub(crate) safety: bundles::SafetyBundle,
    /// Cross-channel routing + delivery. See [`bundles::ChannelBundle`].
    pub(crate) channels: bundles::ChannelBundle,
    /// Motor cortex + capability kernel. See [`bundles::CapabilityBundle`].
    pub(crate) capability: bundles::CapabilityBundle,
    /// Event bus + cancellation registry. See [`bundles::ObservabilityBundle`].
    pub(crate) observability: bundles::ObservabilityBundle,

    // ── Top-level optionals that didn't fit a bundle ─────────────────────
    /// Task orchestrator — decomposes requests into executable plans.
    orchestrator: Option<std::sync::Arc<orchestrate::TaskOrchestrator>>,
    /// Registry of specialist agent delegates (Claude Code, custom subprocess, etc.).
    agent_registry: Option<std::sync::Arc<delegate::AgentRegistry>>,
    /// Authorization store. When set and the incoming `Signal` carries a
    /// `Principal`, the pipeline gates the classified intent through
    /// `IdentityStore::check` before executing it. Unwired = back-compat
    /// (no enforcement).
    identity_store: Option<std::sync::Arc<dyn identity::IdentityStore>>,
    /// Per-client rate limiter registry (Issue 51). Wired through adapters
    /// (HTTP/WS/gRPC) to throttle abusive callers without changing
    /// identity resolution. Unwired processors disable rate limiting.
    client_rate_limits: Option<std::sync::Arc<resilience::RateLimitRegistry>>,
    /// Brain's grounded self-knowledge (CLI commands, config schema, policy),
    /// injected into the chat prompt as an authoritative "About Brain" section
    /// so the SOUL stops fabricating the product's own surface. Built from code
    /// at bootstrap; `None` leaves the prompt unchanged (back-compat for
    /// non-chat/test processors).
    product_self_model: Option<std::sync::Arc<selfmodel::ProductSelfModel>>,
    /// Brain's grounded knowledge of the host machine (OS/arch, cores, RAM,
    /// GPU budget, disk) — the situational sibling of the product self-model.
    /// Probed once at bootstrap; the capability digest names the machine
    /// class from it. `None` leaves the digest unchanged (back-compat).
    host_model: Option<std::sync::Arc<selfmodel::HostModel>>,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_signal_new() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello world");
        assert!(!signal.id.is_nil());
        assert_eq!(signal.source, SignalSource::Cli);
        assert_eq!(signal.channel, "cli");
        assert_eq!(signal.sender, "user");
        assert_eq!(signal.content, "hello world");
        assert!(signal.metadata.is_empty());
    }

    #[test]
    fn test_signal_response_ok() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "success");
        assert_eq!(resp.signal_id, id);
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert!(matches!(resp.response, ResponseContent::Text(_)));
        assert_eq!(resp.memory_context.facts_used, 0);
        assert_eq!(resp.memory_context.episodes_used, 0);
    }

    #[test]
    fn test_signal_response_error() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::error(id, "something went wrong");
        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(matches!(resp.response, ResponseContent::Error(_)));
    }

    #[test]
    fn test_memory_context_default() {
        let ctx = MemoryContext::default();
        assert_eq!(ctx.facts_used, 0);
        assert_eq!(ctx.episodes_used, 0);
    }

    #[test]
    fn test_signal_source_serde() {
        let sources = vec![
            SignalSource::Cli,
            SignalSource::Http,
            SignalSource::WebSocket,
            SignalSource::Mcp,
            SignalSource::Grpc,
        ];
        for s in &sources {
            let json = serde_json::to_string(s).unwrap();
            let back: SignalSource = serde_json::from_str(&json).unwrap();
            assert_eq!(s, &back);
        }
    }

    #[test]
    fn test_signal_serde() {
        let signal = Signal::new(SignalSource::Http, "http", "apiclient", "Remember coffee");
        let json = serde_json::to_string(&signal).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(signal.id, back.id);
        assert_eq!(signal.content, back.content);
    }

    #[test]
    fn test_signal_with_agent() {
        let signal =
            Signal::new(SignalSource::Http, "http", "apiclient", "hello").with_agent("claude-code");
        assert_eq!(signal.agent.as_deref(), Some("claude-code"));

        // Serialization round-trip preserves agent
        let json = serde_json::to_string(&signal).unwrap();
        assert!(json.contains("claude-code"));
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.agent.as_deref(), Some("claude-code"));
    }

    #[test]
    fn test_signal_without_agent_omits_field() {
        let signal = Signal::new(SignalSource::Cli, "cli", "user", "hello");
        assert!(signal.agent.is_none());
        let json = serde_json::to_string(&signal).unwrap();
        // skip_serializing_if = "Option::is_none" should omit agent entirely
        assert!(!json.contains("agent"));
    }

    #[test]
    fn test_signal_response_serde() {
        let id = Uuid::new_v4();
        let resp = SignalResponse::ok(id, "hello");
        let json = serde_json::to_string(&resp).unwrap();
        let back: SignalResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.signal_id, back.signal_id);
        assert_eq!(resp.status, back.status);
    }
}
