# Brain OS — Architecture

This document covers the internal design of Brain OS: key abstractions, data flow, storage layer, background loops, delegation and channel transport architecture, and step-by-step guides for building new protocol adapters.

---

## Table of Contents

- [Crate Map](#crate-map)
- [Data Flow](#data-flow-signal-ingestion)
- [Key Types](#key-types)
- [Storage Layer](#storage-layer)
- [Background Loops](#background-loop-architecture)
- [Action Backends](#action-dispatcher-backends-internal)
- [Memory Namespaces](#memory-namespaces)
- [Security Model](#security-model)
- [Channel Integration Pattern](#channel-integration-pattern)

---

## Crate Map

<details>
<summary><strong>Full directory tree</strong></summary>

```
brain/
├── crates/
│   ├── core/           # BrainConfig + loader — shared config types used by every crate
│   │
│   ├── signal/         # Signal / SignalResponse / SignalProcessedEvent types
│   │                     SignalAdapter trait
│   │                     SignalProcessor — the single shared engine that wires all subsystems
│   │
│   ├── thalamus/       # Intent classification — the primary user-facing surface
│   │                     Regex fast-path (compiled at startup) + async LLM fallback with timeout
│   │                     11 intent types: StoreFact, Recall, Forget, Chat, SystemStatus,
│   │                     WebSearch, Schedule, SendMessage, ExecuteCommand, DecomposeTask,
│   │                     QueryAgents
│   │                     (New user-facing features add intents here, not CLI subcommands.)
│   │
│   ├── amygdala/       # Importance scoring with per-process novelty detection → [0.0, 1.0]
│   │                     Delegates keyword heuristics to hippocampus::ImportanceScorer,
│   │                     adds novelty bonus for previously-unseen topic tokens
│   │
│   ├── hippocampus/    # Memory engine
│   │   ├── episodic    # Session-based conversation history, BM25 FTS5 full-text search,
│   │   │                 reinforcement counting, namespace support
│   │   ├── semantic    # Subject-predicate-object facts, dual-write: SQLite + HNSW,
│   │   │                 namespace-scoped ANN search, idempotency guards
│   │   ├── search      # RecallEngine: RRF (BM25 + ANN) fusion + forgetting-curve reranking
│   │   ├── consolidation # Consolidator: prune decayed episodes, promote reinforced episodes
│   │   │                   to semantic facts (idempotency via episode_promotions table)
│   │   ├── embedding   # Embedder: Ollama backend (POST /api/embed) +
│   │   │                 OpenAI-compatible backend (POST /v1/embeddings)
│   │   │                 Deterministic normalized fallback when provider is unavailable
│   │   └── importance  # Keyword-based ImportanceScorer (stateless, no LLM cost)
│   │
│   ├── cortex/         # Reasoning core
│   │   ├── llm         # LlmProvider trait: OllamaProvider + OpenAiProvider
│   │   │                 Streaming and non-streaming generate; health_check; model()
│   │   │                 FalloverProvider: ordered retry chain — tries next provider on
│   │   │                 any retriable error (429/5xx/timeout/unavailable); non-retriable
│   │   │                 errors (4xx auth, InvalidFormat) propagate immediately.
│   │   │                 build_failover_chain(): probes providers at startup, builds chain
│   │   │                 with probed winner first and remaining entries as fallbacks.
│   │   ├── context     # ContextAssembler: token-budgeted prompt builder
│   │   │                 Budget: system(500) + user_model(300) + history(2000) +
│   │   │                 response_buffer(400) + memories(remainder of 8192)
│   │   │                 Agent-attributed memories rendered as [source, agent: X]
│   │   │                 assemble_with_addendum(): appends an addendum to the system
│   │   │                 prompt for per-turn mode switching (e.g. onboarding) without
│   │   │                 mutating the shared assembler
│   │   └── actions     # ActionDispatcher: pluggable backend traits
│   │                     MemoryBackend, WebSearchBackend, SchedulingBackend, MessageBackend
│   │                     Deterministic dispatch contract (disabled / not-configured / real)
│   │
│   ├── cerebellum/     # ProcedureStore — trigger-pattern → steps_json automation rules
│   │                     CRUD operations, case-insensitive trigger matching, use-count tracking
│   │
│   ├── ganglia/        # Proactivity / habit engine + open-loop detection
│   │                     HabitEngine: keyword × day-of-week × hour pattern detection
│   │                     OpenLoopDetector: unresolved commitment scanning + reminders
│   │                     Rate limits: max_per_day, min_interval_minutes, quiet_hours
│   │                     State persisted in SQLite (habit_state table)
│   │
│   ├── audit/          # Append-only audit trail for every action
│   │                     AuditTrail trait + SqliteAuditTrail, ActionTier taxonomy
│   │                     (Read / Write / Execute / Destructive / External)
│   │
│   ├── confirm/        # Nonce-backed confirmation engine for destructive/external actions
│   │                     ConfirmationEngine trait, pending-approval store, TTL expiry
│   │
│   ├── budget/         # Cost/token budget enforcement with circuit breaker
│   │                     CostBudget trait, daily/monthly/per-task limits, breach events
│   │
│   ├── sandbox/        # Command execution sandbox
│   │                     IsolatedSandbox: setrlimit (CPU/AS/NOFILE/FSIZE) pre-exec,
│   │                     binary allowlist, process-group SIGKILL on timeout,
│   │                     macOS sandbox-exec network-deny, Linux unshare(NET/IPC/UTS)
│   │
│   ├── vault/          # Credential vault
│   │                     CredentialVault trait, OS-native backends
│   │                     (macOS Keychain, Linux secret-service, encrypted-file fallback)
│   │
│   ├── orchestrate/    # Task decomposition + execution orchestrator
│   │                     TaskOrchestrator, TaskGraph (DAG), StepState, TaskPhase,
│   │                     LlmDecomposer — turns a natural-language request into an
│   │                     approved plan of tiered steps, then drives execution.
│   │                     Surfaced via the DecomposeTask intent.
│   │
│   ├── delegate/       # External agent delegation
│   │                     AgentDelegate trait, AgentRegistry, DelegateDiscovery,
│   │                     SubprocessAgentDelegate, EscalationHandler
│   │                     Config supports auto-discovery, discovery overrides,
│   │                     manual delegates, and ordered fallback escalation
│   │
│   ├── channel/        # Channel routing + learned preferences
│   │                     ChannelRouter (DefaultChannelRouter), ChannelPreferenceStore
│   │                     (EMA-smoothed weights), ConfirmationCorrelator (inbound nonce
│   │                     parsing), RelayAdapter (external WebSocket gateways), and
│   │                     ChannelTransport engines: polled, webhook-inbound,
│   │                     webhook-outbound, preset/jsonpath/send helpers.
│   │                     `channel.transports[]` is first-class; webhook-inbound
│   │                     route `POST /v1/webhooks/:id` is wired in the HTTP adapter.
│   │
│   ├── storage/        # Storage abstraction layer
│   │   ├── sqlite      # SqlitePool: 18 migrations (v1–v18), WAL mode, thread-safe Mutex<Connection>
│   │   │                 Tables: semantic_facts, episodes, procedures, scheduled_intents,
│   │   │                 _migrations, FTS5 virtual tables (episodes_fts)
│   │   ├── ruvector    # RuVectorStore: wraps ruvector-core (external crate, crates.io)
│   │   │                 Multi-table interface: facts_vec.db, episodes_vec.db
│   │   │                 Vector sanitization, deterministic jitter on insert,
│   │   │                 L2 normalization, deterministic fallback for invalid vectors
│   │   └── encryption  # Encryptor: AES-256-GCM + Argon2id key derivation
│   │                     Per-record unique nonce, encrypts content columns at rest
│   │
│   └── adapters/
│       ├── http/       # Axum REST server (port 19789)
│       │                 Auth (Bearer token), OpenAPI spec (hand-built), Swagger UI,
│       │                 built-in diagnostic Web UI, Prometheus metrics, signal cache
│       ├── ws/         # WebSocket adapter (port 19790, tokio-tungstenite)
│       │                 Auth via first frame {"api_key":"..."}, namespace per message
│       │                 Streaming progress frames: sends {"type":"status","stage":"routing"},
│       │                 {"type":"status","stage":"thinking"} etc. via mpsc channel while
│       │                 prepare() runs, so clients see activity before the first LLM token
│       ├── grpc/       # gRPC adapter (port 19792, tonic)
│       │                 MemoryService (Search, Store, GetFacts, StreamSignals)
│       │                 AgentService (Connect, SendSignal, ReceiveSignals fan-out)
│       │                 Auth interceptor, namespace propagation
│       └── mcp/        # MCP adapter (stdio transport + HTTP transport, port 19791)
│                         6 tools: memory_search, memory_store, memory_facts,
│                         memory_episodes, user_profile, memory_procedures
│                         JSON-RPC 2.0, meta-key auth
│
├── crates/cli/         # `brain` binary — deliberately minimal surface.
│                         Two legitimate purposes only:
│                           1. Bootstrap / lifecycle (must run before the daemon):
│                              init, start, stop, status, serve, mcp, bridge,
│                              service install/uninstall, deps up/down/status,
│                              export, import
│                           2. Security-sensitive stdin (cannot route through NL):
│                              vault init/set/delete, auth login
│                         All other user-facing operations go through Thalamus
│                         as natural-language intents — `brain chat "..."`.
│
├── crates/backends/    # Action backends and resilience primitives
│                         (SearxngSearchBackend, TavilySearchBackend, CustomSearchBackend,
│                          CliSchedulingBackend, WebhookMessageBackend),
│                         CircuitBreaker, resilient_send, promote_candidates
│
│
└── crates/bridge/      # External gateway relay library.
                          BridgeClient: WebSocket client with exponential-backoff reconnection,
                          ping/pong keep-alive, JSON message serialization.
                          Bidirectional: connect_and_relay_bidirectional() pushes proactive
                          notifications outbound alongside inbound message relay.
                          Used by external relay projects to connect messaging platforms to Brain.
```

</details>

### Workspace Members

24 crates total:

```
core  storage  hippocampus  cortex  thalamus  amygdala  signal
adapters/http  adapters/ws  adapters/grpc  adapters/mcp
cerebellum  ganglia  bridge  backends
audit  confirm  budget  sandbox  vault
orchestrate  delegate  channel
cli
```

### Dependency Graph

```
cli ──► signal::SignalProcessor (Arc<SignalProcessor>)
            │
            ├── thalamus        (intent classification)
            ├── amygdala        (importance scoring)
            ├── hippocampus     (memory read / write / consolidation)
            │       └── storage (SQLite + ruvector-core HNSW + AES-GCM encryption)
            ├── cortex          (LLM providers + context assembly + action dispatch)
            ├── cerebellum      (procedure store + trigger matching)
            ├── ganglia         (proactivity / habit engine)
            ├── orchestrate     (task planning / execution)
            ├── delegate        (external agent discovery / delegation / escalation)
            └── channel         (routing / preferences / transports / correlator)

adapters/http  ──► Arc<SignalProcessor>
adapters/ws    ──► Arc<SignalProcessor>
adapters/grpc  ──► Arc<SignalProcessor>
adapters/mcp   ──► Arc<SignalProcessor>

External apps  ──► Brain's HTTP / WS / MCP / gRPC API   (live outside this repo)
```

All adapters share **one** `Arc<SignalProcessor>`. There are no per-adapter memory stores. A fact stored via MCP is immediately visible via HTTP or gRPC.

---

## Data Flow: Signal Ingestion

```
Client Request
     │
     ▼
[Adapter] — parse wire format, authenticate, build Signal{id, source, namespace, content, ...}
     │
     ▼
SignalProcessor::process(&signal)
     │
     ├─ 1. Amygdala: score importance (keyword heuristics + novelty) → f32 [0.0–1.0]
     │
     ├─ 2. Thalamus: classify intent
     │         Regex fast-path first → async LLM fallback if no match
     │         → Classification { intent, confidence, method: Regex|Llm|Fallback }
     │
     ├─ 3. Cerebellum: match stored procedure triggers (case-insensitive substring)
     │         → inject matching steps into LLM context, bump use_count
     │
     ├─ 4. Intent-dependent branch:
     │
     │     StoreFact   → embed("{subject} {predicate} {object}")
     │                 → SemanticStore::store_fact(namespace, triple, vector)
     │                    dual-write: SQLite row + ruvector-core HNSW insert
     │                 → return confirmation
     │
     │     Recall      → embed(query)
     │                 → RecallEngine::recall(BM25 + ANN → RRF → forgetting-curve rerank)
     │                 → ContextAssembler::assemble(query, memories, proc_steps)
     │                 → LlmProvider::generate(messages) → response
     │
     │     Forget      → SemanticStore::find_facts_matching(target, namespace)
     │                 → delete matching facts from SQLite + ruvector
     │
     │     Chat        → recall context (hybrid search)
     │                 → EpisodicStore::store_episode(user turn)
     │                 → ContextAssembler + LlmProvider::generate
     │                 → EpisodicStore::store_episode(assistant turn)
     │
     │     WebSearch   → ActionDispatcher::web_search (SearXNG / Tavily / custom HTTP)
     │     Schedule    → ActionDispatcher::schedule_task (SQLite persist + background poller)
     │     SendMessage → ActionDispatcher::send_message (webhook POST with template)
     │     Command     → ActionDispatcher::execute_command (allowlist + timeout)
     │
     ├─ 5. publish_event → broadcast::Sender<SignalProcessedEvent>
     │         (consumed by gRPC AgentService::ReceiveSignals fan-out stream)
     │
     └─ 6. Return SignalResponse { signal_id, status, response, memory_context }
```

The `SignalResponse` is returned directly to the calling adapter, which sends it back in the protocol-appropriate format.

---

## Key Types

<details>
<summary><strong>Signal</strong> — universal input envelope</summary>

```rust
pub struct Signal {
    pub id: Uuid,
    pub source: SignalSource,    // Cli | Http | WebSocket | Mcp | Grpc
    pub channel: String,
    pub sender: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub namespace: String,       // default: "personal"
    pub agent: Option<String>,   // originating AI agent id, when known
}
```

</details>

<details>
<summary><strong>SignalResponse</strong> — output envelope</summary>

```rust
pub struct SignalResponse {
    pub signal_id: Uuid,
    pub status: ResponseStatus,     // Ok | Error | Processing
    pub response: ResponseContent,  // Text(String) | Json(Value) | Error(String)
    pub memory_context: MemoryContext {
        pub facts_used: usize,
        pub episodes_used: usize,
    },
}
```

</details>

<details>
<summary><strong>SignalAdapter</strong> trait</summary>

```rust
#[async_trait]
pub trait SignalAdapter: Send + Sync {
    fn source(&self) -> SignalSource;
    async fn send(&self, response: SignalResponse) -> Result<(), SignalError>;
}
```

Thin by design — adapters translate wire formats, not business logic.

</details>

<details>
<summary><strong>SignalProcessor</strong> — public API surface</summary>

```rust
impl SignalProcessor {
    pub async fn new(config: BrainConfig) -> Result<Self, SignalError>;
    pub async fn new_with_encryptor(config: BrainConfig, encryptor: Option<Encryptor>)
        -> Result<Self, SignalError>;

    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError>;

    // Direct memory operations — adapters bypass intent classification
    pub async fn store_fact_direct(&self, ns: &str, cat: &str, sub: &str,
        pred: &str, obj: &str) -> Result<String, SignalError>;
    pub async fn search_facts(&self, query: &str, top_k: usize,
        namespace: Option<&str>) -> Vec<SemanticResult>;

    // Builder methods — attach optional subsystems before wrapping in Arc
    pub fn with_notification_router(self, router: NotificationRouter) -> Self;
    pub fn with_action_dispatcher(self, dispatcher: ActionDispatcher) -> Self;

    // Safety infrastructure (all optional, composable)
    pub fn with_audit_trail(self, trail: Arc<dyn audit::AuditTrail>) -> Self;
    pub fn with_confirmation_engine(self, eng: Arc<dyn confirm::ConfirmationEngine>) -> Self;
    pub fn with_cost_budget(self, budget: Arc<dyn budget::CostBudget>) -> Self;
    pub fn with_sandbox_executor(self, exec: Arc<dyn sandbox::SandboxExecutor>) -> Self;
    pub fn with_credential_vault(self, vault: Arc<dyn vault::CredentialVault>) -> Self;

    // Task orchestration
    pub fn with_orchestrator(self, orch: Arc<orchestrate::TaskOrchestrator>) -> Self;

    // Agent delegation
    pub fn with_agent_registry(self, registry: Arc<delegate::AgentRegistry>) -> Self;

    // Channel intelligence
    pub fn with_channel_router(self, router: Arc<dyn channel::ChannelRouter>) -> Self;
    pub fn with_channel_preferences(self, preferences: Arc<dyn channel::ChannelPreferenceStore>) -> Self;
    pub fn with_confirmation_correlator(self, correlator: Arc<channel::ConfirmationCorrelator>) -> Self;

    pub fn audit_trail(&self) -> Option<&Arc<dyn audit::AuditTrail>>;
    pub fn confirmation_engine(&self) -> Option<&Arc<dyn confirm::ConfirmationEngine>>;
    pub fn sandbox_executor(&self) -> Option<&Arc<dyn sandbox::SandboxExecutor>>;
    pub fn orchestrator(&self) -> Option<&Arc<orchestrate::TaskOrchestrator>>;

    // Inspector accessors used by adapter route handlers
    pub fn list_facts(&self, namespace: Option<&str>) -> Vec<Fact>;
    pub fn facts_about(&self, subject: &str) -> Vec<Fact>;
    pub fn list_namespaces(&self) -> Vec<NamespaceStats>;
    pub fn recent_episodes(&self, limit: usize) -> Vec<Episode>;
    pub fn procedures(&self) -> &ProcedureStore;
    pub fn episodic(&self) -> &EpisodicStore;
    pub fn config(&self) -> &BrainConfig;
    pub fn shutdown(&self);   // WAL checkpoint before exit

    // Event bus — consumed by gRPC streaming
    pub fn subscribe_events(&self) -> broadcast::Receiver<SignalProcessedEvent>;
}
```

</details>

---

## Storage Layer

### SQLite

Migration-based schema (18 migrations, v1–v18). WAL mode enabled. Thread safety via `Mutex<Connection>`.

<details>
<summary><strong>Tables</strong></summary>

| Table | Purpose |
|-------|---------|
| `sessions` | Chat session tracking (id, channel, namespace, timestamps) |
| `semantic_facts` | S-P-O triples with namespace, importance, source_episode_id |
| `episodes` | Conversation history with role, importance, decay_rate, reinforcement_count |
| `episodes_fts` | FTS5 virtual table (BM25 full-text search over episode content) |
| `user_profile` | Key-value store for user preferences |
| `procedures` | trigger_pattern → steps_json automation rules |
| `audit_log` | Action audit trail (action type, input, output, timestamps) |
| `scheduled_intents` | Persisted scheduling intents |
| `episode_promotions` | Idempotency log for episode → semantic-fact promotions |
| `notification_outbox` | Proactive notification queue with priority and delivery status |
| `habit_state` | Rate-limit state for proactivity engine |
| `_migrations` | Applied migration version log |

</details>

### Vector Index

Uses [`ruvector-core`](https://crates.io/crates/ruvector-core) for HNSW approximate nearest-neighbour search. Storage: `~/.brain/ruvector/{facts_vec.db, episodes_vec.db}`.

<details>
<summary><strong>Robustness guarantees</strong></summary>

Before any insert or search:

1. Dimension check — vectors with wrong size get deterministic fallback
2. Finite check — NaN / Inf values trigger deterministic fallback
3. Zero-norm check — zero vectors get deterministic fallback
4. L2 normalization applied to all vectors
5. Deterministic per-ID jitter on insert to avoid pathological HNSW duplicate-distance panics

The `deterministic_fallback_embedding(seed, dimensions)` function (FNV-1a hash → xorshift64* PRNG → normalized) ensures writes never fail silently.

</details>

### Hybrid Search

<details>
<summary><strong>RecallEngine pipeline</strong></summary>

```
1. EpisodicStore::search_bm25(query, limit, namespace)  → BM25 ranked list
2. SemanticStore::search_similar(query_vector, limit, namespace) → ANN ranked list
3. rrf_fuse([bm25_ranked, ann_ranked], k=60)            → single fused ranking
4. For each fused ID: look up full record, compute forgetting_curve score
   retention = importance × e^(−decay_rate × hours_since_last_access)
   final_score = rrf_score
               + importance_weight × importance
               + recency_weight × retention
5. Sort by final_score descending → return top_k
```

RRF handles overlap boosting (items in both lists score higher) and disjoint lists, with full unit test coverage.

</details>

---

## Background Loop Architecture

`brain serve` / `brain start` spawns adapter tasks and optional intelligence tasks into a single `tokio::task::JoinSet`. All tasks share `Arc<SignalProcessor>` and are aborted cleanly on Ctrl+C or SIGTERM.

<details>
<summary><strong>Pseudocode</strong></summary>

```rust
let mut set = tokio::task::JoinSet::new();

// ── Protocol adapters (always started) ───────────────────────────────────────
set.spawn(httpadapter::serve(processor.clone(), host, http_port));
set.spawn(wsadapter::serve(processor.clone(), host, ws_port));
set.spawn(grpcadapter::serve(processor.clone(), host, grpc_port));  // #[cfg(feature = "grpc")]
set.spawn(mcp::serve_http(processor.clone(), host, mcp_port));

// ── Memory consolidation (enabled: true by default) ───────────────────────────
if config.memory.consolidation.enabled {
    set.spawn(async move {
        let consolidator = Consolidator::new(ConsolidationConfig { prune_threshold, .. });
        let mut ticker = interval(Duration::from_secs(interval_hours * 3600));
        ticker.tick().await;   // skip first tick — don't run at startup
        loop {
            ticker.tick().await;
            let report = consolidator.consolidate(processor.episodic())?;
            promote_candidates(&processor, &report.promotion_candidates).await;
        }
    });
}

// ── Proactivity / habit engine ───────────────────────────────────────────────
if config.proactivity.enabled {
    set.spawn(async move {
        let engine = HabitEngine::new(db, habit_cfg);
        let mut ticker = interval(Duration::from_secs(min_interval_minutes * 60));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if let Some(msg) = engine.generate_proactive()? {
                router.deliver(msg.into()).await;  // outbox + broadcast + webhooks
            }
        }
    });
}

// ── Open-loop detection ──────────────────────────────────────────────────────
if config.proactivity.enabled && config.proactivity.open_loop.enabled {
    set.spawn(async move {
        let detector = OpenLoopDetector::new(db, open_loop_cfg);
        let mut ticker = interval(Duration::from_secs(check_interval_minutes * 60));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            for msg in detector.generate_reminders()? {
                router.deliver(msg.into()).await;
            }
        }
    });
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────
tokio::select! {
    _ = set.join_next() => {}   // an adapter errored
    _ = ctrl_c()        => {}   // interactive Ctrl+C
    _ = sigterm()       => {}   // `brain stop` sends SIGTERM
}
set.abort_all();
processor.shutdown();           // WAL checkpoint
```

</details>

**Default configuration:**

| Loop | Enabled by default | Interval |
|------|--------------------|----------|
| Memory consolidation | Yes | 24 hours |
| Proactivity / habit detection | Yes | `min_interval_minutes` (60) |
| Open-loop detection | Yes | `check_interval_minutes` (120) |
| Scheduled intent poller | No | 60 seconds |

---

## Action Dispatcher Backends (Internal)

<details>
<summary><strong>Traits & dispatch contract</strong></summary>

```rust
trait MemoryBackend    { async fn store_fact(ns, cat, sub, pred, obj) -> Result<String>;
                         async fn recall(query, top_k, ns) -> Result<Vec<MemoryFact>>; }
trait WebSearchBackend { async fn search(query, top_k) -> Result<Vec<SearchHit>>; }
trait SchedulingBackend{ async fn schedule(description, cron, ns) -> Result<ScheduleOutcome>; }
trait MessageBackend   { async fn send(channel, recipient, content, ns) -> Result<MessageOutcome>; }
```

**Dispatch contract (deterministic):**

| State | Result |
|-------|--------|
| Feature disabled in config | Explicit `"disabled by config"` error |
| Feature enabled, no backend wired | Explicit `"backend not configured"` error |
| Feature enabled, backend wired | Real execution with structured success output |

</details>

<details>
<summary><strong>Concrete implementations</strong></summary>

| Backend | Implementation |
|---------|---------------|
| Web search | `SearxngSearchBackend`, `TavilySearchBackend`, `CustomSearchBackend` |
| Scheduling | `CliSchedulingBackend` (SQLite persist + 60s background poller) |
| Messaging | `WebhookMessageBackend` (dual-trait — see below) |
| Memory | `CliMemoryBackend` (wraps `SemanticStore` + `Embedder`) |

</details>

<details>
<summary><strong>WebhookMessageBackend — dual-trait design</strong></summary>

`WebhookMessageBackend` implements two traits for two distinct use cases:

| Aspect | `WebhookSender` (proactive) | `MessageBackend` (explicit) |
|--------|----------------------------|----------------------------|
| Source | `signal::notification` | `cortex::actions` |
| Method | `send_notification(channel, content, namespace)` | `send(channel, recipient, content, namespace)` |
| Return | `Result<(), String>` | `Result<MessageOutcome, ActionError>` |
| Recipient | Hardcoded `""` (no recipient for proactive) | Passed through from intent |
| Response parsing | Only checks HTTP 2xx | Parses JSON for `id`/`delivery_id` + `status` |
| Triggered by | HabitEngine, OpenLoopDetector, scheduler | User: "send via discord to alice saying hi" |

Both share the same channel lookup (lowercased), template rendering, `resilient_send()` with circuit breaker, and `reqwest::Client`.

</details>

<details>
<summary><strong>Resilience layer</strong></summary>

Shared by all HTTP backends (`crates/backends/src/resilience.rs`):

- Retry with exponential backoff (`max_retries`, `retry_base_ms`) on 5xx, 429, 408, timeout, or connection errors
- Other 4xx errors fail immediately without retry (but still record a failure for circuit breaker tracking)
- Backoff capped at `retry_base_ms × 32` (exponent clamped at `min(5)`)
- `CircuitBreaker` per backend: atomic consecutive-failure counter + epoch-based cooldown; half-open probe resets counter on success
- Schema validation: structured `tracing::warn!` on unexpected response shapes — never crashes

</details>

---

## Memory Namespaces

Every fact and episode carries `namespace TEXT NOT NULL DEFAULT 'personal'`. The `namespace` field flows through every layer:

- `Signal.namespace` set by adapter from request payload
- `ActionDispatcher.set_namespace(ns)` scopes all memory operations for a session
- `SemanticStore::store_fact(namespace, ...)` and `search_similar(..., namespace: Option<&str>)`
- `EpisodicStore::store_episode(session, role, content, importance, Some(namespace))`
- Export/import preserves namespace; legacy imports without namespace default to `"personal"`

The default namespace is `"personal"`. Namespaces are a first-class schema concept, not a tag.

---

## Security Model

| Concern | Mechanism |
|---------|-----------|
| API authentication | Bearer token / `x-api-key` checked before processing on every request |
| Per-key permissions | `ApiKeyConfig { permissions: [read, write] }` — read-only keys rejected on POST |
| CORS | `localhost_cors()` — only `127.0.0.1` and `localhost` origins allowed |
| Error exposure | HTTP 500 returns opaque message; real error logged server-side only |
| Shell execution | `security.exec_allowlist` in config; configurable `exec_timeout_seconds` |
| Encryption at rest | AES-256-GCM via `brain init --encrypt` (opt-in); Argon2id key derivation. **Note:** FTS5 full-text search is disabled when encryption is active |
| LLM client failures | `Result<>` throughout — TLS failures surface as errors, never panics |
| Embedding fallback | Deterministic non-zero vectors when provider is down — writes never fail silently |

---

## Channel Integration Pattern

Brain is local and protocol-agnostic, but it now has two outward integration paths:

1. Built-in preset-driven channel transports via `channel.transports[]` for long-poll and webhook-style platforms.
2. External WebSocket gateways via `channel.relays[]` or the standalone `brain bridge` flow for custom bots and bridge processes.

<details>
<summary><strong>Design principle</strong></summary>

```
External Platform          Integration Layer                   Brain OS
──────────────────         ─────────────────                  ──────────────────────
  Telegram bot        ──►  channel.transports[] preset   ──►  SignalProcessor
  Discord webhook           polled / inbound / outbound        channel router
  Slack incoming hook       generic HTTP engines               memory + LLM
  Custom WS gateway    ──►  channel.relays[] / brain bridge ─► ws://localhost:19790
```

Preset-driven transports are built into BrainOS and configured in YAML. They cover long-poll, webhook-inbound, and webhook-outbound flows without platform-specific Rust crates.

External bridges are still valuable when the platform speaks a custom WebSocket protocol or when you want a separate process/repository to own the platform-specific bot logic.

The bridge is **not** inside Brain. It is a separate process (and typically a separate repository) that:

1. Maintains a WebSocket connection to Brain (`ws://localhost:19790`)
2. Receives inbound messages from the external platform
3. Wraps them as `{"content": "...", "sender": "...", "namespace": "..."}` and sends to Brain
4. Receives `SignalResponse` from Brain and relays the response back to the platform

`WebhookInboundTransport` is registered and health-checked, and inbound traffic flows through `POST /v1/webhooks/:id` on the HTTP adapter ([`crates/adapters/http/src/server.rs`](crates/adapters/http/src/server.rs)) into `WebhookInboundTransport::handle_request()`, which runs HMAC/Ed25519 verification before dispatching into `SignalProcessor`.

</details>

<details>
<summary><strong><code>crates/bridge/</code> library</strong></summary>

`BridgeClient` in `crates/bridge/src/lib.rs`:

```rust
pub struct BridgeClient { url: String, config: BridgeConfig }

impl BridgeClient {
    pub fn new(url: impl Into<String>, config: BridgeConfig) -> Self;

    /// Connect to the gateway and relay messages indefinitely.
    /// Reconnects with exponential backoff on disconnect.
    pub async fn connect_and_relay<F, Fut>(&self, handler: F) -> Result<(), BridgeError>
    where
        F: Fn(BridgeMessage) -> Fut + Clone,
        Fut: Future<Output = BridgeMessage>;

    /// Bidirectional: relay inbound messages AND push proactive notifications outbound.
    pub async fn connect_and_relay_bidirectional<F, Fut, N, NFut>(
        &self,
        handler: F,
        notifications: N,
    ) -> Result<(), BridgeError>
    where
        F: Fn(BridgeMessage) -> Fut + Clone,
        Fut: Future<Output = BridgeMessage>,
        N: Fn() -> NFut + Clone,
        NFut: Future<Output = Option<BridgeMessage>>;
}
```

Features:
- Exponential backoff reconnection (1s → 2s → 4s → … → 60s max)
- Ping/pong keep-alive (every 30s)
- JSON message serialization
- `BridgeMessage::reply()`, `BridgeMessage::from_json()`
- `BridgeConfig` with configurable `max_backoff_secs`, `ping_interval_secs`, `connection_timeout_secs`

</details>
