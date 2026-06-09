# Brain OS — Architecture

This document covers the internal design of Brain OS: key abstractions, data flow, storage layer, background loops, delegation and channel transport architecture, and step-by-step guides for building new protocol adapters.

---

## Table of Contents

- [Design Principle: One Capability, Many Faces](#design-principle-one-capability-many-faces)
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

## Design Principle: One Capability, Many Faces

Brain is a kernel, so "what the system can do" is modelled once and projected to every consumer. A **capability** is a typed entry — id, safety tier (read / write / destructive), preconditions, when-to-use, cost — in a single registry. The transports (CLI, HTTP, WebSocket, gRPC, MCP) and the resident reasoner (the SOUL) are all **faces** over that one registry; they hold no private capabilities and no business logic of their own. The reference implementation is the `net` family: the probe logic lives in `backends::net`, the CLI is a thin wrapper, and the capability registry dispatches to the same core — *one core, many faces*.

Two rules follow:

- **Capability vs. operator command.** If something *observes or acts on the user's world* (search, read a file, run a command, probe a host, audit config), it is a capability — a backend core plus a registered descriptor, reachable by every face. If it *manages Brain's own installation or process* (`init`, `doctor`, `service`, `update`, `start`/`stop`/`serve`, `vault`, `config`), it is an operator command and stays CLI-only by design. A kernel's syscalls don't include "reinstall the kernel."
- **Awareness is not permission.** Every descriptor carries its authorization tier. The reasoner *knows* an action needs consent and says so up front; execution still flows through the confirmation, audit, and budget gates. Knowing ≠ doing.

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
│   │                     Intent surface (grouped, ~39 variants — see `crates/thalamus/src/lib.rs`):
│   │                       memory:    StoreFact, Recall, Forget, MemorySummary
│   │                                  (path references in chat are read as attachments, not a
│   │                                   separate intent — the old ProjectInspect intent was removed)
│   │                       chat:      Chat, SystemStatus, ProactivityStatus, SetProactivity
│   │                       actions:   WebSearch, ExecuteCommand, SendMessage,
│   │                                  Schedule, ListSchedules, CancelSchedule
│   │                       audit:     QueryAudit, PruneAudit
│   │                       approvals: ListApprovals, RespondToApproval,
│   │                                  ListStandingApprovals, RevokeStandingApproval
│   │                       budget:    BudgetStatus
│   │                       tasks:     DecomposeTask, ListTasks, TaskStatus,
│   │                                  CancelTask, CancelSignal
│   │                       agents:    QueryAgents, DelegateTask
│   │                       mcp:       MountMcpServer, UnmountMcpServer, ListMcpServers,
│   │                                  ListCapabilities, ToolCall(IntentToken)
│   │                       terminal:  OpenTerminalSession, CloseTerminalSession,
│   │                                  ListTerminalSessions
│   │                       channels:  ChannelPreferences, SetChannelPreference, ListChannels
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
│   ├── observe/        # Observability bus + Observer trait
│   │                     BrainEvent enum, BroadcastObserver, Redactor.
│   │                     Audit-bus unity: every audited action emits a
│   │                     BrainEvent::AuditAppended. Drained by SSE/WS/gRPC and
│   │                     `brain tail`.
│   │
│   ├── identity/       # Principal, tier, authorization for signals
│   │                     IdentityStore trait + ConfigIdentityStore.
│   │                     Principal threaded through Signal; pipeline's
│   │                     enforce_identity gate runs after classification.
│   │                     authz::intent_to_auth maps Intent → AuthorizationRequest.
│   │
│   ├── intent/         # Standardized Intent Token (SIT) + capability routing
│   │                     IntentToken schema, ToolRegistry + IntentRouter traits,
│   │                     CapabilityIndex with hybrid scoring.
│   │
│   ├── mcphost/        # MCP host — mounts external MCP servers
│   │                     MCPHost/MCPClient traits, ServerConfig
│   │                     (Stdio/StreamableHttp/HttpSse), InMemoryMcpHost stub,
│   │                     rmcp_host (real client), CapabilityIndex, OAuth helper,
│   │                     ResilientMcpHost wrapper.
│   │
│   ├── reflex/         # Reactive signal sources
│   │                     ReflexSource trait + FsReflex, CronReflex, SysStateReflex,
│   │                     CompositeReflex, NoopReflex. Triggers emit Signals;
│   │                     they never execute. Standing approvals declared,
│   │                     visible, revocable.
│   │
│   ├── resilience/     # Resilience primitives
│   │                     Hystrix circuit breaker (Closed/Open/HalfOpen),
│   │                     retry with exponential backoff, rate limit, timeout,
│   │                     loop detector, DLQ. BreakerRegistry for per-tool
│   │                     breakers wired by intent router.
│   │
│   ├── storage/        # Storage abstraction layer
│   │   ├── sqlite      # SqlitePool: 20 migrations through v22, WAL mode,
│   │   │                 thread-safe Mutex<Connection>
│   │   │                 Tables: semantic_facts, episodes, procedures, scheduled_intents,
│   │   │                 _migrations, FTS5 virtual tables (episodes_fts),
│   │   │                 dlq_entries (v19), graph nodes/edges (v20),
│   │   │                 standing_approvals (v21), task_states (v22).
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
│       ├── mcp/        # MCP adapter (stdio transport + HTTP transport, port 19791)
│       │                 6 tools: memory_search, memory_store, memory_facts,
│       │                 memory_episodes, user_profile, memory_procedures
│       │                 JSON-RPC 2.0, meta-key auth
│       └── terminal/   # Terminal Bridge gRPC adapter (port 19793, tonic)
│                         TerminalBridge + SessionRegistry, portable-pty backed.
│                         TerminalSvc Open/Close/Attach/Send/Resize/Signal/Interact.
│                         PTY reader pump → broadcast<Bytes>(256); Attach emits
│                         OutputChunk{eof=true} on RecvError::Closed.
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

31 crates total:

```
core  storage  hippocampus  cortex  thalamus  amygdala  signal
adapters/http  adapters/ws  adapters/grpc  adapters/mcp  adapters/terminal
cerebellum  ganglia  bridge  backends
audit  confirm  budget  sandbox  vault
orchestrate  delegate  channel
observe  identity  intent  mcphost  reflex  resilience
cli
```

### Dependency Graph

```
reflex (cron/fs/sysstate/composite) ─► emits Signal{provenance: Reflex}
                                             │
adapters/http  ──┐                           │
adapters/ws    ──┼─► Arc<SignalProcessor> ◄──┘
adapters/grpc  ──┤        │
adapters/mcp   ──┤        │
adapters/terminal ┘       │
                          │
                          ├── identity        (Principal gate after classification)
                          ├── thalamus        (intent classification — regex fast-path + LLM)
                          ├── intent          (SIT schema + IntentRouter + CapabilityIndex)
                          ├── mcphost         (external MCP servers — stdio/HTTP/SSE)
                          ├── amygdala        (importance scoring)
                          ├── hippocampus     (memory read / write / consolidation + graph)
                          │       └── storage (SQLite + ruvector-core HNSW + AES-GCM)
                          ├── cortex          (LLM providers + context assembly + actions)
                          ├── cerebellum      (procedure store + trigger matching)
                          ├── ganglia         (proactivity / habit engine)
                          ├── orchestrate     (task planning / execution / state machine)
                          ├── delegate        (external agent discovery / delegation)
                          ├── channel         (routing / preferences / transports)
                          ├── resilience      (breakers / retry / DLQ — wraps backends)
                          └── observe         (BrainEvent bus — taps every checkpoint)

External apps  ──► Brain's HTTP / WS / MCP / gRPC / Terminal API
```

All adapters share **one** `Arc<SignalProcessor>`. There are no per-adapter memory stores. A fact stored via MCP is immediately visible via HTTP or gRPC. The Terminal adapter is a motor-cortex adapter — it executes PTY sessions and mirrors lifecycle events back into the episodic graph via `signal::terminal_graph_mirror`.

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
     │     Schedule    → ActionDispatcher::schedule_task (SQLite persist; fired by CronReflex)
     │     SendMessage → ActionDispatcher::send_message (webhook POST with template)
     │     Command     → ActionDispatcher::execute_command (allowlist + timeout)
     │
     ├─ 5. publish_event → broadcast::Sender<SignalProcessedEvent>
     │         (consumed by gRPC AgentService::ReceiveSignals fan-out stream)
     │
     └─ 6. Return SignalResponse { signal_id, status, response, memory_context }
```

The `SignalResponse` is returned directly to the calling adapter, which sends it back in the protocol-appropriate format.

### Two routing planes

Classification feeds **two distinct routing planes**, and they have opposite dynamism properties — keeping them separate is deliberate:

- **Control plane** — Brain's own fixed verbs (memory ops, schedules, tasks, approvals, budget, channels, terminal/MCP lifecycle). These live in the closed `Intent` enum and are matched by the regex fast-path + the LLM fallback's static taxonomy. This is the kernel's *syscall table*: it is fixed by design, not discovered at runtime. (The LLM fallback covers the natural-language subset of these; the lifecycle/terminal/MCP/channel verbs are reached via deterministic `/slash` forms and narrow regexes.)
- **Capability plane** — open-ended tool/agent invocation. Routed as `Intent::ToolCall(IntentToken)` → SIT → `CapabilityIndex` (semantic hybrid scoring) → `ToolRegistry`. This is fully **runtime-dynamic**: mounted MCP servers, native backends, and registered agents populate the index on mount and are retrieved top-k by relevance. There are exactly two `ToolCall` producers by design — the explicit `/tool <ns>.<action>` form and the chat **tool-loop** (the reasoner proposes calls in-band with full schemas). The classifier deliberately does **not** guess tools from free text; that belongs to the tool-loop.

The resident reasoner (SOUL) is grounded against the **capability plane** every turn via a live capability digest + product self-model (`signal/pipeline/conversation.rs`). The classifier prompt is *not* yet fed that live manifest — it reasons about the fixed control-plane taxonomy only.

---

## Key Types

<details>
<summary><strong>Signal</strong> — universal input envelope</summary>

```rust
pub struct Signal {
    pub id: Uuid,
    pub source: SignalSource,            // Cli | Http | WebSocket | Mcp | Grpc | Terminal | Reflex
    pub channel: String,
    pub sender: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub namespace: String,               // default: "personal"
    pub agent: Option<String>,           // originating AI agent id, when known
    pub session_id: Option<String>,      // when provided, reuse this session
    pub principal: Option<Principal>,    // resolved by the adapter from its auth
                                         // context; consulted by the pipeline's
                                         // identity gate after intent classification
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
    pub fn with_channel_dispatcher(self, dispatcher: Arc<channel::ChannelDispatcher>) -> Self;

    // Observability, identity, motor cortex, capability routing, resilience
    pub fn with_observer(self, observer: Arc<dyn observe::Observer>) -> Self;
    pub fn with_identity_store(self, store: Arc<dyn identity::IdentityStore>) -> Self;
    pub fn with_terminal_bridge(self, bridge: Arc<terminal::TerminalBridge>) -> Self;
    pub fn with_mcp_host(self, host: Arc<dyn mcphost::MCPHost>) -> Self;
    pub fn with_tool_registry(self, registry: Arc<dyn intent::ToolRegistry>) -> Self;
    pub fn with_intent_router(self, router: Arc<dyn intent::IntentRouter>) -> Self;
    pub fn with_breaker_registry(self, registry: Arc<resilience::BreakerRegistry>) -> Self;
    pub fn with_standing_approvals(self, approvals: Arc<confirm::StandingApprovalStore>) -> Self;
    pub fn with_confirmation_timeout(self, t: std::time::Duration) -> Self;

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

Migration-based schema — 20 migrations through v22. WAL mode enabled. Thread safety via `Mutex<Connection>`. Individual crates that own private tables (e.g. `audit`, `confirm`, `ganglia`) provision them via their own `ensure_tables()` rather than through the shared migrations file; the version numbers below refer to the central `crates/storage/src/sqlite/migrations.rs` log only.

<details>
<summary><strong>Tables</strong></summary>

| Table | Purpose | Source |
|-------|---------|--------|
| `sessions` | Chat session tracking (id, channel, namespace, timestamps) | v1 |
| `episodes` | Conversation history with role, importance, decay_rate, reinforcement_count | v2 |
| `episodes_fts` | FTS5 virtual table (BM25 full-text search over episode content) | v3 |
| `semantic_facts` | S-P-O triples with namespace, importance, source_episode_id | v4–v6 |
| `user_profile` | Key-value store for user preferences | v7 |
| `procedures` | trigger_pattern → steps_json automation rules | v10–v12 |
| `scheduled_intents` | Persisted scheduling intents | v13–v15 |
| `notification_outbox` | Proactive notification queue with priority and delivery status | v16 |
| `episode_promotions` | Idempotency log for episode → semantic-fact promotions | v17 |
| `audit_log` | Action audit trail (action type, input, output, timestamps) | v18 |
| `dlq_entries` | Dead-letter queue for failed/permanently-broken actions | **v19** |
| `nodes`, `edges` | Episodic graph — recursive provenance trace, half-life decay | **v20** |
| `standing_approvals` | Declared, revocable bypass rules for the confirmation engine | **v21** |
| `task_states` | TaskOrchestrator state-machine history | **v22** |
| `habit_state` | Rate-limit state for proactivity engine | crate-managed (`ganglia::ensure_tables`) |
| `audit_entries` | Append-only entries with `principal_json` (identity round-trip) | crate-managed (`audit::ensure_tables`) |
| `_migrations` | Applied migration version log | always present |

Versions 8 and 9 were retired during early development; the migrator skips them. Schema reads are idempotent (`IF NOT EXISTS` throughout).

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
| Cron reflex (scheduled-intent firing) | `reflex.cron.enabled` | per-schedule cron tick |

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
| Per-key permissions | `ApiKeyConfig { permissions: [read, write, export, admin] }` — `admin` is a superset; `export` gates bulk memory export; `write` does NOT imply `read` |
| CORS | `localhost_cors()` — only `127.0.0.1` and `localhost` origins allowed (exact match, no prefix tricks) |
| Error exposure | HTTP 500 returns opaque message; real error logged server-side only |
| Shell execution | `security.exec_allowlist` in config; configurable `exec_timeout_seconds`; deny-list extends to `kill`, `python`, `node`, `nc` family by default |
| Encryption at rest | AES-256-GCM via `brain init --encrypt` (opt-in); Argon2id key derivation (OWASP 2024: 46 MiB / t=1). **Note:** FTS5 full-text search is disabled when encryption is active |
| Vault key material | Derived keys + passphrase strings wrapped in `Zeroizing` so they scrub on drop |
| Vault passphrase | Stdin / `passphrase_file` only; env var ignored with a startup warning (env vars leak via `/proc/<pid>/environ` and `ps -e`) |
| Outbound URL fetch | LLM-controlled URL fetcher rejects loopback / private / link-local / cloud-metadata IPs at both literal-host and DNS-resolved layers |
| LLM client failures | `Result<>` throughout — TLS failures surface as errors, never panics |
| Embedding fallback | Deterministic non-zero vectors when provider is down — writes never fail silently |

### Docker socket containment

Brain's `IsolatedSandbox` does not bind, expose, or read `/var/run/docker.sock`
under any built-in configuration. The sandbox runs each `Action::ExecuteCommand`
with `setrlimit`, the configured `exec_allowlist` (binary names), and the
`forbidden_commands` deny-list — none of which add docker to the visible
PATH or interact with the docker daemon.

If you mount `docker.sock` into a container that hosts Brain, **you have
granted that Brain process root-equivalent control of the host**: any
command the sandbox approves (e.g. `cat`, `ls`) becomes a primitive for
`docker run --privileged -v /:/host ...` once an attacker can shell out.
This is a property of the docker socket itself, not of Brain.

Recommended containment when Brain is running inside docker:

- **Do not** mount `/var/run/docker.sock` into the Brain container.
- If you must (e.g. for a separate orchestration agent), run Brain in a
  sibling container and have it talk to the orchestrator over an
  authenticated network socket instead — never share the docker socket.
- Add `docker`, `docker-compose`, `nerdctl`, `podman` to
  `security.exec_allowlist` only when you understand the full effect
  and have a dedicated rootless container runtime configured.

The Brain default `exec_allowlist` ships without docker/podman/nerdctl
on purpose. Adding them is a deliberate opt-in.

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



---

## Extending Brain

Brain exposes four extension points today. Each is reachable without modifying the kernel, and every one routes through the same shared `SignalProcessor` and the same identity, consent, audit, and budget gates.

| Contract | Direction | What plugs in | Wire format |
|----------|-----------|---------------|-------------|
| **Adapters** | apps → Brain | LLM agents, IDE extensions, custom tools | HTTP / WebSocket / gRPC / MCP |
| **Capabilities** | Brain → tools | external tools, system actions | MCP servers (stdio / Streamable HTTP / SSE), Terminal Bridge (PTY), native backends |
| **Agents** | Brain → AIs | Claude Code, Cursor, Aider, and other subprocess agents | `AgentDelegate` trait + subprocess transport |
| **Channels** | Brain ↔ humans | any HTTP/WebSocket gateway (Telegram, Discord, Slack, custom) | `channel.transports[]` (polled / webhook-in / webhook-out) + Bridge library |

- **Adapters** let any application talk to the one shared memory + reasoning engine over a standard protocol — no per-app memory, no special casing. A fact stored via MCP is immediately visible via HTTP.
- **Capabilities** are how Brain reaches outward. Mount an MCP server and its tools auto-register into the capability manifest; the Terminal Bridge and native backends register the same way and carry the same safety tiers.
- **Agents** hand a subtask to a specialist CLI/HTTP agent through a single subprocess-backed delegate, discovered from `PATH` or `~/.brain/agents.yaml`.
- **Channels** carry messages to and from humans through three generic transport kinds (`http_polled`, `webhook_inbound`, `webhook_outbound`) plus the Bridge relay library. Brain ships no platform-specific code; platforms are expressed entirely as YAML presets.

See [Channel Integration Pattern](#channel-integration-pattern) for the bridge-relay design in detail.
