# Brain OS — Architecture

This document covers the internal design of Brain OS: key abstractions, data flow, storage layer, background loops, the bridge relay pattern for external integrations, and step-by-step guides for building new protocol adapters.

---

## Crate Map

```
brain/
├── crates/
│   ├── core/           # BrainConfig + loader — shared config types used by every crate
│   │
│   ├── signal/         # Signal / SignalResponse / SignalProcessedEvent types
│   │                     SignalAdapter trait
│   │                     SignalProcessor — the single shared engine that wires all subsystems
│   │
│   ├── thalamus/       # Intent classification
│   │                     Regex fast-path (compiled at startup) + async LLM fallback with timeout
│   │                     8 intent types: StoreFact, Recall, Forget, Chat,
│   │                     WebSearch, Schedule, SendMessage, ExecuteCommand
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
│   │   │                 Streaming and non-streaming generate; health_check
│   │   ├── context     # ContextAssembler: token-budgeted prompt builder
│   │   │                 Budget: system(500) + user_model(300) + history(2000) +
│   │   │                 response_buffer(400) + memories(remainder of 8192)
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
│   ├── storage/        # Storage abstraction layer
│   │   ├── sqlite      # SqlitePool: 16 migrations, WAL mode, thread-safe Mutex<Connection>
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
│       ├── grpc/       # gRPC adapter (port 19792, tonic)
│       │                 MemoryService (Search, Store, GetFacts, StreamSignals)
│       │                 AgentService (Connect, SendSignal, ReceiveSignals fan-out)
│       │                 Auth interceptor, namespace propagation
│       └── mcp/        # MCP adapter (stdio transport + HTTP transport, port 19791)
│                         6 tools: memory_search, memory_store, memory_facts,
│                         memory_episodes, user_profile, memory_procedures
│                         JSON-RPC 2.0, meta-key auth
│
├── crates/cli/         # `brain` binary — all CLI commands:
│                         init, chat, status, start, stop, serve, mcp,
│                         export, import, service install/uninstall, deps up/down/status
│                         Contains: BrainSession, all action backend impls
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

### Workspace Members (`Cargo.toml`)

```
core  storage  hippocampus  cortex  thalamus  amygdala  signal
adapters/http  adapters/ws  adapters/grpc  adapters/mcp
cerebellum  ganglia  bridge  cli
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
            └── ganglia         (proactivity / habit engine)

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
     │         Regex fast-path first → async LLM fallback (timeout-bounded) if no match
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
     │     Schedule    → ActionDispatcher::schedule_task (SQLite persist-only)
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

### `Signal` (`crates/signal/src/lib.rs`)

The universal input envelope — every adapter builds one of these before calling `SignalProcessor::process`.

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
    pub agent: Option<String>,   // originating AI agent (e.g. "claude-code")
}
```

### `SignalResponse`

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

### `SignalAdapter` trait

The interface every protocol adapter implements. It is thin by design — adapters translate wire formats, not business logic.

```rust
#[async_trait]
pub trait SignalAdapter: Send + Sync {
    fn source(&self) -> SignalSource;
    async fn send(&self, response: SignalResponse) -> Result<(), SignalError>;
}
```

### `SignalProcessor`

Constructed once at startup and shared via `Arc<>`:

```rust
impl SignalProcessor {
    pub async fn new(config: BrainConfig) -> Result<Self, SignalError>;
    pub async fn new_with_encryptor(config: BrainConfig, encryptor: Option<Encryptor>)
        -> Result<Self, SignalError>;

    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError>;

    // Direct memory operations — used by adapters that bypass intent classification
    pub async fn store_fact_direct(&self, ns: &str, cat: &str, sub: &str,
        pred: &str, obj: &str) -> Result<String, SignalError>;
    pub async fn search_facts(&self, query: &str, top_k: usize,
        namespace: Option<&str>) -> Vec<SemanticResult>;

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

---

## Storage Layer

### SQLite (`crates/storage/src/sqlite.rs`)

Migration-based schema versioned in a `MIGRATIONS` slice. The runner compares `MAX(version)` in the `_migrations` table against the max in-code version and runs missing migrations in order.

**Tables:**

| Table | Purpose |
|-------|---------|
| `sessions` | Chat session tracking (id, channel, namespace, timestamps) |
| `semantic_facts` | S-P-O triples with namespace, importance, source_episode_id |
| `episodes` | Conversation history with role, importance, decay_rate, reinforcement_count |
| `episodes_fts` | FTS5 virtual table (BM25 full-text search over episode content) |
| `user_profile` | Key-value store for user preferences |
| `procedures` | trigger_pattern → steps_json automation rules |
| `audit_log` | Action audit trail (action type, input, output, timestamps) |
| `scheduled_intents` | Persisted scheduling intents (persist-only mode) |
| `episode_promotions` | Idempotency log for episode → semantic-fact promotions |
| `notification_outbox` | Proactive notification queue with priority and delivery status |
| `habit_state` | Rate-limit state for proactivity engine (daily count, last sent) |
| `_migrations` | Applied migration version log |

**WAL mode** is enabled for concurrent reads alongside writes.  
**Thread safety** is via `Mutex<Connection>` — one connection, one writer at a time.  
**Encryption** is opt-in: `SqlitePool::with_encryptor(enc)` wraps the pool so `encrypt_content` / `decrypt_content` are called transparently on write/read of content columns.

### Vector Index (`crates/storage/src/ruvector.rs`)

Brain uses [`ruvector-core`](https://crates.io/crates/ruvector-core) (crates.io) as an external dependency for HNSW approximate nearest-neighbour search. The wrapper in `storage/src/ruvector.rs` provides a multi-table interface.

> All vector logic lives in `storage/src/ruvector.rs` using `ruvector-core` from crates.io.

**Storage layout:**

```
~/.brain/ruvector/
  facts_vec.db      # HNSW index for semantic fact vectors
  episodes_vec.db   # HNSW index for episode vectors (future use)
```

**Robustness guarantees before any insert or search:**

1. Dimension check — vectors with wrong size get a deterministic fallback
2. Finite check — NaN / Inf values trigger deterministic fallback
3. Zero-norm check — zero vectors get a deterministic fallback
4. L2 normalization applied to all vectors
5. Deterministic per-ID jitter applied on insert to avoid pathological duplicate-distance panics in HNSW

The `deterministic_fallback_embedding(seed, dimensions)` function (FNV-1a hash → xorshift64* PRNG → normalized) ensures that even when the embedding provider is down, all memory writes succeed with meaningful (though semantically approximate) vectors.

### Hybrid Search (`crates/hippocampus/src/search.rs`)

`RecallEngine::recall()` pipeline:

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

RRF correctly handles overlap boosting (items appearing in both lists score higher) and disjoint lists, with full unit test coverage.

---

## Background Loop Architecture

`brain serve` / `brain start` spawns adapter tasks and optional intelligence tasks into a single `tokio::task::JoinSet`. All tasks share `Arc<SignalProcessor>` and are aborted cleanly on Ctrl+C or SIGTERM.

```rust
// Pseudocode of what brain serve spawns
let mut set = tokio::task::JoinSet::new();

// ── Protocol adapters (always started) ───────────────────────────────────────
set.spawn(httpadapter::serve(processor.clone(), host, http_port));
set.spawn(wsadapter::serve(processor.clone(), host, ws_port));
set.spawn(grpcadapter::serve(processor.clone(), host, grpc_port));
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

// ── Proactivity / habit engine (enabled: false by default) ───────────────────
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

// ── Open-loop detection (enabled: true under proactivity) ────────────────────
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

**Default configuration:**

| Loop | Enabled by default | Interval |
|------|--------------------|----------|
| Memory consolidation | Yes | 24 hours |
| Proactivity / habit detection | No (opt-in) | `min_interval_minutes` (60) |
| Open-loop detection | No (opt-in, under proactivity) | `check_interval_minutes` (120) |

---

## Action Dispatcher Backends (Internal)

`cortex::ActionDispatcher` supports pluggable backend traits. All action execution remains internal — no new public HTTP or gRPC endpoints expose action dispatch in this cycle.

**Traits:**

```rust
trait MemoryBackend    { async fn store_fact(..) / async fn recall(..) }
trait WebSearchBackend { async fn search(query, top_k) -> Vec<SearchHit> }
trait SchedulingBackend{ async fn schedule(description, cron, namespace) -> ScheduleOutcome }
trait MessageBackend   { async fn send(channel, recipient, content, namespace) -> MessageOutcome }
```

**Dispatch contract (deterministic):**

| State | Result |
|-------|--------|
| Feature disabled in config | Explicit `"disabled by config"` error — never silently ignored |
| Feature enabled, no backend wired | Explicit `"backend not configured"` error |
| Feature enabled, backend wired | Real execution with structured success output |

**Concrete implementations in `crates/cli/src/main.rs`:**

| Backend | Implementation |
|---------|---------------|
| Web search | `SearxngSearchBackend`, `TavilySearchBackend`, `CustomSearchBackend` |
| Scheduling | `CliSchedulingBackend` (SQLite persist-only) |
| Messaging | `WebhookMessageBackend` (configurable channel → webhook URL + body template) |
| Memory | `CliMemoryBackend` (wraps `SemanticStore` + `Embedder`) |

**Resilience layer** (shared by all HTTP backends):

- Retry with exponential backoff (`max_retries`, `retry_base_ms`) on 5xx / timeout / connection-refused
- 4xx errors fail immediately without retry
- `CircuitBreaker` per backend: atomic consecutive-failure counter + epoch-based cooldown; half-open probe after cooldown elapses
- Schema validation: structured `tracing::warn!` on unexpected response shapes — never crashes

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
| Encryption at rest | AES-256-GCM via `brain init --encrypt` (opt-in); Argon2id key derivation |
| LLM client failures | `Result<>` throughout — TLS failures surface as errors, never panics |
| Embedding fallback | Deterministic non-zero vectors when provider is down — writes never fail silently |

---

## Bridge Pattern (External Gateway Relay)

Brain is local and protocol-agnostic. It does not reach outward to any external platform. External applications that live on messaging platforms (Slack, Telegram, Discord, custom agents) connect **inward** to Brain via its standard protocols.

### Design Principle

```
External Platform          Bridge (external repo)              Brain OS
──────────────────         ───────────────────────             ─────────────────
  Slack bot           ──►  thin relay process           ──►   ws://localhost:19790
  Telegram bot              uses crates/bridge library         SignalProcessor
  Custom chat agent         translates platform format         memory + LLM
  Any WebSocket bot         reconnects automatically
```

The bridge is **not** inside Brain. It is a separate process (and typically a separate repository) that:

1. Maintains a WebSocket connection to Brain (`ws://localhost:19790`)
2. Receives inbound messages from the external platform
3. Wraps them as `{"content": "...", "sender": "...", "namespace": "..."}` and sends to Brain
4. Receives `SignalResponse` from Brain and relays the response back to the platform

### `crates/bridge/` Library

`BridgeClient` in `crates/bridge/src/lib.rs` is a ready-to-use Rust client for building these relays:

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
    pub async fn connect_and_relay_bidirectional<F, Fut>(
        &self, handler: F,
        proactive_rx: Option<broadcast::Receiver<BridgeMessage>>,
    ) -> Result<(), BridgeError>;
}

pub struct BridgeConfig {
    pub initial_backoff_ms: u64,          // default: 1 000ms
    pub max_backoff_ms: u64,              // default: 60 000ms
    pub max_reconnect_attempts: Option<u32>, // default: None (reconnect forever)
}
```

`BridgeMessage` is a simple JSON-serializable envelope `{ id, content, source?, metadata? }`. The client handles ping/pong keep-alive, clean disconnect detection, and backoff.

### Minimal bridge example (external project)

```rust
// In YOUR relay project — not inside Brain OS
use bridge::{BridgeClient, BridgeConfig, BridgeMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = BridgeClient::new(
        "ws://your-external-gateway.example.com/brain-relay",
        BridgeConfig::default(),
    );

    client.connect_and_relay(|msg| async move {
        // Forward to Brain's WebSocket, get response, relay back
        let brain_reply = send_to_brain_ws(&msg.content).await;
        BridgeMessage::reply(&msg, brain_reply)
    }).await?;

    Ok(())
}
```

### Why this design is correct

- Brain has **zero** platform-specific code
- Adding support for a new platform requires **no changes** to Brain
- The bridge reconnects automatically — Brain never needs to know the bridge was disconnected
- All Brain protocols (HTTP, WS, gRPC, MCP) are available to the bridge; WebSocket is the natural fit for bidirectional relay



---

## Integrating External Applications

Brain is protocol-agnostic. External applications connect via its standard interfaces — they do not live inside this repository.

```
Brain OS (this repo)                  External App (separate repo / process)
────────────────────                  ──────────────────────────────────────
  brain serve                           OpenCode, Claude Code, shell script, etc.
       │                                        │
       │  HTTP REST / WS / MCP / gRPC           │
       │◄────────────────────────────────────── │
       │                                        │
  SignalProcessor                        app-specific logic
  hippocampus                            thin Brain client (HTTP calls)
```

### Integration Patterns

**1. HTTP REST (simplest — any language, no SDK needed)**

```bash
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"source":"myapp","sender":"agent","content":"user prefers tabs over spaces"}'
```

**2. MCP (for AI coding assistants)**

AI agents that speak MCP declare Brain as a server and call `memory_search`, `memory_store`, etc. as native tools — no HTTP client code needed:

```json
{
  "mcpServers": {
    "brain": { "command": "brain", "args": ["mcp"] }
  }
}
```

**3. WebSocket (real-time / streaming)**

Connect to `ws://localhost:19790`, authenticate with first frame, then send signal payloads:

```json
{ "api_key": "your-key" }
```

Then:

```json
{ "content": "what do I know about Rust?", "namespace": "work" }
```

**4. gRPC (high-throughput / typed clients)**

Generate client stubs from the proto files in `crates/adapters/grpc/proto/`:
- `memory.proto` — `MemoryService` (Search, Store, GetFacts, StreamSignals)
- `agent.proto` — `AgentService` (Connect, SendSignal, ReceiveSignals)

`AgentService.ReceiveSignals` is a live server-streaming RPC: subscribers receive an initial `connected` event, then fan-out updates for every signal processed by `SignalProcessor`.

### Minimal Python client (no SDK needed)

```python
import requests

class BrainClient:
    def __init__(self, api_key, base="http://localhost:19789"):
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {api_key}"
        self.base = base

    def remember(self, text, namespace="personal"):
        self.s.post(f"{self.base}/v1/signals",
            json={"source": "python", "content": text, "namespace": namespace})

    def search(self, query, top_k=5, namespace=None):
        body = {"query": query, "top_k": top_k}
        if namespace:
            body["namespace"] = namespace
        return self.s.post(f"{self.base}/v1/memory/search", json=body).json()
```

A Rust client crate or Python/JS SDK can be published separately as the API stabilises.

---

## Building a New Protocol Adapter

### What an adapter is

An adapter is a **protocol transport layer** — it listens on a socket (or stdin/stdout), authenticates the caller, translates the wire format into a `Signal`, calls the shared `SignalProcessor`, and sends `SignalResponse` back in the protocol-appropriate format.

Brain ships four protocol adapters: HTTP REST, WebSocket, gRPC, and MCP (stdio + HTTP). Adding a new adapter means adding a new _transport_, not a new platform integration.

```
External apps never live inside Brain.
They connect to Brain using an existing protocol:

  Your script      ──── HTTP ────► Brain
  Your agent       ──── MCP  ────► Brain
  Any chat UI      ──── WS   ────► Brain
  Any gRPC client  ──── gRPC ────► Brain
  External relay   ──── WS   ────► Brain (via crates/bridge)
```

If you want to connect a messaging platform, CLI tool, or AI agent to Brain, call Brain's existing HTTP/WS/MCP/gRPC API from that app — you do not add a "Slack adapter" or "Telegram adapter" inside Brain.

### When to add a new protocol adapter

Add a new adapter when you need a transport that Brain doesn't yet speak — for example Unix domain sockets, AMQP, or a custom binary protocol.

### SSE (Server-Sent Events) — Already Implemented

SSE is built into the HTTP adapter and provides one-directional streaming (server pushes, client reads) for proactive notifications. It's ideal for browser clients that want to receive real-time nudges without maintaining a WebSocket connection.

**Endpoint:** `GET /v1/events`

```bash
# Authenticated SSE stream
curl -N http://localhost:19789/v1/events \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response format:**
```json
event: notification
data: {"type":"proactive","content":"You usually work on \"auth\" around this time...","triggered_by":"habit:auth","priority":1,"agent":null}
```

**Implementation:**
- Route registered in `crates/adapters/http/src/lib.rs`
- Handler at `sse_events_handler`
- Subscribes to `NotificationRouter::subscribe()` broadcast channel
- Streams proactive notifications as SSE events
- Supports keep-alive and lagged client detection

The SSE endpoint is used by:
- Browser-based clients that want real-time proactive notifications
- `brain chat` for displaying nudges at session start

**Key insight:** all adapters follow the same contract: receive input → build `Signal` → call `processor.process()` → return output. The `SignalProcessor` is shared by `Arc` so memory is consistent regardless of which adapter handled the request.

---

## LLM Providers

The `LlmProvider` trait in `crates/cortex/src/llm.rs`:

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError>;
    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError>;
    async fn health_check(&self) -> bool;
    fn name(&self) -> &str;
}
```

**Implementations:**

| Provider | Type | Notes |
|----------|------|-------|
| `OllamaProvider` | Local | Calls `POST /api/chat` on the local Ollama server |
| `OpenAiProvider` | Remote / local | Any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, LM Studio) |

`create_provider(config: &ProviderConfig)` selects the implementation based on `llm.provider` in config. New providers implement `LlmProvider` and register in `create_provider()`.

Both providers support:
- Non-streaming `generate()` — waits for full response
- Streaming `generate_stream()` — returns a `Pin<Box<dyn Stream<Item = ResponseChunk>>>` for real-time output
- `health_check()` — used by `brain status`

---

## Configuration System

Config is loaded with [Figment](https://docs.rs/figment) in priority order:

```
Environment variables   BRAIN_LLM__MODEL=gpt-4o
        ↓ override
~/.brain/config.yaml    (user overrides — created by `brain init`)
        ↓ override
config/default.yaml     (compiled-in defaults via include_str!)
```

The `BrainConfig` struct in `crates/core/src/config.rs` maps 1-to-1 with the YAML keys. Double-underscore (`__`) is the env-var nesting separator (e.g. `BRAIN_ACTIONS__WEB_SEARCH__ENABLED=false`).

`BrainConfig::validate()` runs before `brain serve` and `brain start`. It returns:
- `Err(String)` for hard errors that must block startup (port conflicts, invalid LLM URL)
- `Ok(Vec<String>)` for soft warnings printed to stderr (demo key in use, zero timeout, etc.)

---

## Proactivity Engine (`crates/ganglia`)

`HabitEngine` detects recurring patterns in episodic memory using keyword × day-of-week × hour histograms and generates proactive suggestions. `OpenLoopDetector` scans for unresolved commitments and generates reminders. Both are **disabled by default** (`proactivity.enabled: false`).

When enabled, `brain serve` spawns background tasks on the `JoinSet`:
- **HabitEngine** fires every `min_interval_minutes`. If a recurring pattern matches the current time slot and all rate limits pass, it delivers through the `NotificationRouter` (outbox + broadcast + webhooks).
- **OpenLoopDetector** fires every `check_interval_minutes`. It scans for commitment phrases ("I need to", "remind me to", "I should", etc.) in episodic memory and checks whether a later episode resolves the commitment. Unresolved items older than `resolution_window_hours` trigger a reminder.

Both engines share the `NotificationRouter` for delivery and the `HabitEngine` rate limits (`max_per_day`, `min_interval_minutes`, quiet hours).

Rate-limit state (`last_sent`, `daily_count`) is persisted in a `habit_state` SQLite table, so limits survive daemon restarts.

At `brain chat` session start, pending outbox notifications are drained and displayed as nudges. Open-loop detection also runs inline for immediate feedback.

> **Note:** Quiet hours are evaluated in UTC. If you are not in UTC, adjust `start`/`end` by your UTC offset.

---

## Memory Consolidation (`crates/hippocampus/consolidation`)

`Consolidator` prunes low-retention episodes using the forgetting curve and promotes reinforced episodes to permanent semantic facts. **Enabled by default** (`memory.consolidation.enabled: true`, interval: 24 hours).

**Pipeline per run:**

1. Fetch all episodes ordered by importance ASC (up to `max_prune_per_run × 2`)
2. For each: compute `retention = importance × e^(−decay_rate × hours_since_last_access)`
3. If `retention < prune_threshold` (default: 0.05): `DELETE FROM episodes WHERE id = ?`
4. If `reinforcement_count >= promotion_threshold` (default: 3): add to `promotion_candidates`
5. Return `ConsolidationReport { episodes_pruned, episodes_promoted, episodes_remaining, promotion_candidates }`

The daemon loop in `brain serve` calls `promote_candidates()` after each consolidation run, which parses each candidate's content with `IntentClassifier::parse_store_fact_content()`, stores it as a semantic fact, and records the promotion in `episode_promotions` for idempotency.

---

## Procedure Store (`crates/cerebellum`)

`ProcedureStore` stores trigger-pattern → `steps_json` automation rules. When a signal arrives, `SignalProcessor` calls `procedures.match_trigger(&signal.content)` (case-insensitive substring match). Matching procedures have their steps injected into the LLM context as synthetic prior messages and their `use_count` incremented.

**API:**

```rust
impl ProcedureStore {
    fn store_procedure(trigger: &str, steps: &[String]) -> Result<String, Result<(), SignalError>;  // returns id
    fn match_trigger(input: &str)  -> Result<Vec<Procedure>>;
    fn get_procedure(id: &str)     -> Result<Procedure>;
    fn list_procedures()           -> Result<Vec<Procedure>>;
    fn update_steps(id, new_steps) -> Result<()>;
    fn delete_procedure(id: &str)  -> Result<()>;
    fn record_execution(id: &str)  -> Result<()>;
    fn count()                     -> Result<i64>;
}
```

Managed via the MCP `memory_procedures` tool (`list` / `store` / `delete` actions).
