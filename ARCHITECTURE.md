# Brain OS — Architecture

This document covers the internal design of Brain OS, the key abstractions, data flow, and a step-by-step guide for building a new adapter.

---

## Crate Map

```
brain/
├── crates/
│   ├── core/           # BrainConfig, loader — shared types used by all crates
│   ├── signal/         # Signal / SignalResponse types + SignalAdapter trait
│   │                     SignalProcessor (the single shared engine)
│   ├── thalamus/       # Intent classification (regex fast-path + optional LLM fallback)
│   ├── amygdala/       # Importance scoring (keyword heuristics)
│   ├── hippocampus/    # Memory: EpisodicStore, SemanticStore, RecallEngine,
│   │                     Embedder (Ollama / OpenAI-compatible), ImportanceScorer
│   ├── cortex/         # LLM clients: OllamaProvider, OpenAiProvider
│   │                     ContextAssembler + ActionDispatcher (pluggable backends)
│   ├── cerebellum/     # ProcedureStore — trigger-pattern → steps automation
│   ├── ganglia/        # Proactivity engine (scheduled reminders)
│   ├── ruvector/       # (not a crate — ruvector-core is used as an external dep)
│   ├── storage/        # SQLite migrations + ruvector-core wrapper + encryption
│   └── adapters/
│       ├── http/       # Axum REST API (port 19789)
│       ├── ws/         # WebSocket adapter (port 19790)
│       ├── grpc/       # gRPC adapter (port 19792)
│       └── mcp/        # MCP adapter (stdio + HTTP transports, port 19791)
└── crates/cli/         # `brain` binary — commands, service management, chat
```

### Dependency Graph

```
cli ──► signal::SignalProcessor
           │
           ├── thalamus (intent classification)
           ├── amygdala (importance scoring)
           ├── hippocampus (memory read/write)
           │       ├── ruvector-core (HNSW index)
           │       └── storage (SQLite + encryption)
           ├── cortex (LLM + context + actions)
           ├── cerebellum (procedure store + trigger matching)
           └── ganglia (proactivity, runs as background task)

adapters/http ──► signal::SignalProcessor (Arc<>)
adapters/ws   ──► signal::SignalProcessor (Arc<>)
adapters/grpc ──► signal::SignalProcessor (Arc<>)
adapters/mcp  ──► signal::SignalProcessor (Arc<>)

External apps ──► Brain's HTTP / WS / MCP / gRPC API  (not inside this repo)
```

All adapters share **one** `Arc<SignalProcessor>`. There are no per-adapter memory stores.

---

## Data Flow: Signal Ingestion

```
Client Request
     │
     ▼
[Adapter] — parse, authenticate, build Signal
     │
     ▼
SignalProcessor::process(&signal)
     │
     ├─ 1. Thalamus: classify intent
     │         STORE_FACT | RECALL | CONVERSE | COMMAND | ...
     │
     ├─ 2. Amygdala: score importance (0.0 – 1.0)
     │
     ├─ 3. Hippocampus: store episode (chat path)
     │
     ├─ 4. Intent-dependent branch:
     │     STORE_FACT  → extract triple, store semantic fact
     │     RECALL      → hybrid_search(query) → RRF → top-k facts/episodes
     │     COMMAND     → exec_allowlist check → run subprocess
     │     CONVERSE    → skip to step 5
     │
     ├─ 5. Hippocampus: recall_engine.search() → build context
     │
     ├─ 6. Cortex: ContextAssembler → build prompt messages
     │
     ├─ 7. LlmProvider::generate(messages) → Response
     │
     └─ 8. Return SignalResponse + publish event bus update
```

The response is returned directly to the calling adapter, which sends it back to the client in the protocol-appropriate format.

---

## Key Types

### `Signal` (`crates/signal/src/lib.rs`)

The universal input envelope passed to `SignalProcessor::process`.

```rust
pub struct Signal {
    pub id: Uuid,
    pub source: SignalSource,
    pub channel: String,
    pub sender: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub namespace: String,  // default: "personal"
}
```

### `SignalResponse`

The universal output.

```rust
pub struct SignalResponse {
    pub signal_id: Uuid,
    pub status: ResponseStatus,
    pub response: ResponseContent,
    pub memory_context: MemoryContext,
}
```

### `SignalAdapter` trait

The single interface every adapter implements:

```rust
#[async_trait]
pub trait SignalAdapter: Send + Sync {
    /// Return the source type tag for this adapter.
    fn source(&self) -> SignalSource;

    /// Send a response back to this adapter's client.
    async fn send(&self, response: SignalResponse) -> Result<(), SignalError>;
}
```

### `SignalProcessor`

The shared engine. Constructed once at startup:

```rust
pub struct SignalProcessor { /* private fields */ }

impl SignalProcessor {
    pub async fn new(config: BrainConfig) -> Result<Self, SignalError>;
    pub async fn new_with_encryptor(config: BrainConfig, encryptor: Option<Encryptor>) -> Result<Self, SignalError>;
    pub async fn process(&self, signal: Signal) -> Result<SignalResponse, SignalError>;

    // Direct memory operations (used by adapters that bypass signal processing)
    pub async fn store_fact_direct(&self, ns: &str, cat: &str, sub: &str, pred: &str, obj: &str) -> Result<String, SignalError>;
    pub async fn search_facts(&self, query: &str, top_k: usize, namespace: Option<&str>) -> Vec<SemanticResult>;

    // Inspector accessors (for adapters that expose sub-resources)
    pub fn list_facts(&self, namespace: Option<&str>) -> Vec<Fact>;
    pub fn facts_about(&self, subject: &str) -> Vec<Fact>;
    pub fn list_namespaces(&self) -> Vec<NamespaceStats>;
    pub fn recent_episodes(&self, limit: usize) -> Vec<Episode>;
    pub fn procedures(&self) -> &ProcedureStore;
    pub fn episodic(&self) -> &EpisodicStore;
    pub fn config(&self) -> &BrainConfig;
    pub fn shutdown(&self);  // WAL checkpoint
}
```

---

## Storage Layer

### SQLite (`crates/storage`)

Migration-based schema versioned in `MIGRATIONS` slice. The migration runner compares `MAX(version)` in the `_migrations` table against the max version in code and runs all missing migrations in order.

Tables:
- `semantic_facts` — subject / predicate / object triples with namespace + importance
- `episodes` — conversation history with timestamps + importance
- `_migrations` — applied schema version log
- `procedures` — trigger_pattern → steps_json automation rules
- FTS5 virtual tables for BM25 full-text search

### Vector Index (`ruvector-core`)

Brain uses [`ruvector-core`](https://github.com/ruvnet/ruvector) (crates.io: `ruvector-core`) as an external dependency rather than implementing HNSW from scratch.

`storage/src/ruvector.rs` wraps `ruvector_core::VectorDB` with Brain's multi-table interface. Each logical table (e.g. `facts_vec`, `episodes_vec`) maps to one `VectorDB` persisted at `~/.brain/ruvector/<table>.db`. Supports:
- `insert(VectorEntry { id, vector })` — HNSW insert, file-backed
- `search(SearchQuery { vector, k })` → `Vec<SearchResult { id, score }>`
- `delete(id)` — remove by ID

Before insert/search, vectors are sanitized (finite values, expected dimensions, L2 normalization). Invalid vectors are replaced with deterministic normalized fallback vectors, and insert vectors receive a tiny deterministic id-based jitter to avoid HNSW pathological duplicate-distance asserts.

### Hybrid Search

`RecallEngine::search()` runs both:
1. HNSW approximate nearest-neighbour (cosine similarity)
2. SQLite FTS5 BM25 full-text search

Results are merged with Reciprocal Rank Fusion (RRF, `k=60`) and re-ranked by importance score.

---

## Action Dispatcher Backends (Internal)

Action execution is internal to the shared engine/CLI flow and not exposed as a new public adapter API in this cycle.

`cortex::ActionDispatcher` supports backend traits for:
- memory (`store_fact`, `recall`)
- web search (`web_search`)
- scheduling (`schedule_task`)
- outbound messaging (`send_message`)

Dispatch contract is explicit:
- feature disabled -> failure (`disabled by config`)
- enabled without backend wiring -> failure (`backend not configured`)
- enabled with backend -> real execution with structured success output

Current concrete wiring in CLI is platform-agnostic:
- Web search: generic HTTP JSON endpoint
- Scheduling: persist-only SQLite intent storage (`scheduled_intents`)
- Messaging: channel -> webhook endpoint map

---

## Memory Namespaces

Every fact and episode carries a `namespace: TEXT NOT NULL DEFAULT 'personal'`. Callers pass `namespace` through the `Signal` and all query methods accept `Option<&str>` to filter. The default namespace is `"personal"`.

---

## Security Model

| Concern | Mechanism |
|---------|-----------|
| API authentication | Bearer token / `x-api-key` checked on every request before processing |
| CORS | `localhost_cors()` — only `127.0.0.1` and `localhost` origins allowed |
| Error exposure | HTTP 500 returns opaque message; real error logged server-side only |
| Shell execution | Allowlist (`security.exec_allowlist` in config); configurable timeout |
| Encryption at rest | AES-256-GCM via `brain init --encrypt`; opt-in |
| LLM client failures | Constructors return `Result<>` — TLS failures surface as errors, not panics |

---

## Integrating External Applications

Brain is local and protocol-agnostic. External applications (OpenClaw, coding agents, shell scripts, chat UIs) connect to Brain via its standard interfaces — they do not live inside this repository.

### Where integration code lives

```
Brain OS (this repo)          External App (separate repo / process)
─────────────────────         ──────────────────────────────────────
  brain serve                   OpenClaw, agent, script, etc.
       │                              │
       │  HTTP REST / WS / MCP /      │
       │◄─────────── gRPC ───────────►│
       │                              │
  SignalProcessor               app-specific logic
  hippocampus                   thin Brain client (HTTP calls)
```

### Integration patterns

**1. HTTP REST (simplest)**

Any app that can make HTTP requests can use Brain immediately:

```bash
# From any shell script, Python script, or app
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"source":"openclaw","sender":"agent","content":"user prefers tabs over spaces"}'
```

**2. MCP (for AI agents)**

AI agents that speak MCP can declare Brain as a server and call `memory_search`, `memory_store`, etc. as native tools — no HTTP client code needed.

**3. WebSocket (for real-time / streaming)**

Apps that need push notifications or streaming responses connect to `ws://localhost:19790` and authenticate with the first frame:
```json
{"api_key": "your-key"}
```

**4. gRPC (for high-throughput or typed clients)**

Generate client stubs from Brain's proto files in `crates/adapters/grpc/proto/` for any language. The `MemoryService` and `AgentService` RPCs map 1-to-1 with the HTTP routes. `AgentService.ReceiveSignals` is a live server stream: subscribers receive an initial `connected` event plus fan-out updates emitted by `SignalProcessor` after successful processing.

### SDK / client library

There is no official Brain client SDK yet. The simplest integration is a thin HTTP wrapper in your language of choice:

```python
# Minimal Python client — no SDK needed
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
```

If you want to connect a messaging app, CLI tool, or AI agent to Brain, call Brain's existing HTTP/WS/MCP/gRPC API from that app — you do not add a "Slack adapter" or "Telegram adapter" inside Brain.

### When to add a new protocol adapter

Add a new adapter when you need a transport that Brain doesn't yet speak — for example Server-Sent Events, Unix domain sockets, AMQP, or a custom binary protocol.

### Step-by-step: SSE (Server-Sent Events) Adapter

SSE is a good example: it's one-directional (server pushes, client reads), so it suits streaming responses to browser clients.

**1. Create the crate**

```
crates/adapters/sse/
├── Cargo.toml
└── src/lib.rs
```

`Cargo.toml`:

```toml
[package]
name = "sseadapter"
version.workspace = true
edition.workspace = true

[dependencies]
signal  = { workspace = true }
brain-core = { workspace = true }
tokio   = { workspace = true }
axum    = { workspace = true }
serde   = { workspace = true }
serde_json = { workspace = true }
```

Add to workspace `Cargo.toml` `members` list and `[workspace.dependencies]`.

**2. Implement the adapter**

```rust
// crates/adapters/sse/src/lib.rs

use std::sync::Arc;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::sse::{Event, Sse},
    routing::get,
    Router,
};
use signal::{Signal, SignalSource};
use tokio_stream::StreamExt as _;

pub struct SseAdapterState {
    pub processor: Arc<signal::SignalProcessor>,
    pub api_keys: Vec<brain_core::ApiKeyConfig>,
}

pub fn router(state: Arc<SseAdapterState>) -> Router {
    Router::new()
        .route("/v1/stream", get(stream_handler))
        .with_state(state)
}

async fn stream_handler(
    State(state): State<Arc<SseAdapterState>>,
    headers: HeaderMap,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, StatusCode> {
    // 1. Authenticate
    let key = headers.get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !state.api_keys.iter().any(|k| k.key == key) {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // 2. Build a stream that replays recent episodes as SSE events
    let episodes = state.processor.recent_episodes(50);
    let stream = tokio_stream::iter(episodes)
        .map(|ep| {
            let data = serde_json::json!({ "role": ep.role, "content": ep.content });
            Ok(Event::default().data(data.to_string()))
        });

    Ok(Sse::new(stream))
}
```

**3. Wire into `brain serve`**

In `crates/cli/src/main.rs`, add a `--sse` flag to `ServeArgs` and spawn the adapter:

```rust
if args.sse {
    let sse_state = Arc::new(sseadapter::SseAdapterState {
        processor: Arc::clone(&processor),
        api_keys: config.access.api_keys.clone(),
    });
    let port = 19793u16;
    tokio::spawn(async move {
        let app = sseadapter::router(sse_state);
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });
}
```

The key insight is that all adapters follow the same contract: receive input → build `Signal` → call `processor.process()` → return output. The `SignalProcessor` is shared by reference (`Arc`) so memory is always consistent regardless of which adapter handled the request.

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

Implementations: `OllamaProvider` (local Ollama server), `OpenAiProvider` (OpenAI-compatible, works with OpenRouter, local vLLM, etc.).

New providers implement `LlmProvider` and are registered in `create_provider()`.

---

## Configuration System

Config is loaded with [Figment](https://docs.rs/figment) in priority order:

```
Environment variables  (BRAIN_LLM__MODEL=gpt-4o)
    ↓ override
~/.brain/config.yaml   (user overrides)
    ↓ override
config/default.yaml    (compiled-in defaults via include_str!)
```

The `BrainConfig` struct in `crates/core/src/config.rs` maps 1-to-1 with the YAML keys. Double-underscore (`__`) is the env-var nesting separator.

---

## Proactivity Engine (`crates/ganglia`)

A background loop that detects recurring patterns in episodic memory (keyword × day-of-week × hour histograms) and generates proactive suggestions. Disabled by default (`proactivity.enabled: false`).

When enabled, the daemon spawns a `HabitEngine` task that runs on a configurable interval (default: every 60 minutes). If a recurring pattern matches the current time slot and all rate limits are satisfied (`max_per_day`, `min_interval_minutes`, `quiet_hours`), the engine logs a proactive message to `~/.brain/logs/proactive.log`.

The engine persists its state (last fire times, daily counts) in a `habit_state` key-value table in SQLite, so rate limits survive daemon restarts.

---

## Memory Consolidation (`crates/hippocampus/consolidation`)

A background task that prunes low-retention memories using the forgetting curve. Enabled by default (`memory.consolidation.enabled: true`, interval: 24 hours).

The `Consolidator` iterates over episodes, computes retention as `importance * e^(-decay_rate * hours_since_last_access)`, deletes episodes below `forgetting_threshold` (default: 0.05), and returns concrete promotion candidates (`episode_id`, `namespace`, `content`, `reinforcement`). The daemon loop promotes these candidates to semantic facts using deterministic StoreFact parsing and records `episode_promotions` in SQLite for idempotency across runs.

---

## Procedure Store (`crates/cerebellum`)

Stores trigger-pattern → steps-JSON automation rules. When a signal arrives, `SignalProcessor` checks all stored procedures for a trigger match (case-insensitive substring of the signal content). If a procedure matches, its steps are injected into the LLM context so the response incorporates the workflow.

Procedures are managed via:
- MCP `memory_procedures` tool (list / store / delete)
- Direct `ProcedureStore` API (used by SignalProcessor)

---
