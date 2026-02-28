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
│   ├── thalamus/       # Intent classification (rules + LLM fallback)
│   ├── amygdala/       # Importance scoring (keyword heuristics + LLM)
│   ├── hippocampus/    # Memory: EpisodicStore, SemanticStore, RecallEngine,
│   │                     Embedder (ONNX bge-small-en-v1.5), ImportanceScorer
│   ├── cortex/         # LLM clients: OllamaProvider, OpenAiProvider
│   │                     ContextAssembler (builds prompts from recall results)
│   ├── cerebellum/     # ProcedureStore — trigger-pattern → steps automation
│   ├── ganglia/        # Proactivity engine (scheduled reminders)
│   ├── ruvector/       # HNSW vector index (pure Rust, no external deps)
│   ├── storage/        # SQLite migrations + raw DB helpers
│   ├── bridge/         # WebSocket bridge for multi-device/cloud sync
│   ├── mcp/            # MCP adapter (stdio + HTTP transports)
│   └── adapters/
│       ├── http/       # Axum REST API (port 19789)
│       ├── ws/         # WebSocket adapter (port 19790)
│       └── grpc/       # gRPC adapter (port 19792)
└── crates/cli/         # `brain` binary — commands, service management, chat
```

### Dependency Graph

```
cli ──► signal::SignalProcessor
           │
           ├── thalamus (intent)
           ├── amygdala (importance)
           ├── hippocampus (memory read/write)
           │       ├── ruvector (HNSW index)
           │       └── storage (SQLite)
           ├── cortex (LLM + context)
           └── cerebellum (procedures)

adapters/http ──► signal::SignalProcessor (Arc<>)
adapters/ws   ──► signal::SignalProcessor (Arc<>)
adapters/grpc ──► signal::SignalProcessor (Arc<>)
mcp           ──► signal::SignalProcessor (Arc<>)
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
     ├─ 3. Hippocampus: store episode (always)
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
     └─ 8. Return SignalResponse { text, memory_context }
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
    pub channel: Option<String>,
    pub sender: Option<String>,
    pub content: String,
    pub namespace: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

### `SignalResponse`

The universal output.

```rust
pub struct SignalResponse {
    pub signal_id: Uuid,
    pub status: ResponseStatus,
    pub content: ResponseContent,
    pub memory_context: Option<MemoryContext>,
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
    pub async fn process(&self, signal: &Signal) -> Result<SignalResponse, SignalError>;

    // Inspector accessors (for adapters that expose sub-resources)
    pub fn list_facts(&self, namespace: Option<&str>) -> Vec<Fact>;
    pub fn list_namespaces(&self) -> Vec<NamespaceStats>;
    pub fn recent_episodes(&self, limit: usize) -> Vec<Episode>;
    pub fn procedures(&self) -> &ProcedureStore;
    pub fn config(&self) -> &BrainConfig;
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

### Vector Index (`crates/ruvector`)

Pure-Rust HNSW index persisted to `~/.brain/ruvector/`. No external native dependencies. Supports:
- `insert(id, vector)`
- `search(query_vector, k, ef_search) -> Vec<(id, distance)>`

### Hybrid Search

`RecallEngine::search()` runs both:
1. HNSW approximate nearest-neighbour (cosine similarity)
2. SQLite FTS5 BM25 full-text search

Results are merged with Reciprocal Rank Fusion (RRF, `k=60`) and re-ranked by importance score.

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

An optional background loop that fires scheduled suggestions based on memory patterns. Disabled by default (`proactivity.enabled: false`). Respects `quiet_hours`, `max_per_day`, and `min_interval_minutes`.

---

## Bridge (`crates/bridge`)

A WebSocket client that forwards signals to a remote Brain gateway for multi-device sync. Includes exponential backoff reconnect (`initial_backoff_ms` → `max_backoff_ms`). Disabled by default.
