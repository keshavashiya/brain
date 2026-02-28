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

## Building a New Adapter

An adapter:
1. Holds an `Arc<SignalProcessor>` received at startup.
2. Authenticates incoming requests against the configured API keys.
3. Translates the protocol-specific input into a `Signal`.
4. Calls `processor.process(&signal).await`.
5. Translates the `SignalResponse` back into the protocol-specific output.

### Step-by-step: Telegram Bot Adapter

**1. Create the crate**

```bash
mkdir -p crates/adapters/telegram/src
```

`crates/adapters/telegram/Cargo.toml`:

```toml
[package]
name = "telegramadapter"
version.workspace = true
edition.workspace = true

[dependencies]
signal = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
teloxide = "0.13"   # Telegram bot framework
```

Add to workspace `Cargo.toml`:

```toml
[workspace]
members = [
    ...
    "crates/adapters/telegram",
]
```

**2. Implement the adapter**

```rust
// crates/adapters/telegram/src/lib.rs

use std::sync::Arc;
use signal::{Signal, SignalSource};
use teloxide::prelude::*;

pub async fn run(
    processor: Arc<signal::SignalProcessor>,
    bot_token: &str,
) {
    let bot = Bot::new(bot_token);

    teloxide::repl(bot, move |bot: Bot, msg: Message| {
        let processor = Arc::clone(&processor);
        async move {
            // 1. Authenticate — check if user is allowed
            //    (e.g. check config for telegram_allowed_users)

            // 2. Build a Signal
            let text = msg.text().unwrap_or("").to_string();
            let signal = Signal::new(
                SignalSource::Http,            // use Http or add a Telegram variant
                Some(msg.chat.id.to_string()), // channel
                Some(msg.from().unwrap().username.clone().unwrap_or_default()),
                text,
                None, // namespace — default "personal"
            );

            // 3. Process
            match processor.process(&signal).await {
                Ok(response) => {
                    let text = match &response.content {
                        signal::ResponseContent::Text(t) => t.clone(),
                        _ => "Done.".to_string(),
                    };
                    bot.send_message(msg.chat.id, text).await?;
                }
                Err(e) => {
                    bot.send_message(msg.chat.id, "Something went wrong.").await?;
                    tracing::error!("Telegram adapter error: {e}");
                }
            }

            respond(())
        }
    })
    .await;
}
```

**3. Wire into the CLI**

In `crates/cli/src/main.rs`, add a `--telegram` flag to the `serve` command and spawn the adapter task alongside the existing adapters:

```rust
if cfg.serve_telegram {
    let token = std::env::var("TELEGRAM_BOT_TOKEN")
        .expect("TELEGRAM_BOT_TOKEN not set");
    let p = Arc::clone(&processor);
    tokio::spawn(async move {
        telegramadapter::run(p, &token).await;
    });
}
```

**4. Add to config**

```yaml
adapters:
  telegram:
    enabled: false
    # Set TELEGRAM_BOT_TOKEN env var
```

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
