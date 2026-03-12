# Brain OS 🧠

**Stop giving your AI amnesia.**

Brain OS is a biologically-inspired, central cognitive engine written in pure Rust. Instead of every script, coding assistant, and chat UI keeping its own isolated, fragmented context, Brain OS acts as your single source of truth.

It routes intents through a Thalamus, scores importance via an Amygdala, and stores everything in a unified Hippocampus (FTS5 + HNSW Vector Search). Whether you connect via HTTP, WebSocket, gRPC, or MCP, your AI tools now share one localized, ever-growing memory that runs 24/7 on your machine.

*Your data never leaves your hardware. Your AI never forgets.*

---

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25 FTS5), fuses results via Reciprocal Rank Fusion, and reranks by importance and recency. A forgetting curve runs every 24 hours to prune low-value memories and promote reinforced episodes to permanent semantic facts.

---

## Install

**Requirements:** Rust 1.82+, [Ollama](https://ollama.com) (or any OpenAI-compatible API), Docker (optional, for web search)

```bash
# Install the CLI
cargo install --path crates/cli

# Initialize data directory (~/.brain/)
brain init

# Pull the default LLM + embedding models (Ollama)
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# Start external services (SearXNG web search — optional)
brain deps up
```

`brain init` creates `~/.brain/` with config, database, vector index, and log directories.

`brain deps up` starts a Docker container for SearXNG (web search, port 8888). This is optional — Brain works without it but web search intents will return "backend not configured".

If the embedding provider is unavailable, Brain uses deterministic normalized fallback vectors so writes and search continue without panics. Semantic quality is lower until the embedding provider is healthy.

---

## Usage

```bash
# Start Brain as a background daemon (all adapters enabled)
brain start

# Stop the daemon
brain stop

# Check daemon + adapter status
brain status

# Interactive chat (connects to running daemon or starts inline)
brain chat

# One-shot message
brain chat "remember that I use dark mode"
```

---

## External Services (Docker)

Brain uses an optional Docker container for web search:

```bash
brain deps up       # Start SearXNG
brain deps status   # Check if running
brain deps down     # Stop
```

| Service | Port | Purpose |
|---------|------|---------|
| SearXNG | 8888 | Web search backend (metasearch engine) |

`brain status` automatically checks if SearXNG is reachable.

---

## Auto-Start on Login

Install Brain as a system service so it starts automatically on login:

```bash
# Install (creates launchd / systemd / Task Scheduler entry)
brain service install

# Remove the service
brain service uninstall
```

| Platform | Mechanism | Privileges required |
|----------|-----------|---------------------|
| macOS | launchd (LaunchAgents) | None |
| Linux | systemd user service | None |
| Windows | Task Scheduler (ONLOGON) | None |

After installation the daemon starts immediately and will restart after crashes.

---

## MCP Integration

Any MCP-compatible client can connect to Brain as a stdio MCP server. MCP (Model Context Protocol) is an open standard for connecting AI assistants to tools and data sources.

Configure your MCP client to spawn Brain as a subprocess:

```json
{
  "mcpServers": {
    "brain": {
      "command": "brain",
      "args": ["mcp"]
    }
  }
}
```

Brain also exposes MCP over HTTP (`brain serve --mcp`) for clients that prefer HTTP transport.

### MCP Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `top_k?`, `namespace?` | Hybrid semantic + full-text search |
| `memory_store` | `subject`, `predicate`, `object`, `category`, `namespace?` | Store a semantic fact |
| `memory_facts` | `subject`, `namespace?` | All facts about a subject (optional namespace filter) |
| `memory_episodes` | `limit?` | Recent conversation history |
| `user_profile` | — | Current user configuration |
| `memory_procedures` | `action`, `trigger?`, `steps?`, `procedure_id?` | Manage learned workflows (list / store / delete) |

### MCP Authentication

MCP stdio passes auth in the `_meta` field of every request:

```json
{
  "method": "tools/call",
  "params": {
    "_meta": { "x-api-key": "your-key" },
    "name": "memory_search",
    "arguments": { "query": "dark mode" }
  }
}
```

MCP over HTTP uses the `x-api-key` header.

---

## HTTP API

Default port: `19789`. All `/v1/*` routes require `Authorization: Bearer <key>`.

```bash
# Health check (no auth)
curl http://localhost:19789/health

# Prometheus metrics (no auth)
curl http://localhost:19789/metrics

# Web UI — diagnostic tool (no auth)
open http://localhost:19789/ui

# OpenAPI spec
curl http://localhost:19789/openapi.json

# Swagger UI
open http://localhost:19789/api

# Store a fact (only "content" is required; source/sender/namespace/agent are optional)
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"I prefer dark mode"}'

# Search memory
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"UI preferences","top_k":5}'

# List all facts
curl http://localhost:19789/v1/memory/facts \
  -H "Authorization: Bearer YOUR_API_KEY"

# Namespace statistics
curl http://localhost:19789/v1/memory/namespaces \
  -H "Authorization: Bearer YOUR_API_KEY"

# SSE stream of proactive notifications (open loop reminders, habit nudges)
curl -N http://localhost:19789/v1/events \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Liveness check |
| `GET` | `/metrics` | No | Prometheus metrics |
| `GET` | `/ui` | No | Browser UI (diagnostic) |
| `GET` | `/openapi.json` | No | OpenAPI spec |
| `GET` | `/api` | No | Swagger UI |
| `POST` | `/v1/signals` | Yes | Submit a signal |
| `GET` | `/v1/signals/:id` | Yes | Poll cached response |
| `POST` | `/v1/memory/search` | Yes | Hybrid semantic search |
| `GET` | `/v1/memory/facts` | Yes | List all facts |
| `GET` | `/v1/memory/namespaces` | Yes | Namespace stats |
| `GET` | `/v1/events` | Yes | SSE stream of proactive notifications |

---

## Services & Ports

`brain start` launches all adapters together. They share a single processor so memory is consistent across all protocols.

| Adapter | Default Port | Notes |
|---------|-------------|-------|
| HTTP REST | 19789 | REST API + Web UI + Swagger + OpenAPI |
| WebSocket | 19790 | Bidirectional streaming, real-time |
| MCP HTTP | 19791 | MCP over HTTP transport |
| gRPC | 19792 | Protobuf RPC + server streaming |
| MCP stdio | stdin/stdout | `brain mcp` for subprocess MCP clients |

### Adapter Behavior Matrix

| Adapter | Auth | Namespace Input | Streaming | Memory Semantics |
|---------|------|-----------------|-----------|------------------|
| HTTP | Bearer API key | `namespace` on `/v1/signals` and `/v1/memory/search` | Request/response | Shared semantic+episodic stores |
| WebSocket | First frame `api_key` | `namespace` in each message | Bidirectional socket | Shared semantic+episodic stores |
| gRPC | Interceptor (`x-api-key` or Bearer metadata) | `namespace` on signal/search/store requests | Server streaming (`ReceiveSignals`, `StreamSignals`) | Shared semantic+episodic stores |
| MCP (stdio/http) | `_meta.x-api-key` / `x-api-key` header | Tool args (`memory_store`, `memory_search`, `memory_facts`) | JSON-RPC request/response | Shared semantic+episodic stores |

For development, `brain serve` runs everything in the foreground with optional flags:

```bash
brain serve               # all adapters (foreground)
brain serve --http        # HTTP only
brain serve --http --ws   # HTTP + WebSocket
brain serve --mcp         # MCP HTTP only
```

---

## Memory Namespaces

Scope facts and episodes to a context. The default namespace is `"personal"`.

```bash
# Store a project-specific fact
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"use bun not npm","namespace":"my-project"}'

# Search only within that namespace
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"package manager","namespace":"my-project"}'
```

---

## External Gateway Relay (Brain Bridge)

Brain is a local service — it does not reach outward to external messaging platforms. Instead, a thin external **bridge** connects a platform-specific bot or gateway to Brain's WebSocket API and translates messages in both directions.

```
External Platform           Bridge (your code / external repo)        Brain OS
────────────────────        ──────────────────────────────────        ────────────────
  Slack / Telegram    ────► BridgeClient (crates/bridge library) ──► ws://localhost:19790
  Custom chat agent          exponential-backoff reconnection          SignalProcessor
  Any WebSocket bot          thin message translation                  memory + LLM
```

The `crates/bridge/` library provides a `BridgeClient` for building these relays. It handles reconnection with exponential backoff, ping/pong keep-alive, and JSON message serialization automatically. No platform-specific code lives inside Brain itself.

A minimal bridge connecting an external gateway to Brain:

```rust
// In your own external relay project — not inside the Brain OS repo
use bridge::{BridgeClient, BridgeConfig, BridgeMessage};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Connect to YOUR gateway (e.g. a Slack bot WebSocket endpoint)
    let client = BridgeClient::new(
        "ws://your-gateway.example.com/brain-relay",
        BridgeConfig::default(), // exponential backoff: 1s → 2s → 4s → … → 60s
    );

    // For each inbound message from the gateway, forward to Brain and relay the response
    client.connect_and_relay(|msg| async move {
        BridgeMessage::reply(&msg, call_brain_ws(&msg.content).await)
    }).await?;

    Ok(())
}
```

Brain's WebSocket API (`ws://localhost:19790`) is the entry point — the bridge is external and lives in its own repository. This keeps Brain small, stable, and protocol-agnostic.

### Bridge CLI Command

Brain provides a built-in `brain bridge` command that simplifies connecting external gateways:

```bash
# Connect to an external WebSocket gateway
brain bridge ws://localhost:8080/gateway

# With custom API key
brain bridge ws://localhost:8080/gateway --api-key YOUR_KEY
```

The bridge command:
1. Connects to your external WebSocket gateway
2. Connects to Brain's WebSocket synapse internally
3. Relays messages bidirectionally between the gateway and Brain
4. Automatically handles reconnection with exponential backoff

This is useful for quickly testing bridge connections or for simple relay setups without writing custom code.

---

## Background Intelligence

`brain serve` and `brain start` spawn background tasks alongside the protocol adapters, sharing the same `SignalProcessor`:

### Memory Consolidation (enabled by default)

Runs every 24 hours. Uses an Ebbinghaus forgetting curve to prune low-retention episodes and promote frequently-reinforced episodes to permanent semantic facts with an idempotency guard.

```yaml
memory:
  consolidation:
    enabled: true          # on by default
    interval_hours: 24
    forgetting_threshold: 0.05   # episodes with retention < 5% are pruned
```

### Proactivity Engine (opt-in)

When enabled, Brain becomes bidirectional — it proactively reminds you of things instead of only responding when asked.

**Habit Detection** — scans episodic memory for recurring patterns (keyword × day-of-week × hour histograms) and nudges you when a pattern matches the current time slot.

**Open-Loop Detection** — scans for unresolved commitments ("I need to...", "remind me to...", "I should...") and generates reminders when no resolution is found within the configured window.

**Delivery** — proactive messages are delivered through three tiers:
1. **Outbox** — written to SQLite, drained on next `brain chat` session
2. **Broadcast** — pushed to live WebSocket and SSE (`GET /v1/events`) sessions
3. **Webhooks** — pushed to configured messaging channels (Slack, Discord, Telegram, etc.)

```yaml
proactivity:
  enabled: false           # opt-in; set to true to activate
  max_per_day: 5
  min_interval_minutes: 60
  quiet_hours:
    start: "22:00"
    end: "08:00"
  delivery:
    outbox: true           # always write to outbox; drain on next interaction
    broadcast: true        # push to live WS/SSE sessions
    webhook_channels: []   # channel keys from actions.messaging.channels
    max_outbox_age_days: 7
  open_loop:
    enabled: true          # detect unresolved commitments (requires proactivity.enabled)
    scan_window_hours: 72
    resolution_window_hours: 24
    check_interval_minutes: 120
```

### Agent Identity

Every signal can carry an `agent` field identifying the originating AI tool (e.g. `"claude-code"`, `"cursor"`). Agent identity flows through the entire pipeline — recall, habit detection, and proactive messages reference the originating agent when known.

```bash
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"deploy staging server","agent":"devops-agent"}'
```

---

## Action Backends (Internal)

Action intents routed by Thalamus (`web_search`, `schedule_task`, `send_message`) are handled by internal `ActionDispatcher` backends. These are internal-only — no public HTTP or gRPC endpoints expose them directly.

Behavior contract:
- Disabled in config → explicit `disabled by config` error
- Enabled but backend missing → explicit `backend not configured` error
- Backend configured → real execution with structured success output

### Web Search Providers

```yaml
actions:
  web_search:
    enabled: true
    provider: "searxng"               # searxng | tavily | custom
    endpoint: "http://localhost:8888"  # SearXNG instance URL
    api_key: ""                        # required for tavily
    timeout_ms: 3000
    default_top_k: 5
```

| Provider | Auth | Self-hosted | Setup |
|----------|------|-------------|-------|
| `searxng` | None | Yes | `brain deps up` (or `docker run -d -p 8888:8080 searxng/searxng`) |
| `tavily` | API key (free, no CC) | No | Sign up at tavily.com, set `api_key` |
| `custom` | None | — | Set `endpoint` to any OpenAI-compatible JSON search API |

### Messaging (Webhooks)

Brain sends messages via configurable webhook URLs. Any service that accepts HTTP POST works — Slack, Discord, Telegram, ntfy.sh, or a custom endpoint. No platform SDK is bundled.

```yaml
actions:
  messaging:
    enabled: true
    timeout_ms: 3000
    channels:
      slack:
        url: "https://hooks.slack.com/services/T/B/x"
        body: '{"text": "{{content}}"}'
      discord:
        url: "https://discord.com/api/webhooks/123/abc"
        body: '{"content": "[{{channel}}] {{content}}"}'
      telegram:
        url: "https://api.telegram.org/bot<TOKEN>/sendMessage"
        body: '{"chat_id": "<ID>", "text": "{{content}}"}'
        headers:
          Content-Type: "application/json"
      simple: "https://example.com/hook"  # shorthand: URL only, default JSON body
```

Template placeholders: `{{content}}`, `{{channel}}`, `{{recipient}}`, `{{namespace}}`, `{{timestamp}}`. Values are JSON-escaped automatically. Custom `headers` are optional (useful for auth-requiring APIs like Telegram).

### Backend Resilience

All HTTP backends (web search + messaging) share a retry and circuit breaker configuration:

```yaml
actions:
  resilience:
    max_retries: 2                     # retries on 5xx / timeout / connection refused
    retry_base_ms: 500                 # exponential backoff: 500 → 1000 → 2000ms
    circuit_breaker_threshold: 5       # consecutive failures before circuit opens
    circuit_breaker_cooldown_secs: 60  # seconds before retrying after circuit opens
```

4xx errors (auth, bad request) fail immediately without retries. When a circuit opens, all requests to that backend return an instant error until the cooldown elapses.

### Scheduling

```yaml
  scheduling:
    enabled: false
    mode: "persist_only"    # intents stored in SQLite; background poller fires due intents
```

When `scheduling.enabled: true`, a background task in `brain serve` polls every 60 seconds for
pending intents and delivers them as proactive notifications via the `NotificationRouter`.

---

## Export & Import

Back up and restore all memory:

```bash
# Export to stdout (pipe to file)
brain export > backup.json

# Export directly to file
brain export --output backup.json

# Preview what an import would do (dry-run)
brain import backup.json --dry-run

# Import from backup
brain import backup.json
```

The export format is a self-contained JSON file containing all facts and episodes with timestamps, importance scores, and namespace labels. Import is idempotent — re-importing the same backup is safe.

---

## Authentication

| Adapter | Method |
|---------|--------|
| HTTP REST | `Authorization: Bearer <key>` |
| WebSocket | First frame: `{"api_key":"<key>"}` |
| MCP HTTP | `x-api-key: <key>` header |
| MCP stdio | `params._meta["x-api-key"]` |
| gRPC | Interceptor checks `x-api-key` or `authorization` metadata |

Configure keys in `~/.brain/config.yaml`:

```yaml
access:
  api_keys:
    - key: "your-secret-key"
      name: "Production Key"
      permissions: [read, write]
    - key: "readonly-key"
      name: "Read Only"
      permissions: [read]
```

`brain init` generates a unique API key (prefixed `brk_`) and prints it to the terminal. Find your key in `~/.brain/config.yaml` under `access.api_keys`.

---

## Configuration

Config is loaded from three sources (highest priority wins):

1. **Environment variables** — `BRAIN_LLM__MODEL=gpt-4o brain serve`
2. **User config** — `~/.brain/config.yaml`
3. **Defaults** — [`config/default.yaml`](config/default.yaml)

Double-underscore (`__`) is the nesting separator in env var names.

### LLM Provider

```yaml
llm:
  provider: "ollama"               # ollama | openai
  model: "qwen2.5-coder:7b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 4096
  intent_llm_fallback: false       # enable LLM fallback when regex intent classification is uncertain
```

To use OpenAI or OpenRouter:

```yaml
llm:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4o"
```

### Embedding Model

```yaml
embedding:
  model: "nomic-embed-text"       # must be pulled: `ollama pull nomic-embed-text`
  dimensions: 768                  # must match the model output size
```

For OpenAI-compatible embeddings:

```yaml
embedding:
  model: "text-embedding-3-small"
  dimensions: 1536
```

### Encryption (at-rest)

```bash
# Generate a salt and enable encryption
brain init --encrypt
```

Then set `encryption.enabled: true` in `~/.brain/config.yaml` and provide a passphrase:

```bash
# Via environment variable (for daemon/CI)
BRAIN_PASSPHRASE="your-passphrase" brain serve

# Or Brain will prompt interactively on startup
brain serve
```

---

## Data Directory

```
~/.brain/
├── config.yaml        # User configuration (overrides defaults)
├── db/
│   ├── brain.db       # SQLite — facts, episodes, procedures, FTS5 index
│   └── salt           # Encryption salt (only if --encrypt was used)
├── ruvector/          # HNSW vector index files (ruvector-core)
├── logs/
│   └── brain.log      # Daemon logs
└── exports/           # Export output directory
```

---

## Re-initialise

```bash
# Regenerate config with a new API key (data directories are preserved)
brain init --force

# Also enable encryption
brain init --force --encrypt
```

`--force` overwrites `~/.brain/config.yaml` with defaults and a fresh API key. Your database, vector index, and exports remain untouched.

---

## License

MIT
