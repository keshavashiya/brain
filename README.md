# Brain OS 🧠

**Stop giving your AI amnesia.** 

Brain OS is a biologically-inspired, central cognitive engine written in pure Rust. Instead of every script, coding assistant, and chat UI keeping its own isolated, fragmented context, Brain OS acts as your single source of truth. 

It routes intents through a Thalamus, scores importance via an Amygdala, and stores everything in a unified Hippocampus (FTS5 + HNSW Vector Search). Whether you connect via HTTP, WebSocket, gRPC, or MCP, your AI tools now share one localized, ever-growing memory that runs 24/7 on your machine.

*Your data never leaves your hardware. Your AI never forgets.*

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25 FTS5), fuses results via Reciprocal Rank Fusion, and reranks by importance and recency.

## Install

**Requirements:** Rust 1.82+, [Ollama](https://ollama.com) (or any OpenAI-compatible API), Docker (optional, for web search + notifications)

```bash
# Install the CLI
cargo install --path crates/cli

# Initialize data directory (~/.brain/)
brain init

# Pull the default LLM + embedding models (Ollama)
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# Start external services (SearXNG web search)
brain deps up
```

`brain init` creates `~/.brain/` with config, database, vector index, and log directories.

`brain deps up` starts a Docker container for SearXNG (web search, port 8888). This is optional — Brain works without it but web search intents will return "backend not configured".

If the embedding provider is unavailable, Brain uses deterministic normalized fallback vectors so writes and search continue without panics. Semantic quality is lower until the embedding provider is healthy.

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

## MCP Integration

Any MCP-compatible client can connect to Brain as a stdio MCP server:

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

## HTTP API

Default port: `19789`. All `/v1/*` routes require `Authorization: Bearer <key>`.

```bash
# Health check (no auth)
curl http://localhost:19789/health

# Prometheus metrics (no auth)
curl http://localhost:19789/metrics

# Web UI (no auth)
open http://localhost:19789/ui

# OpenAPI spec
curl http://localhost:19789/openapi.json

# OpenAPI info.title: "Brain OS — Synapse HTTP API"

# Swagger UI
open http://localhost:19789/api

# Store a fact
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer demokey123" \
  -H "Content-Type: application/json" \
  -d '{"source":"http","sender":"user","content":"I prefer dark mode"}'

# Search memory
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer demokey123" \
  -H "Content-Type: application/json" \
  -d '{"query":"UI preferences","top_k":5}'

# List all facts
curl http://localhost:19789/v1/memory/facts \
  -H "Authorization: Bearer demokey123"

# Namespace statistics
curl http://localhost:19789/v1/memory/namespaces \
  -H "Authorization: Bearer demokey123"
```

### Routes

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Liveness check |
| `GET` | `/metrics` | No | Prometheus metrics |
| `GET` | `/ui` | No | Browser UI |
| `GET` | `/openapi.json` | No | OpenAPI spec |
| `GET` | `/api` | No | Swagger UI |
| `POST` | `/v1/signals` | Yes | Submit a signal |
| `GET` | `/v1/signals/:id` | Yes | Poll cached response |
| `POST` | `/v1/memory/search` | Yes | Hybrid semantic search |
| `GET` | `/v1/memory/facts` | Yes | List all facts |
| `GET` | `/v1/memory/namespaces` | Yes | Namespace stats |

## Services & Ports

`brain start` launches all adapters together. They share a single processor so memory is consistent across all protocols.

| Adapter | Default Port | Notes |
|---------|-------------|-------|
| HTTP REST | 19789 | REST API + Web UI + Swagger |
| WebSocket | 19790 | Streaming, real-time |
| MCP HTTP | 19791 | MCP over HTTP transport |
| gRPC | 19792 | Protobuf RPC |
| MCP stdio | stdin/stdout | `brain mcp` for MCP clients |

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

## Memory Namespaces

Scope facts and episodes to a context. The default namespace is `"personal"`.

```bash
# Store a project-specific fact
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer demokey123" \
  -H "Content-Type: application/json" \
  -d '{"source":"http","sender":"user","content":"use bun not npm","namespace":"my-project"}'

# Search only within that namespace
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer demokey123" \
  -H "Content-Type: application/json" \
  -d '{"query":"package manager","namespace":"my-project"}'
```

## Action Backends (Internal)

Action intents routed by Thalamus (`web_search`, `schedule_task`, `send_message`) are handled by internal `ActionDispatcher` backends. These are internal-only (no public HTTP/gRPC endpoints).

Behavior contract:
- Disabled in config -> explicit `disabled by config` failure
- Enabled but backend missing -> explicit `backend not configured` failure
- Backend configured -> real execution with structured success output

### Web Search Providers

Set `provider` to choose how web search works:

```yaml
actions:
  web_search:
    enabled: true
    provider: "searxng"               # searxng | tavily | custom
    endpoint: "http://localhost:8888"  # SearXNG instance URL
    api_key: ""                       # required for tavily
    timeout_ms: 3000
    default_top_k: 5
```

| Provider | Auth | Self-hosted | Setup |
|----------|------|-------------|-------|
| `searxng` | None | Yes | `brain deps up` (or `docker run -d -p 8888:8080 searxng/searxng`) |
| `tavily` | API key (free, no CC) | No | Sign up at tavily.com, set `api_key` |
| `custom` | None | — | Set `endpoint` to any JSON search API |

### Messaging (Webhooks)

Brain sends messages via configurable webhook URLs. Any service that accepts HTTP POST works — Slack, Discord, Telegram, ntfy.sh, custom endpoints:

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

Template placeholders: `{{content}}`, `{{channel}}`, `{{recipient}}`, `{{namespace}}`, `{{timestamp}}`. Values are JSON-escaped automatically. Custom `headers` are optional (useful for APIs requiring auth tokens).

### Backend Resilience

All HTTP backends (web search + messaging) share retry and circuit breaker settings:

```yaml
actions:
  resilience:
    max_retries: 2                    # retries on 5xx / timeout / connection refused
    retry_base_ms: 500                # exponential backoff: 500 → 1000 → 2000ms
    circuit_breaker_threshold: 5      # consecutive failures before circuit opens
    circuit_breaker_cooldown_secs: 60 # seconds before retrying after circuit opens
```

4xx errors (auth, bad request) fail immediately without retries. When a circuit opens, all requests to that backend return an instant error until the cooldown elapses.

### Other Action Backends

```yaml
  scheduling:
    enabled: false
    mode: "persist_only"        # intents stored in SQLite, no internal cron
```

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

The export format is a self-contained JSON file containing all facts and episodes with timestamps, importance scores, and namespace labels.

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

The default key for local use is `demokey123`. Change it before exposing Brain to a network.

## Configuration

Config is loaded from three sources (highest priority wins):

1. **Environment variables** — `BRAIN_LLM__MODEL=gpt-4o brain serve`
2. **User config** — `~/.brain/config.yaml`
3. **Defaults** — [`config/default.yaml`](config/default.yaml)

### LLM Provider

```yaml
llm:
  provider: "ollama"               # ollama | openai
  model: "qwen2.5-coder:7b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 4096
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
# Generate a salt and enable encryption setup
brain init --encrypt
```

Then set `encryption.enabled: true` in `~/.brain/config.yaml` and provide a passphrase:

```bash
# Via environment variable (for daemon/CI)
BRAIN_PASSPHRASE="your-passphrase" brain serve

# Or Brain will prompt interactively on startup
brain serve
```

## Data Directory

```
~/.brain/
├── config.yaml        # User configuration (overrides defaults)
├── db/
│   ├── brain.db       # SQLite — facts, episodes, procedures, FTS5 index
│   └── salt           # Encryption salt (only if --encrypt was used)
├── ruvector/          # HNSW vector index files
├── logs/
│   ├── brain.log      # Daemon logs
│   └── proactive.log  # Proactivity engine output (when enabled)
└── exports/           # Export output directory
```

## Re-initialise

```bash
# Wipe and re-create all data directories (keeps config)
brain init --force

# Also enable encryption
brain init --force --encrypt
```

## License

MIT
