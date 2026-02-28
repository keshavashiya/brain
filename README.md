# Brain OS

A **Central AI Operating System** written in pure Rust — a unified memory hub that any AI tool can connect to via HTTP, WebSocket, MCP, gRPC, or CLI.

Instead of each tool keeping its own isolated context, Brain OS acts as a single source of truth. Scripts, agents, coding assistants, and chat interfaces all read from and write to one shared memory that grows smarter over time.

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25 FTS5), fuses results via Reciprocal Rank Fusion, and reranks by importance and recency.

## Install

**Requirements:** Rust 1.82+, [Ollama](https://ollama.com) (or any OpenAI-compatible API)

```bash
# Install the CLI
cargo install --path crates/cli

# Initialize data directory (~/.brain/)
brain init

# Pull the default LLM model (Ollama)
ollama pull qwen2.5:14b
```

`brain init` creates `~/.brain/` with config, database, vector index, and model directories. It also downloads the embedding model (`bge-small-en-v1.5`) automatically on first use.

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
brain chat -m "remember that I use dark mode"
```

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
| `memory_facts` | `subject` | All facts about a subject |
| `memory_episodes` | `limit?` | Recent conversation history |
| `user_profile` | — | Current user configuration |

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
| gRPC | `authorization` metadata |

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
  model: "qwen2.5:14b"
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
  model: "bge-small-en-v1.5"      # downloaded automatically
  device: "auto"                   # auto | cpu | cuda | coreml
```

### Encryption (at-rest)

```bash
# Enable AES-256-GCM encryption for stored memories
brain init --encrypt
```

```yaml
encryption:
  enabled: true
  salt: "<auto-generated>"
```

## Data Directory

```
~/.brain/
├── config.yaml        # User configuration (overrides defaults)
├── db/
│   └── brain.db       # SQLite — facts, episodes, profiles, FTS5 index
├── ruvector/          # HNSW vector index
├── models/            # ONNX embedding models (auto-downloaded)
├── logs/              # Daemon logs
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
