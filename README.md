# Brain OS

A **Central AI Operating System** written in pure Rust — a unified memory hub that any AI tool can connect to via HTTP, WebSocket, MCP, gRPC, or CLI.

Instead of each tool maintaining isolated context, Brain OS acts as a single source of truth. Any application — scripts, agents, coding tools, chat interfaces — can read from and write to one shared memory that grows smarter over time.

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25), fuses results via RRF, and reranks by importance and recency.

## Install

```bash
cargo install --path crates/cli
brain init
```

`brain init` creates `~/.brain/` with the default config and data directories.

## Usage

```bash
# Start Brain as a background daemon (all adapters, survives terminal close)
brain start

# Stop the daemon
brain stop

# System status (shows whether daemon is running)
brain status

# Interactive chat with memory
brain chat
```

## MCP Integration

Any MCP-compatible client can connect to Brain by adding it as a server. Run `brain mcp` and register it as a stdio server in your client's config:

```json
{
  "mcpServers": {
    "brain-memory": {
      "command": "brain",
      "args": ["mcp"]
    }
  }
}
```

Brain also exposes MCP over HTTP (`brain serve --mcp`) for clients that prefer HTTP transport instead of stdio.

### MCP Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `top_k?`, `namespace?` | Semantic + full-text hybrid search |
| `memory_store` | `subject`, `predicate`, `object`, `category`, `namespace?` | Store a fact |
| `memory_facts` | `subject` | All facts about a subject |
| `memory_episodes` | `limit?` | Recent conversation history |
| `user_profile` | — | Current user configuration |

## HTTP API

Default port: `19789`. All `/v1/*` routes require `Authorization: Bearer <key>`.

```bash
# Health (no auth)
curl http://localhost:19789/health

# Store a fact via signal
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"source":"http","sender":"user","content":"remember that I prefer dark mode"}'

# Search memory
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"query":"UI preferences","top_k":5}'

# List all facts
curl http://localhost:19789/v1/memory/facts \
  -H "Authorization: Bearer demo-key-123"
```

### Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/v1/signals` | Submit a signal |
| `GET` | `/v1/signals/:id` | Get cached response |
| `POST` | `/v1/memory/search` | Semantic search |
| `GET` | `/v1/memory/facts` | List all facts |
| `GET` | `/v1/memory/namespaces` | Namespace stats |

## Services & Ports

`brain start` launches all adapters together in the background. They share a single processor so memory is consistent across all protocols.

| Adapter | Default Port | Notes |
|---------|-------------|-------|
| HTTP REST API | 19789 | |
| WebSocket | 19790 | |
| MCP HTTP | 19791 | |
| gRPC | 19792 | |
| MCP stdio | stdin/stdout | `brain mcp` — stdio transport for MCP clients |

**For development**, `brain serve` runs everything in the foreground with optional adapter flags:

```bash
brain serve               # all adapters (foreground)
brain serve --http        # HTTP only
brain serve --http --ws   # HTTP + WebSocket
```

## Memory Namespaces

Scope facts and episodes to a specific context. The default namespace is `"personal"`.

```bash
# Store in a project namespace
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"source":"http","sender":"user","content":"use bun not npm","namespace":"my-project"}'

# Search within that namespace
curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"query":"package manager","namespace":"my-project"}'
```

## Authentication

Each adapter uses a different auth mechanism:

| Adapter | Method |
|---------|--------|
| HTTP | `Authorization: Bearer <key>` |
| WebSocket | First frame: `{"api_key":"<key>"}` |
| MCP HTTP | `x-api-key: <key>` header |
| MCP stdio | `params._meta["x-api-key"]` |
| gRPC | gRPC metadata |

Configure keys in `~/.brain/config.yaml`:

```yaml
access:
  api_keys:
    - key: "your-secret-key"
      name: "My Key"
      permissions: [read, write]
```

The default key shipped for local use is `demo-key-123`.

## Configuration

Config is loaded from three sources (highest priority wins):

1. Environment variables — `BRAIN_LLM__MODEL=gpt-4o`
2. User config — `~/.brain/config.yaml`
3. Defaults — [`config/default.yaml`](config/default.yaml)

Key settings:

```yaml
llm:
  provider: "ollama"       # ollama | openai
  model: "qwen2.5:14b"
  base_url: "http://localhost:11434"

embedding:
  model: "bge-small-en-v1.5"
  device: "auto"           # auto | cpu | cuda | coreml
```

## Data Directory

```
~/.brain/
├── db/brain.db        # SQLite — facts, episodes, profiles, FTS5 index
├── ruvector/          # Vector index (HNSW)
├── models/            # ONNX embedding models
├── logs/
└── exports/
```

## License

MIT
