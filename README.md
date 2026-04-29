# Brain OS 🧠

[![Crates.io](https://img.shields.io/crates/v/brainos.svg)](https://crates.io/crates/brainos)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Stop giving your AI amnesia.**

Brain OS is a biologically-inspired, central cognitive engine written in pure Rust. Instead of every script, coding assistant, and chat UI keeping its own isolated, fragmented context, Brain OS acts as your single source of truth.

It routes intents through a Thalamus, scores importance via an Amygdala, and stores everything in a unified Hippocampus (FTS5 + HNSW Vector Search). Whether you connect via HTTP, WebSocket, gRPC, or MCP, your AI tools now share one localized, ever-growing memory that runs 24/7 on your machine.

*Your data never leaves your hardware. Your AI never forgets.*

---

## Quick Start

```bash
cargo install brainos && brain init
ollama pull qwen2.5-coder:7b && ollama pull nomic-embed-text
brain start
brain chat "remember that I use dark mode"
```

See [full install guide](#install) below for details.

---

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25 FTS5), fuses results via Reciprocal Rank Fusion, and reranks by importance and recency. A forgetting curve runs every 24 hours to prune low-value memories and promote reinforced episodes to permanent semantic facts.

---

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [MCP Integration](#mcp-integration)
- [HTTP API](#http-api)
- [Services & Ports](#services--ports)
- [Memory Namespaces](#memory-namespaces)
- [Agent Delegation](#agent-delegation)
- [Background Intelligence](#background-intelligence)
  - [Memory Consolidation](#memory-consolidation)
  - [Proactivity Engine](#proactivity-engine)
  - [Messaging, Webhooks & Transports](#messaging-webhooks--transports)
- [Action Backends](#action-backends-internal)
- [Authentication](#authentication)
- [Configuration](#configuration)
- [Export & Import](#export--import)
- [Channel Integrations](#channel-integrations)
- [Development](#development)

---

## Install

**Requirements:** [Ollama](https://ollama.com) (or any OpenAI-compatible API). Docker is optional — only needed if you want to upgrade the default DuckDuckGo web search to the higher-quality bundled SearXNG container.

<details>
<summary><strong>From crates.io (recommended)</strong></summary>

```bash
cargo install brainos          # requires Rust 1.82+
brain init                     # creates ~/.brain/ with config, database, vector index
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
brain deps up                  # optional: upgrade web search from DuckDuckGo (default) to SearXNG on port 8888
```

</details>

<details>
<summary><strong>From source</strong></summary>

```bash
git clone https://github.com/keshavashiya/brain.git && cd brain
cargo install --path crates/cli
brain init
```

</details>

<details>
<summary><strong>External services & auto-start</strong></summary>

**Docker (optional web search):**
```bash
brain deps up       # Start SearXNG
brain deps status   # Check if running
brain deps down     # Stop
```

**Auto-start on login:**
```bash
brain service install    # launchd (macOS) / systemd (Linux) / Task Scheduler (Windows)
brain service uninstall  # Remove
```

</details>

---

## Usage

The CLI covers lifecycle (`init`, `start`, `stop`, `status`, `serve`, `mcp`, `service`, `deps`) and security-sensitive input (`vault`). Everything else — recall, approvals, budgets, schedules, audit queries, task decomposition, export/import — goes through `brain chat`.

### Lifecycle commands

```bash
brain start                          # Start daemon (or via service if installed)
brain stop                           # Stop daemon (also stops service if installed)
brain status                         # Check daemon status via HTTP health probe
```

**Recommended setup order:**

```bash
# 1. Initialize (one-time)
brain init

# 2. Quick test — direct daemon
brain start
brain status
brain stop

# 3. Production — auto-start on login
brain service install    # registers launchd/systemd/Task Scheduler
# Brain now wakes automatically on every login — no `brain start` needed
brain service uninstall  # remove auto-start
```

### Interactive usage

```bash
brain chat                           # Interactive chat
brain chat "remember that I use bun" # One-shot message
```

<details>
<summary><strong>Foreground mode for development</strong></summary>

```bash
brain serve               # All adapters (foreground)
brain serve --http        # HTTP only
brain serve --http --ws   # HTTP + WebSocket
brain serve --grpc        # gRPC only
brain serve --mcp         # MCP HTTP only
```

</details>

---

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

<details>
<summary><strong>MCP Tools</strong></summary>

| Tool | Arguments | Description |
|------|-----------|-------------|
| `memory_search` | `query`, `top_k?`, `namespace?` | Hybrid semantic + full-text search |
| `memory_store` | `subject`, `predicate`, `object`, `category`, `namespace?` | Store a semantic fact |
| `memory_facts` | `subject`, `namespace?` | All facts about a subject |
| `memory_episodes` | `limit?` | Recent conversation history |
| `user_profile` | — | Current user configuration |
| `memory_procedures` | `action`, `trigger?`, `steps?`, `procedure_id?` | Manage learned workflows |

**Auth:** MCP stdio passes auth in `_meta.x-api-key`; HTTP uses `x-api-key` header.

</details>

---

## HTTP API

Default port: `19789`. All `/v1/*` routes require `Authorization: Bearer <key>`.

<details>
<summary><strong>Routes</strong></summary>

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/ui` | Browser UI (diagnostic) |
| `GET` | `/openapi.json` | OpenAPI spec |
| `GET` | `/api` | Swagger UI |
| `POST` | `/v1/signals` | Submit a signal |
| `GET` | `/v1/signals/:id` | Poll cached response |
| `POST` | `/v1/memory/search` | Hybrid semantic search |
| `GET` | `/v1/memory/facts` | List all facts |
| `GET` | `/v1/memory/namespaces` | Namespace stats |
| `GET` | `/v1/memory/export` | Export all memory to JSON |
| `POST` | `/v1/memory/import` | Import memory from JSON backup |
| `GET` | `/v1/schedules` | List scheduled intents |
| `DELETE` | `/v1/schedules/:id` | Cancel a scheduled intent |
| `GET` | `/v1/events` | SSE stream of proactive notifications |
| `POST` | `/v1/webhooks/:id` | Inbound webhook (channel transport) |

**Example:**
```bash
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"I prefer dark mode"}'

curl -X POST http://localhost:19789/v1/memory/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"UI preferences","top_k":5}'

curl -N http://localhost:19789/v1/events \
  -H "Authorization: Bearer YOUR_API_KEY"
```

</details>

---

## Services & Ports

`brain start` launches all adapters together. They share a single processor so memory is consistent across all protocols.

<details>
<summary><strong>Adapter details</strong></summary>

| Adapter | Default Port | Notes |
|---------|-------------|-------|
| HTTP REST | 19789 | REST API + Web UI + Swagger + OpenAPI |
| WebSocket | 19790 | Bidirectional streaming, real-time |
| MCP HTTP | 19791 | MCP over HTTP transport |
| gRPC | 19792 | Protobuf RPC + server streaming |
| MCP stdio | stdin/stdout | `brain mcp` for subprocess MCP clients |

</details>

---

## Memory Namespaces

Scope facts and episodes to a context. The default namespace is `"personal"`.

```bash
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"use bun not npm","namespace":"my-project"}'
```

---

## Background Intelligence

`brain serve` and `brain start` spawn background tasks alongside the protocol adapters, sharing the same `SignalProcessor`.

### Memory Consolidation

Runs every 24 hours. Uses an Ebbinghaus forgetting curve to prune low-retention episodes and promote frequently-reinforced episodes to permanent semantic facts.

<details>
<summary><strong>Configuration</strong></summary>

```yaml
memory:
  consolidation:
    enabled: true          # on by default
    interval_hours: 24
    forgetting_threshold: 0.05   # episodes with retention < 5% are pruned
```

</details>

### Proactivity Engine

Enabled by default with conservative guardrails (max 2/day, wide quiet hours) — Brain is bidirectional out of the box, proactively reminding you of things instead of only responding when asked.

<details>
<summary><strong>Habit Detection & Open-Loop Detection</strong></summary>

**Habit Detection** — scans episodic memory for recurring patterns (keyword × day-of-week × hour histograms) and nudges you when a pattern matches the current time slot.

**Open-Loop Detection** — scans for unresolved commitments ("I need to...", "remind me to...", "I should...") and generates reminders when no resolution is found within the configured window.

**Delivery** — proactive messages are delivered through three tiers:
1. **Outbox** — written to SQLite, drained on next `brain chat` session (no background drain loop)
2. **Broadcast** — pushed to live WebSocket and SSE (`GET /v1/events`) sessions (capacity: 256)
3. **Webhooks** — pushed to configured messaging channels (Slack, Discord, Telegram, etc.)
   > Proactive webhook notifications always use `"personal"` namespace.

```yaml
proactivity:
  enabled: true            # on by default; set to false to disable
  max_per_day: 2
  min_interval_minutes: 60
  quiet_hours:
    start: "20:00"
    end: "10:00"
  delivery:
    outbox: true
    broadcast: true
    webhook_channels: []   # channel keys from actions.messaging.channels
    max_outbox_age_days: 7
  open_loop:
    enabled: true
    scan_window_hours: 72
    resolution_window_hours: 24
    check_interval_minutes: 120
```

</details>

### Agent Identity

Every signal can carry an `agent` field identifying the originating AI tool (e.g. `"claude-code"`, `"cursor"`). Agent identity flows through the entire pipeline — recall, habit detection, and proactive messages reference the originating agent when known.

```bash
curl -X POST http://localhost:19789/v1/signals \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"deploy staging server","agent":"devops-agent"}'
```

---

## Agent Delegation

Brain can delegate subtasks to external agents without embedding those runtimes into the daemon. The delegation layer is external-first and config-driven:

- `agents.auto_discovery`: scans `$PATH` for known agent CLIs and probes capabilities.
- `agents.discovery_overrides`: merges local overrides onto discovered agents.
- `agents.delegates[]`: registers manual subprocess-backed delegates.
- `agents.fallbacks[]`: defines ordered escalation targets when a delegate fails.

Use natural language through `brain chat` or MCP/HTTP/WS clients. Thalamus now includes both `DecomposeTask` and `QueryAgents`, so prompts like "break this task down" and "what agents do I have?" route through the same intent layer as the rest of Brain.

---

## Messaging, Webhooks & Transports

Brain has two messaging layers:

- `actions.messaging.channels`: simple outbound HTTP POST templates used by the internal `SendMessage` backend and proactive webhook pushes.
- `channel.transports[]`: preset-driven channel runtimes for long-poll and webhook-style platforms (Telegram, Discord interactions, Slack webhooks, and user-defined presets).

<details>
<summary><strong>Setting Up Outbound Webhook Channels</strong></summary>

Each channel key under `actions.messaging.channels` becomes a simple outbound destination for **proactive notifications** and **explicit SendMessage** intents ("send via discord to alice saying hello").

**Discord** — Channel Settings → Integrations → Webhooks → Create Webhook:
```yaml
channels:
  discord:
    url: "https://discord.com/api/webhooks/<WEBHOOK_ID>/<WEBHOOK_TOKEN>"
    body: '{"content": "{{content}}"}'
    headers: {}
```

**Telegram** — Create a bot via [@BotFather](https://t.me/botfather), get `CHAT_ID` by messaging your bot and visiting `https://api.telegram.org/bot<TOKEN>/getUpdates`:
```yaml
channels:
  telegram:
    url: "https://api.telegram.org/bot<BOT_TOKEN>/sendMessage"
    body: '{"chat_id": "<CHAT_ID>", "text": "{{content}}"}'
    headers: {}
```

**Slack** — Apps → Incoming Webhooks → Add to workspace:
```yaml
channels:
  slack:
    url: "https://hooks.slack.com/services/T00/B00/xxx"
    body: '{"text": "{{content}}"}'
    headers: {}
```

**Generic webhook** — any HTTP POST endpoint:
```yaml
channels:
  webhook:
    url: "https://hooks.example.com/services/brain"
    body: '{"channel": "{{channel}}", "message": "{{content}}", "ts": "{{timestamp}}"}'
    headers:
      X-API-Key: "your-secret-key"
```

**Shorthand** — URL only, uses default JSON body:
```yaml
channels:
  simple: "https://example.com/hook"
```

**Template Placeholders:** `{{channel}}`, `{{recipient}}`, `{{content}}` (auto JSON-escaped), `{{namespace}}`, `{{timestamp}}`.

Default body (when omitted): `{"channel":"{{channel}}","recipient":"{{recipient}}","content":"{{content}}","namespace":"{{namespace}}","timestamp":"{{timestamp}}"}`

**Proactive Delivery:** To receive habit patterns and open-loop reminders on your channels:
```yaml
proactivity:
  delivery:
    webhook_channels: ["discord", "telegram"]
```

</details>

<details>
<summary><strong>Preset-Driven Channel Transports</strong></summary>

Use `channel.transports[]` when the platform needs more than a one-way POST: long polling, signed inbound webhooks, response extraction, cursor tracking, or preset-specific URL/body templates.

```yaml
channel:
  transports:
    - id: "telegram-main"
      label: "Telegram"
      preset: "telegram"
      namespace: "personal"
      credential: "<BOT_TOKEN>"

    - id: "discord-main"
      label: "Discord Interactions"
      preset: "discord-interactions"
      namespace: "personal"
      credential: "<APPLICATION_ID>"
      signing_secret: "<ED25519_PUBLIC_KEY>"

    - id: "slack-main"
      label: "Slack Webhook"
      preset: "slack-webhook"
      namespace: "work"
      credential: "https://hooks.slack.com/services/T00/B00/xxx"
```

These transports are generic engines backed by presets in `crates/channel/presets/` or `~/.brain/presets/<id>.yaml`.

</details>

---

## Action Backends (Internal)

Action intents routed by Thalamus (`web_search`, `schedule_task`, `send_message`) are handled by internal `ActionDispatcher` backends. These are internal-only — no public HTTP or gRPC endpoints expose them directly.

<details>
<summary><strong>Web Search, Scheduling, Resilience</strong></summary>

**Web Search Providers:**

| Provider | Auth | Self-hosted | Setup | Notes |
|----------|------|-------------|-------|-------|
| `duckduckgo` | None | — | None — works out of box | Default. Single-engine HTML scrape; basic quality |
| `searxng` | None | Yes | `brain deps up` | Multi-engine metasearch; best quality |
| `tavily` | API key (free, no CC) | No | Sign up at tavily.com | Hosted, AI-tuned snippets |
| `custom` | None | — | Any OpenAI-compatible JSON search API | |

**Scheduling:**
```yaml
actions:
  scheduling:
    enabled: false
    mode: "persist_only"    # SQLite persist + background poller fires due intents
```

**Resilience** (shared by all HTTP backends):
```yaml
actions:
  resilience:
    max_retries: 2                     # retries on 5xx / timeout / connection refused
    retry_base_ms: 500                 # exponential backoff: 500 → 1000 → 2000ms
    circuit_breaker_threshold: 5       # consecutive failures before circuit opens
    circuit_breaker_cooldown_secs: 60  # seconds before retrying after circuit opens
```

4xx errors fail immediately without retries.

</details>

---

## Authentication

| Adapter | Method |
|---------|--------|
| HTTP REST | `Authorization: Bearer <key>` |
| WebSocket | First frame: `{"api_key":"<key>"}` |
| MCP HTTP | `x-api-key: <key>` header |
| MCP stdio | `params._meta["x-api-key"]` |
| gRPC | Interceptor checks `x-api-key` or `authorization` metadata |

<details>
<summary><strong>Configuring keys</strong></summary>

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

MCP stdio can also authenticate via the `BRAIN_API_KEY` environment variable.

</details>

---

## Configuration

Config is loaded from three sources (highest priority wins):

1. **Environment variables** — `BRAIN_LLM__MODEL=gpt-4o brain serve`
2. **User config** — `~/.brain/config.yaml`
3. **Defaults** — [`crates/core/default.yaml`](crates/core/default.yaml)

Double-underscore (`__`) is the nesting separator in env var names.

<details>
<summary><strong>LLM, Embedding, Encryption</strong></summary>

**LLM Provider:**
```yaml
llm:
  provider: "ollama"               # ollama | openai
  model: "qwen2.5-coder:7b"
  base_url: "http://localhost:11434"
  api_key: ""                      # required for openai provider; or set BRAIN_LLM__API_KEY
  temperature: 0.7
  max_tokens: 4096
```

**Embedding Model:**
```yaml
embedding:
  model: "nomic-embed-text"       # must be pulled: `ollama pull nomic-embed-text`
  dimensions: 768                  # must match the model output size
```

**Encryption (at-rest):**
```bash
brain init --encrypt
```
Then set `encryption.enabled: true` in `~/.brain/config.yaml` and provide a passphrase via `BRAIN_PASSPHRASE` env var or interactive prompt.

> **Note:** When encryption is enabled, FTS5 full-text search is disabled — hybrid search relies on vector similarity only.

</details>

---

## Export & Import

```bash
brain export > backup.json        # Export all memory
brain import backup.json --dry-run  # Preview what import would do
brain import backup.json            # Import from backup
```

Import is idempotent — re-importing the same backup is safe.

---

## Channel Integrations

Brain can integrate with external channels in two ways:

1. Built-in preset-driven transports via `channel.transports[]`.
2. External WebSocket bridges via `channel.relays[]`.

<details>
<summary><strong>Bridge Library & CLI</strong></summary>

```
External Platform           Bridge (your code / external repo)        Brain OS
────────────────────        ──────────────────────────────────        ────────────────
  Slack / Telegram    ────► BridgeClient (crates/bridge library) ──► ws://localhost:19790
  Custom chat agent          exponential-backoff reconnection          SignalProcessor
  Any WebSocket bot          thin message translation                  memory + LLM
```

The `crates/bridge/` library provides a `BridgeClient` for custom relays. It handles reconnection with exponential backoff, ping/pong keep-alive, and JSON message serialization.

Use this path when the platform speaks a custom WebSocket protocol or when you want the platform-specific bot logic to live outside the daemon. For HTTP/webhook-style platforms, prefer `channel.transports[]`.

**Bridge CLI command (optional compatibility path):**
```bash
brain bridge ws://localhost:8080/gateway --api-key YOUR_KEY
```

1. Connects to your external WebSocket gateway
2. Connects to Brain's WebSocket synapse internally
3. Relays messages bidirectionally between the gateway and Brain
4. Automatically handles reconnection with exponential backoff

</details>

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
brain init --force           # Regenerate config with new API key (data preserved)
brain init --force --encrypt # Also enable encryption
```

---

## Development

```bash
git clone https://github.com/keshavashiya/brain.git && cd brain
cargo build
cargo test
cargo run -p brainos -- chat "hello"
cargo run -p brainos -- serve --http --mcp
```

<details>
<summary><strong>Workspace Structure</strong></summary>

The project is a Cargo workspace with 24 crates. All internal dependencies use both `path` (for local development) and `version` (for crates.io), so no Cargo.toml changes are needed to switch between local and published builds.

```
crates/
├── core/           # brainos-core        — Config and bootstrapping
├── storage/        # brainos-storage     — SQLite + HNSW vector index
├── hippocampus/    # brainos-hippocampus — Episodic + semantic memory
├── cortex/         # brainos-cortex      — LLM providers + context assembly
├── thalamus/       # brainos-thalamus    — Intent classification (primary UI)
├── amygdala/       # brainos-amygdala    — Importance scoring
├── signal/         # brainos-signal      — Central signal processor
├── cerebellum/     # brainos-cerebellum  — Procedural memory
├── ganglia/        # brainos-ganglia     — Proactivity engine
├── backends/       # brainos-backends    — Resilience, search & messaging backends
├── bridge/         # brainos-bridge      — WebSocket relay client
├── audit/          # brainos-audit       — Append-only action audit trail
├── confirm/        # brainos-confirm     — Nonce-backed confirmations
├── budget/         # brainos-budget      — Cost/token budgets + circuit breaker
├── sandbox/        # brainos-sandbox     — Command execution sandbox
├── vault/          # brainos-vault       — OS-native credential vault
├── orchestrate/    # brainos-orchestrate — Task decomposition + execution DAG
├── delegate/       # brainos-delegate    — Agent discovery, registry, delegation, escalation
├── channel/        # brainos-channel     — Channel routing + learned preferences
├── adapters/
│   ├── http/       # brainos-httpadapter — Axum REST API
│   ├── ws/         # brainos-wsadapter   — WebSocket adapter
│   ├── grpc/       # brainos-grpcadapter — gRPC adapter
│   └── mcp/        # brainos-mcp         — MCP adapter
└── cli/            # brainos (binary: brain) — CLI entry point
```

</details>

<details>
<summary><strong>Publishing</strong></summary>

Crates must be published in dependency order (leaves first). Run `cargo publish` in this order:

```
core → storage → hippocampus → amygdala → cortex → thalamus → cerebellum → ganglia
     → audit → confirm → budget → sandbox → vault
     → orchestrate → delegate → channel
     → signal → backends → bridge → adapters/* → cli
```

</details>

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full internal design document covering key abstractions, data flow, storage layer, background loops, the bridge relay pattern for external integrations, and step-by-step guides for building new protocol adapters.

---

## License

[MIT](LICENSE)
