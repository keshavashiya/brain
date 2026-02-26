# Brain OS

A **Central AI Operating System** that provides unified memory and multi-protocol access for AI applications.

## Vision

Brain OS serves as the central memory hub for AI tools via standard protocols:
- **HTTP** - REST API access
- **WebSocket** - Real-time bidirectional communication
- **MCP** - Claude Code, OpenCode, and other MCP clients
- **CLI** - Interactive terminal interface
- **gRPC** - Programmatic access

All AI tools connect through protocol adapters - the SignalProcessor is agnostic to the source.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ADAPTERS (Sensors - like ears/eyes)           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   CLI   │  │  HTTP   │  │   WS   │  │   MCP   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │               │
│       └────────────┴────────────┴────────────┴────────────    │
│                             │                                 │
│                    Unified Signal Pipeline                     │
│                             │                                 │
│       ┌─────────────────────┴─────────────────────┐          │
│       │           Thalamus (Intent Router)        │          │
│       └─────────────────────┬─────────────────────┘          │
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │ Cortex + Amygdala        │  (Reasoning + Importance)    │
│  └──────────────────────────┼────────────────────────────────┤
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │     Hippocampus (Memory) │                                │
│  │  Episodic + Semantic + Recall Engine                     │
│  └──────────────────────────┬────────────────────────────────┤
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │ Storage: SQLite + RuVector (HNSW + GNN + Self-Learning)   │
└─────────────────────────────────────────────────────────────────┘
```
┌─────────────────────────────────────────────────────────────────┐
│                    SignalProcessor (Hub)                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │   CLI   │  │  HTTP   │  │   WS   │  │   MCP   │        │
│  │ Adapter │  │ Adapter │  │ Adapter │  │ Adapter │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │               │
│       └────────────┴────────────┴────────────┴────────────    │
│                             │                                 │
│                    Unified Signal Pipeline                     │
│                             │                                 │
│       ┌─────────────────────┴─────────────────────┐          │
│       │           Thalamus (Intent Router)        │          │
│       └─────────────────────┬─────────────────────┘          │
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │ Cortex + Amygdala        │  (Reasoning + Importance)    │
│  └──────────────────────────┼────────────────────────────────┤
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │     Hippocampus (Memory) │                                │
│  │  Episodic + Semantic + Recall Engine                     │
│  └──────────────────────────┬────────────────────────────────┤
│                             │                                 │
│  ┌──────────────────────────┼────────────────────────────────┤
│  │ Storage: SQLite + RuVector (HNSW + GNN + Self-Learning)   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Central Memory** | Unified memory accessible from all AI tools |
| **Self-Learning** | Memory improves over time via RuVector GNN |
| **Multi-Protocol** | HTTP, WebSocket, MCP, gRPC, CLI |
| **Local-First** | All data stays on your machine |

## Protocol Ports

| Service | Port | Purpose |
|---------|------|---------|
| HTTP API | 19789 | REST API |
| WebSocket | 19790 | Real-time |
| MCP HTTP | 19791 | Remote MCP |
| gRPC | 19792 | Programmatic |

## Crate Structure

| Crate | Role |
|-------|------|
| `core` | Orchestration and config |
| `signal` | SignalProcessor hub |
| `adapters/cli` | CLI input adapter (like ears) |
| `adapters/*` | Protocol adapters (HTTP, WS, MCP, gRPC) |
| `thalamus` | Intent classification |
| `cortex` | LLM reasoning |
| `amygdala` | Importance scoring |
| `hippocampus` | Memory engine |
| `storage` | SQLite + RuVector |

## Quick Start

```bash
# Build
cargo build --workspace

# Run HTTP server
cargo run --bin brain -- serve

# Run MCP (for Claude Code)
cargo run --bin brain -- mcp --stdio

# CLI chat
cargo run --bin brain -- chat

# Test
cargo test --workspace

# Install globally
cargo install --path crates/cli
brain serve
```

## Configuration

Brain loads config from multiple sources (highest priority wins):

1. **Environment variables**: `BRAIN_LLM__MODEL=gpt-4o`
2. **User config**: `~/.brain/config.yaml`
3. **Default config**: `config/default.yaml`

See [`config/default.yaml`](config/default.yaml) for all available options.

## Data Directory

All data lives in `~/.brain/`:

```
~/.brain/
├── db/brain.db     # SQLite (episodes, facts, profile)
├── ruvector/       # RuVector (vector embeddings + HNSW index)
├── models/         # ONNX models (downloaded on first run)
├── logs/           # Log files
└── exports/        # Memory exports
```

## Documentation

- [`docs/IMPLEMENTATION_PLAN_V2.md`](docs/IMPLEMENTATION_PLAN_V2.md) — Central AI OS implementation plan
- [`docs/TECHNICAL_SPECS.md`](docs/TECHNICAL_SPECS.md) — Architecture, schemas, security
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — Development setup and commands

## License

MIT
