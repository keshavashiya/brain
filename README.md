# Brain

A personal AI that remembers, learns, and acts — inspired by the human brain's architecture.

Brain is a local-first AI assistant that stores all your conversations, extracts facts, learns your preferences, and proactively helps you — while keeping everything encrypted on your machine.

## Architecture

```
┌─────────────┐
│   CLI / API  │  ← You talk here
├─────────────┤
│  Thalamus    │  ← Routes signals (intent classification)
├──────┬──────┤
│Cortex│Amyg. │  ← Thinks (LLM) + Feels (importance scoring)
├──────┴──────┤
│ Hippocampus  │  ← Remembers (episodic + semantic memory)
├─────────────┤
│   Storage    │  ← SQLite + LanceDB + AES-256-GCM encryption
└─────────────┘
```

| Crate | Role |
|-------|------|
| `cli` | Binary — `brain chat`, `brain status`, `brain daemon` |
| `core` | Orchestrator — config, subsystem wiring |
| `thalamus` | Signal router — intent classification |
| `cortex` | LLM client — Ollama/OpenAI, context assembly, tool dispatch |
| `amygdala` | Importance scoring — keyword-based for v1 |
| `hippocampus` | Memory engine — episodic, semantic, procedural, hybrid search |
| `cerebellum` | Procedure store — learned workflows (Phase 3) |
| `ganglia` | Habit engine — pattern detection, proactive triggers (Phase 3) |
| `bridge` | OpenClaw integration — multi-channel messaging (Phase 3) |
| `storage` | SQLite + LanceDB + encryption |

## Quick Start

```bash
# Build
cargo build --workspace

# Run
cargo run --bin brain -- status
cargo run --bin brain -- chat

# Test
cargo test --workspace

# Install globally
cargo install --path crates/cli
brain status
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
├── lance/          # LanceDB (vector embeddings)
├── models/         # ONNX models (downloaded on first run)
├── logs/           # Log files
└── exports/        # Memory exports
```

## Documentation

- [`docs/TECHNICAL_SPECS.md`](docs/TECHNICAL_SPECS.md) — Architecture, schemas, security
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — 12-week build plan
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — Development setup and commands

## License

MIT
