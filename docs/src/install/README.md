# Installation

## Requirements

- [Ollama](https://ollama.com) (or any OpenAI-compatible API)
- Rust 1.82+ (only for building from source)
- Docker (optional — for SearXNG web search backend)

## From crates.io (recommended)

```bash
cargo install brainos          # requires Rust 1.82+
brain init                     # creates ~/.brain/ with config, database, vector index
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
brain deps up                  # optional: upgrade web search from DuckDuckGo (default) to SearXNG
```

## From source

```bash
git clone https://github.com/keshavashiya/brain.git && cd brain
cargo install --path crates/cli
brain init
```

## One-liner (pre-built binary)

```bash
curl -fsSL https://raw.githubusercontent.com/keshavashiya/brain/main/scripts/install.sh | sh
```

This downloads a pre-built binary when available, falling back to `cargo install` from source.

## External services & auto-start

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

## Verify your install

```bash
brain doctor   # verify Ollama, models, ports — fix anything red
brain start    # wake the daemon
brain status   # check daemon health
```
