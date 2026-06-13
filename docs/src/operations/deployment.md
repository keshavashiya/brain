# Running & Deployment

## Daemon lifecycle

```bash
brain start      # Start (or via service if installed)
brain stop       # Stop
brain status     # Health check
brain tail       # Stream events (observability)
```

## Service management

```bash
brain service install    # launchd (macOS) / systemd (Linux) / Task Scheduler (Windows)
brain service uninstall  # Remove auto-start
```

## Docker

Brain can run with Docker for the optional SearXNG web search backend:

```bash
brain deps up       # Start SearXNG
brain deps status   # Check
brain deps down     # Stop
```

## Data layout

Paths created at `brain init`:
- `~/.brain/config.yaml` — user config
- `~/.brain/db/brain.db` — SQLite database
- `~/.brain/ruvector/` — HNSW vector store
- `~/.brain/vault/` — encrypted credentials
- `~/.brain/logs/brain.log`

## Encryption

Enable encryption-at-rest with `brain init --encrypt`. This uses AES-256-GCM with Argon2id key derivation. Encrypted exports are the default when at-rest encryption is enabled.
