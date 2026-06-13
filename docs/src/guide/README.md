# Quick Start

## Recommended setup order

```bash
# 1. Initialize (one-time)
brain init

# 2. Quick test — direct daemon
brain start
brain status
brain stop

# 3. Production — auto-start on login
brain service install    # registers launchd/systemd/Task Scheduler
# Brain now wakes automatically on every login
```

## Lifecycle commands

```bash
brain start    # Start daemon
brain stop     # Stop daemon
brain status   # Check daemon status
brain tail     # Stream BrainEvent bus (observability tap for headless/SSH)
```

## Interactive usage

```bash
brain chat                           # Interactive chat
brain chat "remember that I use bun" # One-shot message
```

## Foreground mode (development)

```bash
brain serve               # All adapters (foreground)
brain serve --http        # HTTP only
brain serve --http --ws   # HTTP + WebSocket
brain serve --grpc        # gRPC only
brain serve --mcp         # MCP HTTP only
```

## Checking memory

```bash
# Search memory
brain chat "what do I know about Rust?"

# Store a fact
brain chat "remember that my favorite editor is Neovim"

# List grants
brain chat "show me my grants"
```
