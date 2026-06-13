# Architecture Overview

Brain OS is built as a collection of specialized crates, each modelling a biological brain structure. The system is organized around a single `SignalProcessor` that all adapters share.

## Crate Map

```
brain/
├── crates/
│   ├── core/           # BrainConfig + shared config types
│   ├── signal/         # SignalProcessor — the single shared engine
│   ├── thalamus/       # Intent classification (regex + LLM fallback)
│   ├── amygdala/       # Importance scoring
│   ├── hippocampus/    # Memory engine (episodic + semantic + search)
│   ├── cortex/         # Reasoning core (LLM, context, action dispatch)
│   ├── cerebellum/     # Procedural memory (trigger→patterns)
│   ├── ganglia/        # Proactivity / habit engine
│   ├── audit/          # Append-only audit trail
│   ├── confirm/        # Confirmation engine (nonce-based)
│   ├── budget/         # Cost/token budget enforcement
│   ├── sandbox/        # Command execution sandbox
│   ├── vault/          # Credential vault
│   ├── orchestrate/    # Task decomposition + execution
│   ├── delegate/       # External agent delegation
│   ├── channel/        # Channel routing + presets
│   ├── observe/        # Observability bus + BrainEvent
│   ├── identity/       # Principal, tier, authorization
│   ├── intent/         # Intent Token + capability routing
│   ├── mcphost/        # MCP host for external servers
│   ├── reflex/         # Reactive signal sources
│   ├── resilience/     # Circuit breaker, retry, rate limit
│   ├── storage/        # SQLite pool + migrations
│   ├── backends/       # World-touching backends
│   ├── selfmodel/      # Self-model (host, capability, connectivity)
│   ├── metrics/        # Performance metrics
│   ├── bridge/         # Bridge library for external relays
│   ├── adapters/       # Transport adapters (HTTP, WS, gRPC, MCP, Terminal)
│   └── cli/            # CLI binary (thin wrapper over backends)
├── docs/               # Documentation (mdBook)
├── scripts/            # Build + release scripts
└── docker/             # Docker compose for SearXNG
```

## Design Principle: One Capability, Many Faces

A **capability** is a typed entry — id, safety tier, preconditions — in a single registry. All transports (CLI, HTTP, WS, gRPC, MCP) and the resident reasoner are **faces** over that one registry. They hold no private capabilities and no business logic.
