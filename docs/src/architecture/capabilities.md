# Capability System

Brain's capability system is the unified substrate for everything the system can do.

## Capability Registry

All capabilities are registered in a single `ToolRegistry`. Three producers feed into it:

1. **Native backends** — built-in capabilities declared by each backend module
2. **MCP mounts** — external MCP servers registered at runtime
3. **Skill packs** — (future) declarative capability bundles

Every capability has:
- **ID** — unique identifier
- **Safety tier** — Read / Write / Execute / Destructive / External
- **Preconditions** — what must be true for it to work
- **When-to-use** — guidance for the LLM
- **Embedding** — semantic descriptor for hybrid retrieval

## Capability Discovery

The `IntentRouter` and `CapabilityIndex` route intents to registered capabilities using hybrid (cosine + keyword) scoring. A learned `CapabilityFitnessStore` tracks per-tool success/failure and provides a bounded tiebreaker in ranking.

## Runtime Health

Capabilities carry runtime health state:
- **Verified** — working normally
- **Degraded** — dependency unavailable (e.g., embedder down for `memory.store`)
- **BreakerOpen** — circuit breaker tripped
- **PreconditionFailed** — missing prerequisite

The capability digest renders all health state in chat, and `tools/list` annotates per-tool status.
