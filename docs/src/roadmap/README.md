# Roadmap

> This page summarizes the public-facing plan.

## Active: Close the Loops

The current development focus is closing feedback loops across the system. Derived from a four-lens architecture review, the work is organized into tracks:

### Track 0 — Ship & hygiene ✅
Tag and publish v0.5.0, fence tool outputs as untrusted, quarantine MCP servers on hash change, add clock to prompt, fix architecture doc drift.

### Track 1 — Trustworthy substrate
Namespace data-residency enforcement, memory-trust (provenance-weighted recall + unattested-writer quarantine), grants ledger, TTL/scoped standing approvals, encrypted export, semantic capability retrieval. *Pre-requisite for inviting third-party connectors and writers.*

### Track 2 — Situated kernel
Hardware self-model, model tiers with per-task routing, connectivity and power awareness as kernel state, observation-to-graph mirroring, live manifest health, per-turn telemetry.

### Track 3 — Knowing companion
Discovery reflex, learned-normal monitoring, answer-quality fitness, project/workspace model, composable config.

### Track 4 — Human surface
Studio as trust console, goal model, presence awareness.

## After v1.0.0

- Multi-device CRDT sync (v1.1)
- IDE integration (v1.1)
- Dual-memory unification (v1.1)
