# Product Vision

Brain OS is a biologically-inspired, central cognitive engine — the persistent memory and mediation layer between AI tools and the user's world.

## The vision in one sentence

**Your AI tools share one localized, ever-growing memory that runs 24/7 on your machine — and they all play by the same rules.**

## Core principles

1. **Local-first, always** — Your data never leaves your hardware. There is no account, no cloud, no telemetry.
2. **One capability ontology** — A capability is a typed entry in one registry. CLI, HTTP, MCP, and the reasoner are faces over it.
3. **Memory that earns its place** — Importance scoring, forgetting curves, and consolidation keep the signal sharp.
4. **Fail safe, never silently** — Every error path is explicit. Degraded-but-functional is the target.
5. **Open to any LLM** — Ollama, OpenAI, OpenRouter, or any OpenAI-compatible endpoint.

## Version arc

| Version | Focus | Status |
|---------|-------|--------|
| v0.1.0 | Memory layer (episodic + semantic + procedural) | ✅ Released |
| v0.2.0 | Autonomous agent layer (safety, isolation, orchestration, delegation) | ✅ Released |
| v0.3.0 | Natural language interface (chat, intents, approvals) | ✅ Released |
| v0.4.0 | Wire pillars + fix stubs (all 30+ crates wired) | ✅ Released |
| v0.5.0 | Structural polish (release automation, capability coherence) | ✅ Released |
| v0.6.0 | The Connector (SDKs, connector protocol, situated kernel) | 🔴 In progress |
| v0.7.0 | Skill packs (declarative capability bundles) | 🔴 Planned |
| v0.8.0 | Multi-device (cuttable) | 🔴 Planned |
| v0.9.0 | Brain Studio (trust console) | 🔴 Planned |
| v1.0.0 | Launch — the private home for your AI life | 🔴 Planned |

## What Brain is not

- A cloud service
- A platform integration hub
- A consumer product with a polished GUI
- A replacement for tool-specific context windows

The web UI at `/ui` is a diagnostic tool for power users, not the primary interface.
