# Brain OS — Product Vision

> This document captures the long-term vision, design philosophy, and product principles for Brain OS. Use it to evaluate feature proposals, resolve ambiguous design decisions, and stay aligned as the project evolves.

---

## The Problem

AI tools are proliferating, but memory is siloed. Your coding assistant doesn't know what your writing tool learned about you. Your shell scripts have no idea what you discussed with your AI chat. Each new session starts from zero. Each tool rediscovers the same preferences, the same context, the same facts you've shared a hundred times before.

This is the fundamental broken assumption of current AI tooling: that each application is the user's entire world.

---

## The Vision

**Brain OS is the memory layer that every AI application on your machine shares.**

It is not an AI assistant itself. It is the persistent, queryable, self-organising knowledge store that AI tools plug into. Think of it as the hippocampus — the part of the brain responsible for forming and retrieving memories — exposed as a local service.

When something is worth remembering, any tool can store it. When any tool needs context, it queries Brain. The memory compounds over time rather than being discarded at the end of each session.

---

## Design Philosophy

### 1. Local first, always

Brain runs on your machine. Your data never leaves. There is no account, no cloud, no telemetry. The database lives at `~/.brain/` and you own it entirely.

Privacy is not a feature — it is the architecture.

### 2. Protocol-agnostic

Brain does not know or care what application is talking to it. It speaks standard protocols: HTTP REST, WebSocket, gRPC, and MCP. Any tool that can make an HTTP request can use Brain. Any AI agent that speaks MCP can use Brain.

We do not build platform-specific adapters (Slack, Telegram, Notion, etc.) inside Brain. Those platforms connect to Brain via its existing protocols — the same way any other HTTP client would. This keeps Brain small, stable, and unopinionated about which applications matter.

### 3. One memory, many tools

All adapters share exactly one `SignalProcessor`. There is no per-adapter memory. A fact stored via MCP from your coding assistant is immediately visible via HTTP from your shell script. Consistency is not optional.

### 4. Memory that earns its place

Not all information is equally worth keeping. Brain uses importance scoring (amygdala), recency weighting, and a forgetting curve to continuously prune low-value memories. The goal is a memory that stays sharp, not one that accumulates noise.

Memory consolidation (planned) will run on a schedule to merge near-duplicates, promote reinforced facts, and prune forgotten ones — modelling how biological memory works during sleep.

### 5. Open to any LLM

Brain uses whichever LLM you configure — local models via Ollama, OpenAI, OpenRouter, or any OpenAI-compatible endpoint. The LLM is pluggable via the `LlmProvider` trait. Brain should be usable entirely offline with a local model.

---

## What Brain Is Not

| Not this | Why |
|----------|-----|
| An AI assistant | Brain stores and recalls — it doesn't have a personality or agenda |
| A cloud service | Runs locally; no dependency on external infrastructure |
| A platform integration hub | Brain exposes protocols; external apps connect to those |
| A replacement for tool-specific context windows | Brain augments context — it doesn't replace in-session context |
| An automation engine | Cerebellum (procedures) is minimal; complex workflows belong elsewhere |

---

## Who It's For

Brain OS is built for people who use multiple AI tools daily and are frustrated that those tools can't share context. The target user:

- Runs multiple AI-assisted workflows: coding, writing, research, shell automation
- Values privacy and wants data on their own machine
- Is comfortable editing a YAML config file
- Uses standard developer tools (REST APIs, MCP clients, CLI)

It is not trying to be a consumer product with a polished GUI. The web UI at `/ui` is a diagnostic tool for power users, not the primary interface.

---

## Protocols as the Product

The HTTP API, WebSocket, MCP, and gRPC interfaces are not "how you access Brain" — they _are_ Brain's product. A well-designed, stable, authenticated API that any tool can use is the core deliverable.

This means:
- Breaking changes to protocols need versioning (`/v1/`, `/v2/`)
- The OpenAPI spec must stay accurate
- Authentication must be consistent across all adapters
- Error responses must be machine-readable

---

## Memory Model

Brain stores three kinds of memory, mirroring how human memory is structured:

| Type | Crate | What it stores |
|------|-------|----------------|
| **Episodic** | hippocampus | Timestamped conversation history — what happened and when |
| **Semantic** | hippocampus | Subject–predicate–object facts — what is true |
| **Procedural** | cerebellum | Trigger → action patterns — what to do when |

Retrieval is hybrid: vector similarity (HNSW) + keyword matching (BM25 FTS5), fused with Reciprocal Rank Fusion. Results are reranked by importance and recency before being used as LLM context.

---

## Namespaces

Memory can be scoped to a namespace. The default is `"personal"`. Namespaces allow:
- Project-specific facts that don't pollute personal memory
- Tenant isolation in multi-user scenarios
- Clean separation of domains (work, personal, a specific codebase)

Namespaces are a first-class concept in the schema, not a tag bolted on later.

---

## Roadmap Principles

When evaluating what to build next, apply this order of priority:

1. **Correctness and safety first** — graceful shutdown, data integrity, no silent data loss
2. **Core memory quality** — embedding accuracy, search relevance, forgetting curve tuning
3. **Protocol completeness** — streaming responses, richer MCP tool set, stable gRPC contract
4. **Intelligence features** — proactivity (ganglia), consolidation, pattern detection
5. **Connectivity** — external app integration via existing protocols (HTTP, WS, MCP, gRPC)
6. **Developer experience** — better error messages, metrics, config validation

Features that require baking in a specific platform (Slack, Telegram, GitHub, etc.) belong in external adapters built by users — not in this codebase.

---

## Success Criteria

Brain OS succeeds when:

- A developer can run `brain start` and have all their AI tools share memory without any additional configuration
- Memory stored by one tool is immediately available to any other tool via its preferred protocol
- The system has been running for months with no manual intervention, a growing knowledge base, and search results that still feel relevant and fresh
- A new protocol adapter can be built in a few hours by following the architecture guide, without touching the core memory engine
