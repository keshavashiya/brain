# Brain OS 🧠

**Stop giving your AI amnesia.**

Brain OS is a biologically-inspired, central cognitive engine written in pure Rust. Instead of every script, coding assistant, and chat UI keeping its own isolated, fragmented context, Brain OS acts as your single source of truth.

It routes intents through a **Thalamus**, scores importance via an **Amygdala**, and stores everything in a unified **Hippocampus** (FTS5 + HNSW Vector Search). Whether you connect via HTTP, WebSocket, gRPC, or MCP, your AI tools now share one localized, ever-growing memory that runs 24/7 on your machine.

*Your data never leaves your hardware. Your AI never forgets.*

---

## How It Works

Every input — regardless of protocol — flows through the same pipeline:

```
Input → Intent Classification → Importance Scoring → Memory Store/Recall → LLM Response
```

The memory engine combines vector search (HNSW) with full-text search (BM25 FTS5), fuses results via Reciprocal Rank Fusion, and reranks by importance and recency. A forgetting curve runs every 24 hours to prune low-value memories and promote reinforced episodes to permanent semantic facts.

### Beyond memory: the kernel it grew into

Memory is the hook — but the same daemon also mediates what your AI tools can do. Every capability it exposes — search the web, run a sandboxed command, send a notification, probe a host, audit its own config — is a typed entry in one **capability manifest**, each tagged with a safety tier and routed through the same consent, audit, and budget gates. Whether a request comes from your terminal, an MCP client, or Brain's own resident reasoner, it sees the same manifest and is held to the same rules.

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Local-first** | Runs on your machine. No cloud, no telemetry, no account. |
| **Protocol-agnostic** | HTTP, WebSocket, gRPC, MCP — one memory behind every surface. |
| **Memory that earns its place** | Importance scoring + forgetting curve keep the signal sharp. |
| **Open to any LLM** | Ollama, OpenAI, OpenRouter, or any OpenAI-compatible endpoint. |
| **Fail safe, never silently** | Degraded-but-functional is the target state. |
