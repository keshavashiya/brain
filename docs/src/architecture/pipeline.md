# Signal Pipeline

Every input to Brain flows through a single pipeline:

```
Input → Intent Classification → Authorization → Importance Scoring
    → Memory Store/Recall → LLM Response → Output
```

## Processing stages

1. **Signal Ingestion** — signals arrive via any adapter (HTTP, WS, gRPC, MCP, CLI) as a typed `Signal` carrying content, namespace, principal, and metadata.

2. **Intent Classification** — the `Thalamus` classifies each signal into one of 31 intent variants using a regex fast-path with async LLM fallback and timeout.

3. **Authorization** — the `IdentityStore` enforces tier-based authorization on every signal. The pipeline gate runs after classification, checking the principal's rights against the required tier.

4. **Importance Scoring** — the `Amygdala` scores memories on a [0,1] scale using keyword heuristics + per-process novelty detection. No LLM cost.

5. **Memory** — the `Hippocampus` handles storage and recall, combining BM25 FTS5 full-text search with HNSW vector search, fused via Reciprocal Rank Fusion.

6. **LLM Response** — the `Cortex` builds a token-budgeted prompt, invokes the configured LLM provider (with failover chain), and streams the response.

## The Capability Loop

For autonomous actions (tool calls), Brain runs a consent-gated tool loop:

```
LLM Response → Tool Call Request → Authorization → Confirmation
    → Execution → Audit → Result → LLM (next turn)
```

Each tool is a registered capability with a safety tier. Destructive/external actions require user confirmation via nonce-based approval flow.
