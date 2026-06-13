# Memory Model

Brain stores three kinds of memory, mirroring human memory structure:

| Type | What it stores | Storage |
|------|---------------|---------|
| **Episodic** | Timestamped conversation history | SQLite + FTS5 |
| **Semantic** | Subject–predicate–object facts | SQLite + HNSW vector |
| **Procedural** | Trigger → action patterns | SQLite |

## Retrieval

Memory retrieval is hybrid — combining vector similarity (HNSW) with keyword matching (BM25 FTS5), fused via **Reciprocal Rank Fusion (RRF)**. Results are reranked by importance and recency before being used as LLM context.

## Forgetting Curve

```
retention = importance × e^(-decay_rate × hours_since_last_access)
```

High-importance, frequently-accessed memories persist indefinitely. Low-importance, stale memories decay and are pruned during nightly consolidation.

## Namespaces

Memory can be scoped to a namespace (default: `"personal"`). Namespaces allow project-specific facts, clean separation of domains (work, personal, codebase), and residency policies (`local_only` vs `any`).

## Consolidation

A background loop runs every 24 hours to:
- Prune low-retention memories using the forgetting curve
- Promote reinforced episodes into semantic facts (with idempotency guards)
- Apply data-residency enforcement

## Memory Trust

Every memory stores its source agent. Brain supports provenance-weighted recall: per-agent trust weights determine how much agent-written memories influence context assembly. Unattested agent writes land quarantined until reviewed.
