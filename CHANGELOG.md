# Changelog

All notable changes to Brain OS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — unreleased

"Wire the Pillars + Fix the Stubs" release. v0.3.0 → v0.4.0 promotes
every pillar crate shipped between v0.3.0 and this release from
"compiles and tests in isolation" to "wired into production startup."
The umbrella test `build_processor_populates_every_injection_slot` in
`crates/cli/src/bootstrap.rs` is the new contract — a new `with_*`
builder cannot land without a matching bootstrap call.

Workspace-locked: all 31 crates publish at 0.4.0 together.

### Added

_(filled per-fix as security/RFC gap closures land — see PRs in
`D-Reconcile.2` wave for the historical record of the 91 commits
between v0.3.0 and tagging.)_

### Changed

- **Loop-detector hash switched to BLAKE3** (Issue 160, v1.0.0 RFC
  §4). `resilience::loop_detector::hash_call` previously used
  `std::collections::hash_map::DefaultHasher` (SipHash with per-process
  seed). Same field separator (`\x1f`), same `(tool_id, canonical_json)`
  inputs, same `u64` storage shape — the change is keyless determinism
  across process restarts, which is the prerequisite for any future
  on-disk window persistence. Functionally equivalent within a single
  process. Adds `blake3 = "1"` to workspace dependencies.

### Fixed

_(filled per-fix)_

### Security

- **MCP tool descriptions treated as untrusted at the LLM boundary**
  (Issue 162). New `intent::sanitization::render_tool_description_for_prompt`
  fences every attacker-controllable description inside a labelled
  `~~~`-delimited block, strips C0 control bytes + ANSI CSI escapes,
  defangs any literal fence sentinel inside the body, and caps length
  at 2 KiB on a UTF-8 char boundary. `intent::ToolDescriptor.description`
  and `mcphost::ToolDescriptor.description` carry doc-comments marking
  them untrusted; callers wiring descriptions into prompts must use
  the sanitizer. Complements the hash-pin rug-pull detector in
  `mcphost::RmcpHost` (which catches *changes* to descriptions);
  the sanitizer is what stops a single hostile first-mount description
  from landing as live system instructions.

### Deferred

_(carry-overs to v0.5.0 / v1.0.0 are listed here as each is decided)_

## [0.3.0] — 2026-05-14

Natural-language interface release. Focus: collapse the CLI surface
down to bootstrap + security-sensitive stdin, enforce the v0.2 budget
promise end-to-end, and reconcile docs to code so every claim is
verifiable. Workspace-locked version: all 24 crates publish at 0.3.0
together.

### Added

- **Pre-flight LLM budget enforcement** — the main Chat/Recall path
  now calls `CostBudget::check` before invoking the LLM. A configured
  ceiling blocks the call *before* tokens burn and returns a friendly
  message naming the provider and the limit hit. Post-call recording
  prefers real `Usage` from the provider and falls back to the
  pre-flight `chars/2` estimate. New module: `signal::budget_guard`.

### Changed

- **Natural-language is the interface** — every operation that can be
  an intent is one. Thalamus now classifies and `SignalProcessor`
  handles `QueryAudit`, `PruneAudit`, `ListApprovals`,
  `RespondToApproval`, `BudgetStatus`, `ListSchedules`,
  `CancelSchedule`, `ListTasks`, `TaskStatus`, `CancelTask`,
  `ListChannels`, `ChannelPreferences`, `SetChannelPreference`,
  `MemorySummary`, and `ProjectInspect`. Legacy subcommands remain
  as thin deprecation shims; the canonical entry point is
  `brain chat "…"`.
- **Config can no longer lie** — removed three documented "ignored"
  keys: `memory.episodic.max_entries`, `memory.episodic.retention_days`,
  and `memory.search.hybrid_weight`. Forgetting-curve consolidation
  (`decay_rate`) replaces the first two; RRF (`rrf_k`) replaces the
  third. Existing yamls with these keys continue to load — figment
  ignores them — but the warnings are gone and `default.yaml` no
  longer advertises them.

### Fixed

- **Docs reconciled to code** — ARCHITECTURE / IMPLEMENTATION /
  OPERATIONS doc text that still claimed the `POST /v1/webhooks/:id`
  route was "still pending" has been updated. The route is wired in
  `crates/adapters/http/src/server.rs` and dispatches into the
  registered `WebhookInboundTransport` for HMAC/Ed25519 verification.

### Deferred

- **`ExportMemory` / `ImportMemory` intents** — filesystem-path intents
  need capability scoping (v1.0.0 Pillar 7) before they can be safe
  from LLM path-extraction errors. The `brain export` / `brain import`
  CLI subcommands remain the canonical surface.
- **Orchestrator decomposer LLM budget** — single-call per
  `DecomposeTask` request; wired with the v1.0.0 §Pillar 8
  orchestration state machine instead of bolted on now.

## [0.2.0] — 2026-04-11

First minor release after 0.1.0. Focus: security hardening, resilience,
streaming, and workspace cleanup. Workspace-locked version: all 16 crates
publish at 0.2.0 together.

### Added

- **Token streaming over WebSocket** with cancellation-safe finalization —
  live token-by-token chat output; upstream disconnects no longer drop
  in-flight responses.
- **Resilience primitives** (`brainos-backends::resilience`) — circuit
  breaker with half-open probes and exponential-backoff retry for all
  external HTTP calls (search, messaging, LLM). 5xx / 408 / 429 /
  timeout / connect errors are retried; other 4xx fail fast.
- **Metrics endpoint** (`/metrics`) exposes Prometheus counters and
  subsystem gauges (fact/episode counts, intent LLM fallback count,
  circuit open/reset counters).
- **Graceful shutdown** — `brain serve` handles SIGTERM and Ctrl+C,
  runs a WAL checkpoint via `SignalProcessor::shutdown()` before exit.
- **CLI chat UX** — interactive chat now prints a green `Brain:` prefix
  before the first streamed token, matching the non-streaming path.
- **`brainos-backends` crate** — action backends (search, scheduling,
  messaging) and resilience primitives extracted from `brainos` binary
  into a reusable library crate. Workspace now has 16 crates.

### Changed

- **SignalProcessor** — unified adapter entry point (`AdapterRequest`
  struct replaces a 10-parameter positional API).
- **SQLite foreign keys enforced** (`PRAGMA foreign_keys = ON`) so
  future `REFERENCES` clauses in migrations are no longer decorative.
- **Migration version bumped to 17** — adds `notifications` outbox
  table and audit-log schema; idempotent upgrade from v16.
- **CORS hardened** — exact-match origin check (`http://localhost:` /
  `http://127.0.0.1:`) replaces prefix check that could match
  `localhost.attacker.com`.
- **Intent classification prompt injection hardened** — user input is
  now a separate role-based message, not interpolated into the system
  prompt. Same change applied to importance scoring.
- **Novelty detection bounded** — amygdala `seen_topics` replaced with
  a capacity-10 000 LRU cache (was an unbounded HashSet).
- **HTTP body + frame limits** — all three HTTP-based adapters
  (`http`, `mcp` HTTP, `ws`) cap request/frame size at 1 MiB and
  concurrent in-flight requests at 100.
- **N+1 query fix** in semantic search — batched `WHERE id IN (?…)`
  lookups replace per-result round trips.

### Fixed

- **Bridge CLI handshake** — `brain bridge` now performs a two-frame
  auth-then-signal handshake. Previously merged fields into the auth
  frame were silently dropped, causing every bridge invocation to
  submit zero content.
- **Atomic FTS writes** — episode insert + FTS insert, episode delete
  + FTS delete, and semantic fact insert + RuVector write are now
  wrapped in transactions with compensating rollback.
- **`num_predict` honored** — removed incorrect `#[serde(rename_all =
  "camelCase")]` on `OllamaOptions` that silently renamed the field
  to `numPredict` and caused every Ollama call to ignore `max_tokens`.
- **Ollama streaming termination** — streams now return on `done:
  true` instead of waiting for the server to close the socket.
- **Ebbinghaus retention** uses the actual stored importance for BM25
  hits (was hardcoded to 0.5, flattening forgetting-curve reranking).
- **Notification prune** age-checks delivered notifications instead of
  deleting them immediately.
- **Consolidation prune-then-promote** now `continue`s after pruning
  so pruned episodes don't reappear as promotion candidates.
- **PID file permissions** — `0o600` on Unix; no more world-readable
  PID files.
- **`execute_command` path restriction** — absolute-path arguments
  are canonicalized (with parent-dir fallback for non-existent
  targets) and must resolve under `$HOME` or `/tmp`. `..` components
  rejected in any argument.
- **LIKE wildcard escape** — fact search escapes `%` and `_` in user
  queries with an explicit `ESCAPE '\\'` clause.
- **Fact deduplication race** — dedup-check + insert now serialized
  under a tokio async mutex.
- **HTTP gauge cache poison tolerance** — `lock().unwrap_or_else(|e|
  e.into_inner())` instead of panicking on mutex poison.

### Security

- **Constant-time API-key comparison** retained (`subtle::ConstantTimeEq`).
- **Default config ships no pre-shared API key** — `brain init` mints
  a random per-installation key on first run. Legacy `demokey123`
  references remain only in test fixtures.
- **Parameterized SQL everywhere** — no string-interpolated queries.
- **Supply chain** — all external HTTP calls go through the
  circuit-breaker / retry layer with timeout budgets.

### Documentation

- `ARCHITECTURE.md` points action backends at `crates/backends/src/`
  (they moved during the refactor) and documents the 408/429 retry
  additions to the resilience layer.
- `README.md` documents proactivity as on-by-default with guardrails
  (max 2/day, 14-hour quiet window) instead of labeling it opt-in.

### Deferred to 0.2.x

- **`signal/src/lib.rs` split** — the 1 786-line file should be
  broken into `pipeline.rs` / `recall.rs` / `streaming.rs` for
  maintainability. Structural refactor, not a release blocker.
- **Thalamus regex `LazyLock`** — hardcoded patterns still use
  `unwrap_or_else(panic!)`. Style-only; patterns are test-verified.
- **Auth length-mismatch timing leak** — `ct_eq` is skipped when
  key lengths differ. Negligible entropy loss; accepted as a known
  limitation until keys are compared as fixed-length hashes.

## [0.1.0] — Initial release

Baseline release. Local-first AI memory engine with episodic +
semantic memory, five protocol adapters (HTTP, WebSocket, gRPC,
MCP stdio, MCP HTTP), biologically-inspired pipeline (Thalamus /
Amygdala / Hippocampus / Cortex / Cerebellum / Ganglia), and
SQLite + HNSW storage.
