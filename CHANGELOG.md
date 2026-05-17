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

- **Pillar wiring — every `with_*` injection slot now populated in
  bootstrap** (Issues 5–11, 18–19, 21, 29–30; Wave A). The eight
  pillar crates shipped between v0.3.0 and v0.4.0 — `brainos-observe`,
  `brainos-identity`, `brainos-mcphost`, `brainos-terminal`,
  `brainos-intent`, `brainos-reflex`, `brainos-resilience`,
  `brainos-orchestrate` — were compiled and unit-tested but
  `cli::{serve,bootstrap}.rs` only wired `with_standing_approvals`,
  leaving every other guard like `if let Some(observer) = self.observer`
  silently inactive in production. v0.4.0 closes that gap.
  `BroadcastObserver`, `ConfigIdentityStore`, `RmcpHost` (as default
  MCP host wrapped in `ResilientMcpHost`), Terminal Bridge (behind
  `--terminal` serve flag), `InMemoryToolRegistry` +
  `DefaultIntentRouter` + `InMemoryCapabilityIndex`,
  `FsReflex`/`CronReflex`/`SysStateReflex` reflex sources, and the
  `BreakerRegistry` are all now constructed in bootstrap and threaded
  through `SignalProcessor`. The compactor runs as a 24h reflection
  task in `cli::serve`. Terminal sessions mirror into the episodic
  graph via `HippocampusTerminalSink`. `DualMemoryReader` is wired as
  a first-class field on `SignalProcessor`. A DLQ drain task purges
  the dead-letter queue periodically. New umbrella test
  `crates/cli/src/bootstrap.rs::build_processor_populates_every_injection_slot`
  is the new contract — a new `with_*` builder cannot land without a
  matching bootstrap call.

- **Canonical verb vocabulary** (Issue 158, v1.0.0 RFC §1).
  New `intent::verbs::VERBS` lists the 23 verbs the kernel addresses
  (`memory.store`, `shell.exec`, `mcp.mount`, …) with namespace, action,
  conservative `TierHint`, and human summary. New helpers
  `intent::verbs::lookup(ns, action)` and `intent::verbs::namespaces()`
  give consumers a single source of truth for what the kernel can
  authorize. Implemented as compile-time constants rather than a
  separate `verbs.toml` file: the verb surface is kernel contract
  (adding one requires a new `Intent` variant + authz mapping +
  handler), not config. Cross-check test
  `signal::authz::tests::every_static_verb_is_in_registry` asserts
  every typed-Intent → AuthorizationRequest verb resolves through
  `lookup`, AND that `tier_for_verb` matches each entry's
  `tier_hint` — typos in `intent_to_auth` therefore fail the test
  suite rather than the runtime.

### Changed

- **Loop-detector hash switched to BLAKE3** (Issue 160, v1.0.0 RFC
  §4). `resilience::loop_detector::hash_call` previously used
  `std::collections::hash_map::DefaultHasher` (SipHash with per-process
  seed). Same field separator (`\x1f`), same `(tool_id, canonical_json)`
  inputs, same `u64` storage shape — the change is keyless determinism
  across process restarts, which is the prerequisite for any future
  on-disk window persistence. Functionally equivalent within a single
  process. Adds `blake3 = "1"` to workspace dependencies.

- **Config hygiene** (Issues 32, 36–41; Wave D).
  - `default.yaml` brain.version bumped `0.2.0` → `0.4.0` (Issue 32).
  - `impl Default for BrainConfig` proactivity fields synced with
    `default.yaml`; new `default_matches_yaml` regression test
    (Issue 36).
  - `storage.hnsw.{ef_construction,m,ef_search,max_elements}` now
    threaded into `RuVectorStore::open` instead of hardcoded
    (Issue 37).
  - `cli::serve` respects `adapters.{http,ws,grpc,mcp}.enabled` flags
    as a secondary filter after CLI flags (Issue 38).
  - Dead config fields removed: `adapters.mcp.{stdio,http}` and
    empty `memory.episodic` struct (Issues 39, 41).
  - Legacy `llm.provider` marked `#[deprecated]`; startup warns when
    both it and `providers[]` are set (Issue 40).

- **Public surface shrunk** (Issues 33–34, Wave E).
  `channel::CorrelatedCommand` and `channel::RoutingDecision` are no
  longer publicly re-exported.

### Fixed

- **`handle_prune_audit` honours `older_than`** (Issue 1, Wave B.1).
  Now parses `"30d"`/`"7d"`/`"24h"` and passes the real duration to
  `audit.prune(...)`. Previously the parameter was silently ignored.

- **`handle_list_schedules` / `handle_cancel_schedule` actually
  query** (Issues 2–3, Waves B.2–B.3). Both intents now call into the
  scheduled-intent store (`list_scheduled_intents` / `cancel_scheduled_intent`)
  and format real results. Previously returned hardcoded placeholder
  strings.

- **`handle_set_proactivity` toggles state** (Issue 4, Wave B.4).
  Simplified to a runtime-mutable toggle backed by `RwLock`. Full
  windowed-mutation semantics deferred to v1.0.

- **HTTP error responses can no longer panic** (Issue 43, Wave C.1).
  All `Response::builder().unwrap()` call sites replaced with
  `.map_err(...)` returning a 500 fallback.

- **External error messages sanitized** (Issue 44, Wave C.2).
  `SignalError::to_public()` returns a sanitized `PublicError`; all
  adapter error renderers use it. Internal stack-trace-grade details
  no longer leak across the wire.

- **`ProjectInspect` path-sandboxed** (Issue 119, Wave C.6).
  `handle_project_inspect` canonicalizes the requested path and
  rejects anything outside `config.security.allowed_paths`
  (default: `$HOME`). Previously, an LLM-extracted path could read
  arbitrary files.

- **`ActionDispatcher::execute_command` routes through the sandbox**
  (Issue 121, Wave C.7). Replaced raw `tokio::process::Command` with
  the wired `SandboxExecutor`. Resource limits, allowlist, and
  platform isolation now apply to dispatcher invocations.

- **Adapter `enabled` flags are a real gate** (Issue 38, Wave D.4;
  also listed under Changed). Setting `adapters.http.enabled: false`
  now actually prevents the HTTP adapter from starting, instead of
  being silently ignored.

### Security

- **Per-client rate limiting on HTTP/WS/gRPC** (Issue 51, Wave C.3).
  Resilience `RateLimiter` wired into the Tower stack on HTTP and WS
  adapters; gRPC interceptor enforces the same.

- **Webhook endpoint requires Bearer auth when verifier-less**
  (Issue 52, Wave C.4). `POST /v1/webhooks/:id` now demands a Bearer
  token whenever the configured transport has `verifier == None`.

- **Fail-closed startup on empty `api_keys`** (Issues 53–54, Wave C.5).
  `brain serve` exits non-zero on launch if `adapters.api_keys` is
  empty, unless `--no-auth` is passed explicitly. Shell-mode allowlist
  bypass is now documented in OPERATIONS.md.

- **OAuth `aud` claim validation at MCP mount time** (Issue 163,
  CVE-2025-6514 / confused-deputy via token passthrough). New
  `mcphost::aud_check::validate_token_aud` decodes the persisted
  access token (JWT format only — opaque tokens skip validation) and
  rejects a mount when the token's `aud` claim does not include the
  configured RFC 8707 `resource` indicator. `mcphost::manager_from_vault`
  signature gained `expected_resource` + an `Option<Arc<dyn Observer>>`;
  `RmcpHost::mount_http` defaults the resource to the server URL
  (canonical mapping for vanilla MCP deployments) and threads its
  observer through. Mismatches surface as
  `BrainEvent::Error { source: "mcphost.oauth" }` on the live event
  bus AND fail the mount. Signature verification is intentionally
  out of scope — vault + TLS to AS + PKCE are the trust boundary,
  not in-band JWS verification. +9 unit tests covering string-aud
  match, array-aud match, mismatch, missing-aud, opaque-token, and
  malformed-JWT shapes.

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

### Removed

- **Dead code cleanup** (Issues 13–14, 16–17, 20, 23, 27–28; Wave E).
  - `FORGET_RE` and `STORE_FACT_RE` regexes removed (shadowed by
    `classify_explicit`).
  - `set_action_namespace()` and `recall_memories()` removed
    (no callers).
  - `AckSignalHandler` moved into `#[cfg(test)]`.
  - `AgentTask.credentials`, `Markdown::push_blank`/`Markdown::kv`,
    and `extract::is_supported` removed.

- **Dead config fields removed** (Issues 18–19, 21; Wave A side-effects).
  `credential_vault` field dropped, `sandbox_executor` accessor
  visibility downgraded, `RelayConfig.api_key` removed.

### Deferred

- **Tower middleware exact ordering** (Issue 154, RFC §4.2).
  `ResilientMcpHost` remains a hand-rolled decorator. Swap to
  `tower::ServiceBuilder` composition deferred to v1.0.0 — security-
  sensitive layer; must preserve semantics through the transition.

- **Modifiers-based capability scoping** (Issue 159, RFC §12 OQ#2).
  Identity has path-scope (PR1) but not the full
  `AuthorizationRequest.modifiers` per-principal allowlist. Wire
  contract change deferred to v1.0.0.

- **DualMemoryReader → RecallEngine composition.** The reader is
  wired as a `SignalProcessor` field, but does NOT compose into
  `RecallEngine::recall`. `DualMemoryReader` only exposes
  `read_by_id`; `RecallEngine` is hybrid BM25+ANN. They don't compose
  without graph-side FTS5 + ANN, which is out of v0.4.0 scope.
  Reopens as a v0.5.0 slice when graph search lands.

- **Structural follow-ups** clustered for v0.5.0+: performance
  (embedding cache, namespace scan, blocking std::fs in async, N+1
  deletes, SqlitePool mutex serialization, HNSW preallocation, batch
  embed clones); test coverage (vault, adapters, resilience, identity,
  intent, plus benchmarks and fuzzing); async hygiene (blocking I/O
  on async runtime, fire-and-forget spawns, std::sync::mpsc in async);
  SOLID refactors (`SignalProcessor` field reduction, `pipeline.rs` /
  `serve.rs` / `main.rs` splits, large match-arm extraction); duplicate
  consolidation (two `CircuitBreaker` impls, LLM message conversion
  and HTTP error handling repeated across providers, embedding HTTP
  client builder, `identity::Tier` vs `ActionTier`); workspace lints
  + `publish = false` guards; CI matrix (macOS, Windows, MSRV) and
  release automation; cross-platform (Linux protoc, Windows syscall
  guards); medium-severity security hardening (gRPC message limits,
  CORS exact-match, webhook replay protection, SSRF, vault passphrase
  exposure, MCP arbitrary-mount risk); and `brain doctor --deep`
  system inspection.

### Security

- **OAuth `aud` claim validation at MCP mount time** (Issue 163,
  CVE-2025-6514 / confused-deputy via token passthrough). New
  `mcphost::aud_check::validate_token_aud` decodes the persisted
  access token (JWT format only — opaque tokens skip validation) and
  rejects a mount when the token's `aud` claim does not include the
  configured RFC 8707 `resource` indicator. `mcphost::manager_from_vault`
  signature gained `expected_resource` + an `Option<Arc<dyn Observer>>`;
  `RmcpHost::mount_http` defaults the resource to the server URL
  (canonical mapping for vanilla MCP deployments) and threads its
  observer through. Mismatches surface as
  `BrainEvent::Error { source: "mcphost.oauth" }` on the live event
  bus AND fail the mount. Signature verification is intentionally
  out of scope — vault + TLS to AS + PKCE are the trust boundary,
  not in-band JWS verification. +9 unit tests covering string-aud
  match, array-aud match, mismatch, missing-aud, opaque-token, and
  malformed-JWT shapes.
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
