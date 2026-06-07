# Changelog

All notable changes to Brain OS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Per-principal capability scoping (modifier constraints).** An identity
  principal can now constrain individual action *modifiers*, not just the
  filesystem path. A new `constraints:` list under each principal scopes any
  `(verb, modifier)` pair — e.g. allow `net.http` only to `*.github.com`,
  `shell.exec` only to commands starting with `git `, or `mcp.mount` only to a
  named server — with `exact`, `prefix`, or `host_suffix` matching. Checks are
  fail-closed: a constrained verb that arrives without its modifier, or with a
  value outside the allow-list, is denied at signal-entry (not at execution).
  Constraints are opt-in; a principal without them behaves exactly as before.
  This generalises the existing `path_allowlist` (kept as the built-in path
  case) and is the enforcement substrate for future capability-scoped skill
  packs.
- **Model context-window auto-detection.** On startup Brain probes the active
  LLM provider for its real context window and scales the prompt budgets to
  match, instead of clipping everything to the conservative default. Detection
  uses the provider API where available (OpenRouter advertises `context_length`
  per model; Ollama exposes it via `/api/show`) and falls back to model-name
  heuristics for providers that don't (OpenAI, Groq, DeepSeek, and others). A
  detected window only widens the budget — a value you set in
  `llm.context_window` is never lowered, and a configured value larger than the
  detected capacity is kept with a warning. The extra room flows into richer
  grounding: path-attachment snapshots, directory listings, and the number of
  recalled memories all scale with the available budget, so a 128k- or
  1M-window model reads far more file content and surfaces more relevant
  context than the 8k default.
- **Learned capability fitness.** Brain now learns which tools actually work,
  not just which exist. After every tool dispatch it records a per-tool
  success/failure outcome (`learning.capability_fitness`, on by default), and
  those observations decay under the forgetting curve (`half_life_days`,
  default 30). The learned record nudges tool selection — a proven tool wins
  a tie among equally-relevant candidates when the chat model is offered
  tools, but never overtakes a stronger keyword match — and the reasoner's
  capability digest gains a "Proven here" line listing tools with a track
  record. Awareness/preference only: execution stays consent-gated. Backed by
  a new `capability_fitness` table (procedural memory). Set
  `learning.capability_fitness.enabled: false` to opt out (ranking and digest
  then behave exactly as before).
- **Release automation.** Tagging a `vX.Y.Z` release now builds pre-built
  binaries — `brain-<target>.tar.gz` for macOS (`aarch64`/`x86_64`) and
  Linux (`aarch64`/`x86_64`), each with a `.sha256` — and publishes a
  GitHub Release with an SPDX SBOM and the changelog section as notes
  (`.github/workflows/release.yml`). This is what `scripts/install.sh`
  downloads, so the one-line installer now resolves to a real binary on
  every supported platform. Maintainers drive releases with
  `scripts/release.sh X.Y.Z`, which validates the tree/version/changelog,
  runs the full CI-parity gate, `cargo publish`es every crate in dependency
  order, then tags and pushes; `scripts/changelog-extract.sh` prints a
  single version's notes.
- **Structured logging policy.** New `[logging]` config section: base
  `level`, per-subsystem `targets` overrides, `format` (`pretty`|`json`),
  and daemon log-file `rotation` (`daily`|`hourly`|`never`). Long-running
  services (`serve`, `mcp`) now log through a rotating file appender at
  `~/.brain/logs/brain.log` instead of an unbounded shell-redirected file;
  one-shot commands keep stderr. `RUST_LOG` still overrides the computed
  filter.
- **`brain doctor --deep`.** Store-level health probes beyond the
  environment check: SQLite open + schema version, audit-log `prev_hash`
  linkage continuity, episode/fact/graph-node counts, RuVector store
  (per-collection counts + dimension), and an embedder round-trip
  asserting the output dimension matches config. The vector-store probe is
  skipped when a daemon holds the lock.
- **MCP resources & prompts.** The MCP server now implements
  `resources/list` + `resources/read` (`brain://profile`,
  `brain://capabilities`, `brain://namespaces`) and `prompts/list` +
  `prompts/get` (`recall-context`, `daily-review`) instead of stubbing them
  to empty arrays.
- **Graph memory now influences recall.** The episodic graph was
  write-only with respect to retrieval; it now contributes two candidate
  lists to hybrid recall. A full-text index over node bodies (`nodes_fts`,
  migration v23 with sync triggers) backs `EpisodicGraph::search_text`
  (BM25), and the terminal graph sink embeds node bodies at write time
  into a new `graph_vec` vector collection, setting each node's
  `vector_id` for ANN. `DualMemoryReader::recall_candidates` produces both
  lists; `RecallEngine::recall` folds them into RRF fusion alongside
  episodic + semantic results and emits a new `MemorySource::Graph`.
  Terminal/tool activity is now recallable by text and by semantic
  similarity. Embedding on the write path is best-effort — a failure
  leaves the node un-vectored rather than dropping it.
- **Pre-commit gate.** `scripts/install-hooks.sh` installs a repo-managed
  pre-commit hook that runs `cargo fmt --check`, `cargo clippy
  --all-targets -D warnings`, and the crate-naming check. Opt-in;
  bypassable with `git commit --no-verify`.
- **CONTRIBUTING.md, PR template, and bug/feature issue templates.**
  Codifies the commit-message style, MSRV expectations, conventions, and
  required local CI parity steps.
- **MSRV policy.** Workspace declares `rust-version = "1.91"` (current
  stable is 1.93; this matches the stated N-2 floor). The hard floor is
  1.89 — ruvector-core uses `stdarch_x86_avx512`, stabilized in 1.89.
  New `msrv` CI job pins to the declared toolchain and runs `cargo check
  --workspace --locked` on every push/PR.
- **CHANGELOG validation.** `scripts/check-changelog.sh` + `changelog`
  CI job verifies the workspace version has a matching section and
  fails the build when source/manifest changes lack a CHANGELOG entry.
- **`cargo-deny` license + advisory enforcement.** `deny.toml` allow-lists
  permissive licenses, blocks unknown registries/git sources, and the
  `deny` CI job runs `cargo deny check all` on every PR.
- **Third-party attribution scaffolding.** `about.toml` + `about.hbs` +
  `scripts/generate-attribution.sh` produce `THIRD_PARTY_LICENSES.md`
  via `cargo-about`. License set mirrors `deny.toml`.
- **Cross-platform CI matrix.** `check` and `test` now run on Ubuntu and
  macOS. Windows is omitted: both the grpc and terminal adapters depend
  on `protobuf-src` (build dep), which compiles Abseil + protoc from C++
  source and fails to link against MSVC's ucrt (`__imp_nan`,
  `__imp_modf`, …). Since terminal is a non-optional dep of the cli and
  most pillar crates pull it transitively, there's no useful subset of
  the workspace that compiles on Windows without first gating protobuf
  build deps on `cfg(not(windows))`. Tracked as v0.6.0 work.
- **Migration regression tests.** `test_migrations_record_all_expected_versions`
  asserts every declared migration appears in the `_migrations` table
  after `migrate()`; `test_schema_snapshot_matches` compares
  post-migration `sqlite_master` against a committed snapshot to catch
  silent schema drift in existing migrations.
- **Module-boundary enforcement.** `scripts/check-boundaries.sh` plus
  `boundaries` CI job verify no workspace crate depends on the `brainos`
  CLI binary and transport adapters don't depend on each other.
- **Multi-stage Dockerfile + `.dockerignore`.** Distroless `nonroot`
  runtime image; default feature set enabled; documented buildx
  command for multi-arch (linux/amd64 + linux/arm64) images.
- **`scripts/install.sh`** — Linux/macOS one-line installer. Probes the
  GitHub Releases binary for the host triple; falls back to
  `cargo install brainos` or `cargo install --git` from source.
- **Live capability digest in the SOUL prompt.** The chat reasoner's
  "Your Capabilities" section is now rendered from the *live* wired
  capability set — the tool registry (MCP servers, native backends,
  terminal) and the delegate agent registry — instead of a hardcoded
  three-line string. Mount a new MCP server and the next chat turn's
  self-description reflects it. Read-only awareness only: execution stays
  gated by the consent/audit/breaker path, and untrusted MCP tool
  descriptions are not inlined. New `cortex::context::DEFAULT_CAPABILITIES`
  constant + `capabilities` parameter on
  `ContextAssembler::assemble_full`;
  `signal::pipeline::conversation::capability_digest`.
- **Unified capability manifest.** The kernel's *native*
  capabilities (action-dispatcher backends — memory, web, scheduling,
  messaging — plus the terminal bridge) now register into the same
  `intent::ToolRegistry` the MCP host populates, so one manifest describes
  every tool the kernel can dispatch to — not just mounted MCP servers.
  `intent::ToolDescriptor` gains a `usage: ToolUsage` field carrying
  reasoner-facing guidance (`when_to_use` / `when_not_to` / `preconditions`
  / `cost` / `example` / `tier`); native descriptors are stamped with the
  canonical verb summary + tier from `intent::verbs`. New
  `brain capabilities` subcommand and `Intent::ListCapabilities`
  (`/capabilities` in chat) render the live manifest grouped by source and
  tagged with tier; a new read-only `brain_capabilities` MCP tool exposes
  the same manifest to external clients. Awareness only — execution stays
  gated. Untrusted MCP descriptions are sanitized before display.
- **Planner sees the live capability roster.** Task decomposition now
  composes against what the kernel can actually do: `DecompositionContext`
  gains `available_agents` (the real `delegate::AgentRegistry` roster) and
  `available_capabilities` (live manifest summary lines — mounted MCP
  servers with their verbs, native backends, the terminal), both surfaced
  in the decompose and replan prompts. An `implement` step that names an
  agent which isn't registered is now rejected at *plan* time with the
  list of available agents, instead of failing once execution reaches the
  step. The sandbox binary allowlist (`available_tools`) is unchanged and
  still gates `execute`/`test` argv steps. New
  `signal::pipeline::conversation::planner_capabilities`.
- **Tool-use loop in chat.** A chat turn can now use the kernel's tools, not
  just describe them. `cortex::LlmProvider` gains a tools channel
  (`generate_with_tools` with `ToolDef` / `ProposedToolCall`; defaults to
  plain text so providers without one degrade gracefully) implemented for
  OpenAI and Ollama. The pipeline advertises a relevance-ranked slice of the
  unified capability manifest to the model, and when the model proposes a
  call it is mapped to an `intent::IntentToken`, passed through the **same**
  tier-based confirmation gate every other capability invocation uses,
  executed on approval, and its result fed back so the model can ground its
  answer — looping over a bounded number of rounds within one turn.
  `cortex::Message` gains `Role::Tool` and tool-call linkage; untrusted MCP
  descriptions are sanitized before reaching a provider. Awareness stays
  distinct from permission — advertising a tool only lets the model propose
  it. New `signal::pipeline::toolloop`.

### Changed

- **Routine dependency bumps.** `clap` 4.5 → 4.6, `tempfile` 3.25 → 3.27,
  `tracing-subscriber` 0.3.22 → 0.3.23, `ruvector-core` 2.0.5 → 2.2.0, and the
  `criterion` dev-dependency 0.5 → 0.8 (benchmarks recompile cleanly under the
  new API). All verified against the Rust 1.91 MSRV. Batches Dependabot #9–#13.
- **Scheduled intents fire through the pipeline.** The direct-execution
  scheduled-intent poller in `serve` (a fixed 60s ticker that delivered a
  bare `[scheduled] …` notification and bypassed identity, confirmation,
  and per-tool breakers) has been retired. Firing now flows exclusively
  through `reflex::CronReflex`: each due intent becomes a
  `Provenance::Reflex` signal and runs the full pipeline (classification,
  confirmation gate, breakers, audit). `actions.scheduling` remains the
  *write* axis (create/persist intents); `reflex.cron.enabled` is now the
  sole *fire* axis. `brain doctor` and `serve` startup warn when
  scheduling is enabled while the cron reflex is disabled, since intents
  would then persist but never fire.
- **`scripts/check-crate-names.sh`** — header comment and error footer
  no longer reference gitignored docs paths. Rule is inlined; rationale
  moved to `CONTRIBUTING.md`.
- **Dockerfile** — `RUST_VERSION` now tracks the workspace MSRV (was pinned
  at 1.85, below the declared 1.91, so the image no longer built). Corrected
  the inaccurate "musl" header comment (the build is glibc/distroless). A
  new `docker` CI job builds the linux/amd64 image on every PR so it can't
  rot again (no registry push).

### Removed

- **`.cargo/config.toml`** — deleted. The hardcoded `PROTOC =
  "/opt/homebrew/bin/protoc"` was stale: the grpc adapter's `build.rs`
  already vendors protoc via `protobuf-src`, and the hardcoded path
  broke Linux builds.

### Security

- **Patched dependencies flagged by RUSTSEC advisories** (clears the
  `cargo-deny` advisory gate and the corresponding Dependabot alerts):
  `rkyv` 0.8.15 → 0.8.16 (panic-safety bug in `InlineVec`/`SerVec::clear`
  that could enable arbitrary code execution; transitive via
  `ruvector-core`), `rpassword` 7.4.0 → 7.5.4 (partial password reveal on
  interrupted input), `rand` 0.8.5 → 0.8.6 and 0.9.2 → 0.9.4
  (custom-logger unsoundness), and `lru` 0.12 → 0.16 (`IterMut`
  Stacked-Borrows violation). The now-resolved advisory ignores were
  dropped from `deny.toml`.

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

#### Post-closeout soak fixes

- **WebSocket chat confirmation reach.** Each authenticated WS
  connection now registers a `WsChatTransport` with the
  `ChannelDispatcher` (id `ws:<conn-uuid>`) so confirmation prompts
  and any other outbound `DeliveryIntent` reach the user on the same
  socket they're chatting on. Inbound is a no-op — chat content
  already flows through `process_text_frame` into the signal pipeline.
  The connection handler calls `unregister_transport` on disconnect
  so the router stops routing to dead handles.
  `ChannelDispatcher::unregister_transport` and
  `ChannelRouter::unregister` are new (idempotent) and used only by
  short-lived transports. The streaming response path was also
  reshaped around a per-connection `mpsc<Message>` fan-in so the
  reader, writer, and BrainEvent broadcast subscriber don't fight
  over the underlying sink.
- **Observability gap closed: `IntentClassified`,
  `ConfirmationRequested`, `ConfirmationResolved` now actually
  emit** from the signal pipeline. Previously defined on the bus but
  never published; `brain tail` and the WS event stream now surface
  them in real time. Resolution decisions are tagged
  `approved`/`rejected`/`timed_out`/`aborted`/`error`.
- **Late `RespondToApproval` swallowed silently.** If the user's
  chat client buffered an `approve <nonce>` keystroke that lands
  after the engine has already timed out / approved / rejected the
  nonce (`AlreadyResolved` / `NotFound`), the pipeline now returns
  an empty body so the renderer skips it. Previously it surfaced
  "Approval already resolved" as a confusing `Brain:` error.
- **Terminal replay buffer closes late-attach race** in `TerminalBridge`
  (commit 9868dd2).
- **`SignalReceived` published on streaming WS chat + `brain tail`
  prints a connect line** (commit 237dac1).
- **`brain tail` reqwest no longer hangs** on zero-duration timeout
  (commit b5aaf0e).
- **Audit observer wired in bootstrap** so `SqliteAuditTrail` events
  reach the bus (commit 9708294).
- **Soak regressions fixed**: `brain tail` auth handshake, config
  fallback path, terminal close error text (commit d8e0612).

### Security

- **Per-client rate limiting on HTTP/WS/gRPC** (Issue 51, Wave C.3).
  Resilience `RateLimiter` wired into the Tower stack on HTTP and WS
  adapters; gRPC interceptor enforces the same.

- **Webhook endpoint requires Bearer auth when verifier-less**
  (Issue 52, Wave C.4). `POST /v1/webhooks/:id` now demands a Bearer
  token whenever the configured transport has `verifier == None`.

- **Fail-closed on empty `api_keys`** (Issues 53–54, Wave C.5 + soak
  hardening). Config validation returns `Err` when `access.api_keys` is
  empty — `brain serve`, `brain start`, and every other command that
  loads config refuses to boot. The `AuthResult::Open` enum variant
  that previously let empty-keys silently fall through to "allow all"
  has been removed; `check_auth` now returns `MissingKey` so adapters
  fail closed even on direct calls. The original `--no-auth` opt-in
  flag from the first Wave C.5 patch was also removed — running
  anonymously is no longer possible. Shell-mode allowlist bypass
  remains documented in OPERATIONS.md.

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
