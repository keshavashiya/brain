# Contributing to Brain OS

Thanks for your interest in contributing. Brain OS is a local-first AI memory
and orchestration system written in Rust. This guide covers the workflow for
patches, the conventions enforced by CI, and how to run the full validation
suite locally before opening a PR.

## Quick start

```bash
git clone https://github.com/keshavashiya/brain.git
cd brain
./scripts/install-hooks.sh          # opt-in pre-commit gate
cargo check --workspace
cargo test --workspace
```

The repo uses a single-binary architecture (`brain`) with 31 workspace crates
under `crates/`. See `README.md` for an architecture overview and the
workspace `Cargo.toml` for the full crate list.

## Workflow

1. **Open or claim an issue.** Non-trivial work should start with an issue so
   scope, interface, and approach are agreed before code is written.
2. **Branch from `main`.** Name branches descriptively, e.g.
   `wave-a2-changelog-validation`, `fix-channel-router-panic`.
3. **Make focused commits.** One logical change per commit, with messages that
   explain *why* the change is needed.
4. **Update `CHANGELOG.md`.** Any user-visible change (new feature, bugfix,
   behaviour change, removed/renamed API) needs an entry under `[Unreleased]`.
5. **Run the full local CI parity check** before pushing (see below).
6. **Open a PR** against `main`. Fill in the PR template; link the issue.

## Commit messages

We follow Conventional Commits loosely. Common prefixes:

- `feat(scope):` — new functionality
- `fix(scope):` — bug fix
- `refactor(scope):` — code restructure with no behaviour change
- `docs(scope):` — documentation only
- `chore(scope):` — tooling, deps, workspace, CI
- `test(scope):` — adding or fixing tests

The scope is usually a crate name (`cli`, `signal`, `cortex`, ...) or a wave
identifier (`Wave A.2`).

## Local CI parity

CI runs the gates below on Linux + macOS. Before pushing, run the
equivalents locally — much faster than discovering a break in GHA:

```bash
# Format, lints, build
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace
cargo check -p brainos --no-default-features
cargo check -p brainos --no-default-features --features encryption

# Project gates
./scripts/check-crate-names.sh
./scripts/check-boundaries.sh
./scripts/check-changelog.sh

# Tests
cargo test --workspace

# MSRV (install once: `rustup toolchain install 1.91`)
cargo +1.91 check --workspace --locked

# License + advisory (install once: `cargo install cargo-deny --locked`)
cargo deny check all
```

The pre-commit hook (`./scripts/install-hooks.sh`) runs fmt + clippy +
crate-names on every commit so you don't ship a CI-breaking diff by accident.
The MSRV and cargo-deny gates are slower (multi-minute) and intentionally
not in the hook — run them manually before pushing a substantive change.

## Conventions

- **Crate naming.** Folder and workspace-alias names are single lowercase
  words — no underscores, no hyphens. The only exception is the grouping
  folder `crates/adapters/` whose children are transport-adapter crates.
  Enforced by `scripts/check-crate-names.sh` in CI.
- **Workspace lints.** Clippy is `deny`-by-default at the workspace root.
  Specific allows live in `Cargo.toml` under `[workspace.lints.clippy]` with
  a comment justifying each one.
- **No native channel transports.** Channel adapters are generic
  (`http_polled`, `webhook_inbound`, `webhook_outbound`) configured via YAML
  presets. Do not add Telegram/Slack/Discord-specific Rust code.

## Testing & fuzzing

Three layers, all run by `cargo test --workspace`:

- **Unit / integration tests** — the default; example-based, colocated in `src`
  or in each crate's `tests/`.
- **Property tests** (`proptest`) — pin bounded/ordering/round-trip invariants
  of pure-logic seams across the kernel.
- **Fuzz targets** (`bolero`) — coverage-guided fuzzing of the untrusted-input
  parsers (the secret masker, the approval-correlation parser, the regex intent
  classifier, the markdown render pipeline, grounding-text truncation). Each is
  a `fuzz_*` test that asserts no-panic plus a couple of semantic invariants.

Fuzz targets are written with `bolero::check!()`, so they run in bounded
property mode under plain `cargo test` on **stable** — no nightly needed — and
double as regression tests. To run a target coverage-guided (libfuzzer), use
the [`cargo-bolero`](https://github.com/camshaft/bolero) driver on nightly:

```bash
cargo install cargo-bolero
cargo bolero list                                              # discover targets
cargo bolero test -p brainos-signal fuzz_mask_secrets_invariants   # fuzz one
```

Crash inputs are written under the owning crate's `tests/` corpus and replay
automatically on the next `cargo test`. When you touch a parser that consumes
untrusted bytes or strings, add or extend a `fuzz_*` target alongside it.

## Reporting issues

- **Bugs:** use the bug report template. Include `brain --version`, OS, and
  a minimal reproduction.
- **Features:** use the feature request template. State the user problem
  before the proposed solution.
- **Security:** do not open a public issue. Email the maintainer
  (see `Cargo.toml` `authors`).

## Third-party attribution

`THIRD_PARTY_LICENSES.md` ships in every release and quotes the upstream
license text for every bundled dependency. Regenerate before tagging:

```bash
cargo install cargo-about --locked
./scripts/generate-attribution.sh
```

The accepted-license set is mirrored in `deny.toml` and `about.toml`. When
adding a dependency that introduces a new license, update both files in the
same PR and regenerate the attribution.

## Dependency hygiene

Dependencies are kept current by Dependabot (weekly) and guarded by `cargo
audit` + `cargo deny check all` in CI. Declare new dependencies once in root
`[workspace.dependencies]` and reference them with `{ workspace = true }`; the
license must already be on the `deny.toml` allow-list. To audit staleness
locally:

```bash
cargo install cargo-outdated --locked
./scripts/check-freshness.sh           # advisory; --strict to fail
```

## License

By contributing, you agree your contributions will be licensed under the
project's MIT license (see `LICENSE`).
