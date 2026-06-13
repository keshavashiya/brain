# Codebase Conventions

## Crate naming

| Layer | Rule | Example |
|-------|------|---------|
| Folder (`crates/<name>/`) | Single word, lowercase | `crates/mcphost/` |
| Package name | `brainos-<word>` | `brainos-mcphost` |
| Workspace alias | Single word, matches folder | `mcphost = { workspace = true }` |
| Rust import | `brainos_<word>` | `use brainos_mcphost::RmcpHost;` |

## Workspace deps

- All internal crates declared in root `[workspace.dependencies]`
- Consumers always use `<name> = { workspace = true }`
- Workspace version locked across all crates

## Comments + docs

- No internal labels (Phase / PR references) in source code
- No stale feature references in docs
- PR memos and commit messages may use internal labels freely

## CI parity

Every push runs: `cargo fmt --all --check` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` + `cargo check --workspace --no-default-features`.
