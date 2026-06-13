# Contributing

We welcome contributions! The project is organized as a Rust workspace with 29 crates.

## Getting started

```bash
git clone https://github.com/keshavashiya/brain.git
cd brain
cargo build --workspace
cargo test --workspace
```

## Development tools

```bash
just build       # Build workspace (debug)
just test        # Run all tests
just ci          # fmt + clippy + tests
just fmt         # Format code
just lint        # Clippy
just serve-dev   # Start with debug logging
```

## PR checklist

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- One intent per PR
- Conventional commits (`feat:`, `fix:`, `docs:`, etc.)

## Key conventions

- Single-word crate names matching folder names
- `brainos-` package prefix for crates.io
- Every capability lives in its backend crate, not in `cli`
- Operator commands (init, doctor, service, vault, config) stay CLI-only
- CI parity enforced before every push

## Documentation

- Public docs are at **[keshavashiya.github.io/brain](https://keshavashiya.github.io/brain)**
- Root `ARCHITECTURE.md` covers the high-level design
- `CHANGELOG.md` tracks user-facing changes
