//! Cold-start benchmarks for the `brain` binary.
//!
//! These pin the cost of the two dominant pieces of process startup so a
//! regression (an O(n) migration walk, an accidental sync read, a heavier
//! config layer) shows up as a number, not a "feels slow". The wiring/serve
//! path is intentionally excluded — it binds sockets and spawns tasks, which a
//! microbenchmark can't isolate.
//!
//! Run: `cargo bench -p brainos --bench startup`

use std::hint::black_box;

use brain::BrainConfig;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use storage::SqlitePool;

fn startup_benches(c: &mut Criterion) {
    // Config resolution from the embedded defaults + env, with the user-config
    // layer forced off so the measurement is independent of whatever sits in
    // the developer's ~/.brain. (Edition 2021: set_var is safe.)
    std::env::set_var("BRAIN_CONFIG", "/nonexistent/brain-bench-config.yaml");
    c.bench_function("config_load_defaults", |b| {
        b.iter(|| black_box(BrainConfig::load().expect("defaults must load")));
    });

    // Open + migrate a fresh database — the dominant cold-start cost. A fresh
    // temp dir per iteration forces the full migration walk every time, which
    // is the worst case (first run on a new install).
    c.bench_function("sqlite_open_and_migrate_fresh", |b| {
        b.iter_batched(
            || tempfile::tempdir().expect("tempdir"),
            |dir| {
                let path = dir.path().join("brain.db");
                black_box(SqlitePool::open(&path).expect("open + migrate"));
                dir // returned so it drops (and cleans up) outside the timed body
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, startup_benches);
criterion_main!(benches);
