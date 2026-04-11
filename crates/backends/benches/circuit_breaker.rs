//! Benchmarks for `CircuitBreaker` state transitions.
//!
//! These are pure atomic operations — expected to be sub-microsecond.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_circuit_breaker(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker");

    group.bench_function("is_open_closed", |b| {
        let cb = brainos_backends::CircuitBreaker::new("bench", 5, 60);
        b.iter(|| cb.is_open());
    });

    group.bench_function("is_open_after_failures", |b| {
        let cb = brainos_backends::CircuitBreaker::new("bench", 5, 60);
        for _ in 0..5 {
            cb.record_failure();
        }
        b.iter(|| cb.is_open());
    });

    group.bench_function("record_success", |b| {
        let cb = brainos_backends::CircuitBreaker::new("bench", 5, 60);
        b.iter(|| cb.record_success());
    });

    group.bench_function("record_failure", |b| {
        let cb = brainos_backends::CircuitBreaker::new("bench", 1000, 60);
        b.iter(|| cb.record_failure());
    });

    group.bench_function("failure_then_success_cycle", |b| {
        let cb = brainos_backends::CircuitBreaker::new("bench", 1000, 60);
        b.iter(|| {
            cb.record_failure();
            cb.record_success();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_circuit_breaker);
criterion_main!(benches);
