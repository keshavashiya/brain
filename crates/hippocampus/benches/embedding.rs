//! Benchmarks for deterministic fallback embedding generation (FNV-1a + xorshift64*).
//!
//! This measures the offline fallback path — no network latency involved.

use brainos_hippocampus::embedding::{deterministic_fallback_embedding, sanitize_embedding};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_fallback_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_fallback");

    for dim in [384, 768, 1536] {
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, &dim| {
            b.iter(|| deterministic_fallback_embedding("benchmark seed text for embedding", dim));
        });
    }
    group.finish();
}

fn bench_sanitize_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_sanitize");

    // Valid vector (normalize path)
    let valid: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001) - 0.384).collect();
    group.bench_function("valid_768", |b| {
        b.iter(|| sanitize_embedding(valid.clone(), 768, "seed"));
    });

    // Invalid vector (fallback path)
    let nan_vec = vec![f32::NAN; 768];
    group.bench_function("invalid_nan_768", |b| {
        b.iter(|| sanitize_embedding(nan_vec.clone(), 768, "seed"));
    });

    // Wrong dimensions (fallback path)
    let wrong = vec![0.1_f32; 384];
    group.bench_function("wrong_dim_384_to_768", |b| {
        b.iter(|| sanitize_embedding(wrong.clone(), 768, "seed"));
    });

    group.finish();
}

criterion_group!(benches, bench_fallback_embedding, bench_sanitize_embedding);
criterion_main!(benches);
