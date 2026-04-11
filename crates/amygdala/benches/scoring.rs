//! Benchmarks for `ImportanceScorer::score()` with varying novelty cache states.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_importance_scoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("importance_scoring");

    // Empty cache — everything is novel
    group.bench_function("empty_cache", |b| {
        let scorer = brainos_amygdala::ImportanceScorer::new();
        b.iter(|| {
            scorer.score("Remember to deploy the new API server by Friday ASAP");
        });
    });

    // Half-full cache — mix of novel and seen
    group.bench_function("half_full_cache", |b| {
        let scorer = brainos_amygdala::ImportanceScorer::new();
        // Pre-fill with 5000 unique entries
        for i in 0..5_000 {
            scorer.score(&format!(
                "Pre-fill message number {i} about topic {}",
                i % 100
            ));
        }
        b.iter(|| {
            scorer.score("Remember to deploy the new API server by Friday ASAP");
        });
    });

    // Full cache at capacity (10K) — LRU eviction happening
    group.bench_function("full_cache_10k", |b| {
        let scorer = brainos_amygdala::ImportanceScorer::new();
        // Fill to capacity
        for i in 0..10_000 {
            scorer.score(&format!(
                "Pre-fill message number {i} about topic {}",
                i % 100
            ));
        }
        b.iter(|| {
            scorer.score("Remember to deploy the new API server by Friday ASAP");
        });
    });

    group.finish();
}

fn bench_scoring_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("scoring_throughput");
    let scorer = brainos_amygdala::ImportanceScorer::new();

    let messages: Vec<String> = (0..1_000)
        .map(|i| {
            format!(
                "{} {} {}",
                ["Remember", "Note that", "FYI", "Update:", "Important:"][i % 5],
                ["the server", "our API", "the database", "CI/CD", "the team"][i % 5],
                [
                    "needs updating",
                    "is broken",
                    "was deployed",
                    "requires review"
                ][i % 4],
            )
        })
        .collect();

    group.bench_with_input(BenchmarkId::new("batch", 1000), &messages, |b, msgs| {
        b.iter(|| {
            for msg in msgs {
                scorer.score(msg);
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_importance_scoring, bench_scoring_throughput);
criterion_main!(benches);
