//! Benchmarks for the memory consolidation pipeline.
//!
//! Pre-seeds episodes with varying importance/decay and measures consolidation time.

use brainos_hippocampus::consolidation::{ConsolidationConfig, Consolidator};
use brainos_hippocampus::episodic::EpisodicStore;
use criterion::{criterion_group, criterion_main, Criterion};
use storage::sqlite::SqlitePool;

fn seed_episodic_for_consolidation(n: usize) -> EpisodicStore {
    let pool = SqlitePool::open_memory().unwrap();
    let store = EpisodicStore::new(pool);
    let session_id = store.create_session("bench").unwrap();

    for i in 0..n {
        // Vary importance: some below threshold (candidates for pruning), some above
        let importance = if i % 5 == 0 {
            0.01 // very low — pruning candidate
        } else if i % 3 == 0 {
            0.9 // high — promotion candidate
        } else {
            0.3 + (i % 10) as f64 * 0.05
        };

        let content = format!(
            "Episode {} discussing {} in context of {}",
            i,
            [
                "architecture",
                "testing",
                "deployment",
                "monitoring",
                "refactoring"
            ][i % 5],
            ["production", "staging", "development"][i % 3],
        );

        store
            .store_episode(&session_id, "user", &content, importance, None, None)
            .unwrap();
    }
    store
}

fn bench_consolidation(c: &mut Criterion) {
    let mut group = c.benchmark_group("consolidation");

    let config = ConsolidationConfig {
        prune_threshold: 0.05,
        max_prune_per_run: 100,
        ..Default::default()
    };

    let episodic = seed_episodic_for_consolidation(1_000);
    let consolidator = Consolidator::new(config);

    group.bench_function("consolidate_1k_episodes", |b| {
        b.iter(|| {
            consolidator.consolidate(&episodic).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_consolidation);
criterion_main!(benches);
