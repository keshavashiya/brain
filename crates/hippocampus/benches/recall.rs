//! Benchmarks for `RecallEngine::recall()` — BM25, vector, and hybrid search.

use brainos_hippocampus::embedding::deterministic_fallback_embedding;
use brainos_hippocampus::episodic::EpisodicStore;
use brainos_hippocampus::search::{RecallConfig, RecallEngine};
use brainos_hippocampus::semantic::SemanticStore;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use storage::ruvector::RuVectorStore;
use storage::sqlite::SqlitePool;

const VECTOR_DIM: usize = 768;

/// Seed an in-memory episodic store with `n` episodes and return the store.
fn seed_episodic(n: usize) -> EpisodicStore {
    let pool = SqlitePool::open_memory().unwrap();
    let store = EpisodicStore::new(pool);
    let session_id = store.create_session("bench").unwrap();

    for i in 0..n {
        let content = format!(
            "Episode {} about topic {} with details on project {} and notes about {}",
            i,
            ["rust", "python", "typescript", "go", "java"][i % 5],
            ["brain-os", "web-app", "cli-tool", "api-server"][i % 4],
            ["performance", "testing", "deployment", "refactoring"][i % 4],
        );
        store
            .store_episode(
                &session_id,
                "user",
                &content,
                0.5 + (i % 10) as f64 * 0.05,
                None,
                None,
            )
            .unwrap();
    }
    store
}

/// Seed a semantic store (RuVector + SQLite) with `n` facts.
fn seed_semantic(rt: &tokio::runtime::Runtime, dir: &tempfile::TempDir, n: usize) -> SemanticStore {
    let pool = SqlitePool::open_memory().unwrap();
    let ruv = rt
        .block_on(RuVectorStore::open(dir.path(), VECTOR_DIM))
        .unwrap();
    let store = SemanticStore::new(pool, ruv);

    for i in 0..n {
        let content = format!(
            "Fact {}: the {} framework uses {} for {}",
            i,
            ["axum", "actix", "warp", "rocket", "tide"][i % 5],
            ["tokio", "async-std", "smol"][i % 3],
            ["web servers", "APIs", "streaming", "microservices"][i % 4],
        );
        let embedding = deterministic_fallback_embedding(&content, VECTOR_DIM);
        rt.block_on(store.store_fact(
            "bench",
            "benchmark",
            "system",
            "knows",
            &content,
            1.0,
            None,
            embedding,
            None,
        ))
        .unwrap();
    }
    store
}

fn bench_bm25_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_bm25");
    for size in [100, 1_000, 10_000] {
        let episodic = seed_episodic(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                episodic
                    .search_bm25("rust performance testing", 20, None, None)
                    .unwrap();
            });
        });
    }
    group.finish();
}

fn bench_hybrid_recall(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("recall_hybrid");

    for size in [100, 1_000] {
        let episodic = seed_episodic(size);
        let dir = tempfile::tempdir().unwrap();
        let semantic = seed_semantic(&rt, &dir, size);
        let engine = RecallEngine::new(RecallConfig::default());
        let query_vec = deterministic_fallback_embedding("rust performance", VECTOR_DIM);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                rt.block_on(engine.recall(
                    "rust performance",
                    query_vec.clone(),
                    &episodic,
                    &semantic,
                    20,
                    None,
                    None,
                ))
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bm25_search, bench_hybrid_recall);
criterion_main!(benches);
