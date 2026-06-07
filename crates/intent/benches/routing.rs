//! Benchmarks for SIT capability routing.
//!
//! Routing is on the hot path of every `ToolCall`: [`DefaultIntentRouter`]
//! scores the incoming token against *every* registered tool on each
//! `resolve`, so cost scales with the registry size. These benches track both
//! the per-candidate scoring kernel and the end-to-end resolve as the registry
//! grows, so a regression in either shows up before users feel it.

use std::sync::Arc;

use brainos_intent::{
    BackendId, DefaultIntentRouter, InMemoryToolRegistry, IntentRouter, IntentToken, Object,
    Provenance, ToolAnnotations, ToolDescriptor, ToolRegistry, ToolSource, ToolUsage, Verb,
};
use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn token(namespace: &str, action: &str) -> IntentToken {
    IntentToken::new(
        Verb::new(namespace, action),
        Object {
            kind: "text".into(),
            value: serde_json::Value::String("hello".into()),
        },
        Provenance::User {
            raw_input: "do the thing".into(),
            ui_origin: None,
            ts: Utc::now(),
        },
        "bench".into(),
    )
}

fn tool(id: usize, namespace: &str, action: &str) -> ToolDescriptor {
    ToolDescriptor {
        tool_id: format!("tool-{id}"),
        source: ToolSource::NativeBackend {
            backend: BackendId::new(namespace),
        },
        verb: Verb::new(namespace, action),
        description: "a benchmark tool".into(),
        input_schema: serde_json::json!({}),
        output_schema: None,
        capabilities: vec![format!("{namespace}.{action}")],
        annotations: ToolAnnotations::default(),
        usage: ToolUsage::default(),
        embedding: None,
    }
}

/// The pure per-candidate scoring kernel (verb match + capability Jaccard).
fn bench_score(c: &mut Criterion) {
    let tok = token("memory", "store");
    let exact = tool(0, "memory", "store");
    c.bench_function("intent/score_single", |b| {
        b.iter(|| {
            DefaultIntentRouter::score(std::hint::black_box(&tok), std::hint::black_box(&exact))
        })
    });
}

/// End-to-end resolve over registries of increasing size — the realistic hot
/// path (list + score-all + pick-best).
fn bench_resolve(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let namespaces = [
        "memory", "net", "shell", "schedule", "notify", "fs", "terminal", "mcp",
    ];
    let actions = [
        "store", "http", "exec", "create", "send", "read", "open", "list",
    ];

    let mut group = c.benchmark_group("intent/resolve");
    for size in [8usize, 64, 256] {
        let registry = Arc::new(InMemoryToolRegistry::new());
        rt.block_on(async {
            for i in 0..size {
                let ns = namespaces[i % namespaces.len()];
                let act = actions[i % actions.len()];
                registry.register(tool(i, ns, act)).await.unwrap();
            }
            // Guarantee one exact match exists so resolve exercises the
            // selection path, not just the no-match fallback.
            registry
                .register(tool(size, "memory", "store"))
                .await
                .unwrap();
        });
        let router = DefaultIntentRouter::new(registry);
        let tok = token("memory", "store");
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                rt.block_on(router.resolve(std::hint::black_box(&tok)))
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_score, bench_resolve);
criterion_main!(benches);
