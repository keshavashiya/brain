//! Observations → graph acceptance (close-the-loops M1) — operational
//! events mirrored off the bus must be answerable via recall: "what
//! changed around the time X broke" is a graph walk, and a pressure
//! event is findable by content.

use std::sync::Arc;

use brainos_signal::observation_graph_mirror::ObservationGraphMirror;
use chrono::Utc;
use hippocampus::{DualMemoryReader, EpisodicGraph, EpisodicStore, SqliteGraph};
use observe::BrainEvent;
use storage::SqlitePool;
use uuid::Uuid;

#[tokio::test]
async fn recall_surfaces_a_mirrored_pressure_event_by_content() {
    let pool = SqlitePool::open_memory().unwrap();
    let graph: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool.clone()));
    let mirror = ObservationGraphMirror::new(graph.clone());

    mirror
        .mirror(&BrainEvent::ResourcePressure {
            id: Uuid::new_v4(),
            gauge: "rss".into(),
            value: 2304.0,
            threshold: 2048.0,
            severity: "warn".into(),
            ts: Utc::now(),
        })
        .await
        .expect("mirror write")
        .expect("pressure is an observation");

    // The same dual reader `brain serve` wires for recall: graph-first,
    // FTS half only (no vector store attached — degraded installs must
    // still surface observations).
    let legacy = Arc::new(EpisodicStore::new(pool));
    let reader = DualMemoryReader::dual(legacy, graph);
    let candidates = reader
        .recall_candidates("resource pressure rss", vec![], 5, None)
        .await
        .expect("recall candidates");

    assert_eq!(
        candidates.fts.len(),
        1,
        "the mirrored pressure event must surface by content"
    );
    let (id, _) = &candidates.fts[0];
    let hydrated = &candidates.hydration[id];
    assert!(
        hydrated.content.contains("pressure"),
        "hydrated content should read like the observation: {}",
        hydrated.content
    );
}

#[tokio::test]
async fn what_changed_around_a_service_outage_is_a_graph_walk() {
    let pool = SqlitePool::open_memory().unwrap();
    let graph: Arc<dyn EpisodicGraph> = Arc::new(SqliteGraph::new(pool));
    let mirror = ObservationGraphMirror::new(graph.clone());

    // The story: the network degrades, ollama goes down, then recovers.
    for ev in [
        BrainEvent::ConnectivityChanged {
            id: Uuid::new_v4(),
            state: "degraded".into(),
            previous: "online".into(),
            detail: "1 of 2 endpoints unreachable".into(),
            ts: Utc::now(),
        },
        BrainEvent::ServiceHealthChanged {
            id: Uuid::new_v4(),
            service: "ollama".into(),
            target: "http://localhost:11434/api/tags".into(),
            healthy: false,
            detail: "connection refused".into(),
            ts: Utc::now(),
        },
        BrainEvent::ServiceHealthChanged {
            id: Uuid::new_v4(),
            service: "ollama".into(),
            target: "http://localhost:11434/api/tags".into(),
            healthy: true,
            detail: String::new(),
            ts: Utc::now(),
        },
    ] {
        mirror.mirror(&ev).await.unwrap().unwrap();
    }

    // "When did ollama break?" — findable by content…
    let hits = graph.search_text("ollama down", 5, None).unwrap();
    assert!(!hits.is_empty(), "outage node should match by content");
    let down_id = &hits[0].id;

    // …and "what happened to it afterwards" is the transition walk.
    let nb = graph.neighbors(down_id).unwrap();
    let recovery: Vec<_> = nb
        .iter()
        .filter(|(e, _)| e.kind.as_str() == "transition")
        .collect();
    assert_eq!(recovery.len(), 1, "down chains to recovery");
    assert_eq!(
        recovery[0].1.body.get("healthy").and_then(|v| v.as_bool()),
        Some(true)
    );

    // The surrounding context (connectivity degradation) sits in the same
    // graph, timestamped, so "what changed around that time" is answerable.
    let ctx = graph.search_text("connectivity degraded", 5, None).unwrap();
    assert_eq!(ctx.len(), 1);
}
