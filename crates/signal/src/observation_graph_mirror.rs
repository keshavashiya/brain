//! Mirror of operational observations into a [`hippocampus::EpisodicGraph`].
//!
//! Monitoring has sensors but no memory: resource-pressure crossings,
//! service up/down transitions, reflex firings, connectivity and power
//! changes, and baseline drift are all published as [`observe::BrainEvent`]s
//! and then die on the bus. This mirror is the missing loop — a bus
//! subscriber hands each event to [`ObservationGraphMirror::mirror`], which
//! writes an `observation` node into the episodic graph so "what changed
//! around the time X broke" becomes answerable via recall.
//!
//! Lives in `signal` for the same reason as the terminal mirror: this crate
//! is the natural assembly point for `observe` + `hippocampus`, keeping
//! `observe` storage-free and `hippocampus` event-free.
//!
//! ## What each node carries
//!
//! Every mirrored node has kind `observation`, a body holding the event's
//! fields plus a human-phrased `summary` line (so FTS/ANN recall matches the
//! way a user would ask — "memory pressure", "ollama went down"), and the
//! event's correlation id.
//!
//! ## Edges
//!
//! - **`transition`** chains successive observations of the same stream —
//!   a service's down node points at its recovery node, a connectivity
//!   `offline` at the following `online`, one pressure crossing of a gauge
//!   at the next. This is the "what happened to X over time" walk.
//! - **`correlated`** links two mirrored events that share a correlation id
//!   (`BrainEvent::id`), i.e. belong to one signal flow.
//!
//! Both edge maps are in-memory and bounded: observations are edge-triggered
//! and low-volume, so losing chain continuity across a restart (or a rare
//! map reset) costs an edge, never a node.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use hippocampus::{Edge, EdgeKind, Embedder, EpisodicGraph, Node, NodeKind};
use observe::BrainEvent;
use storage::RuVectorStore;
use uuid::Uuid;

use crate::graph_embed::NodeEmbedder;

/// Cap on the in-memory stream / correlation maps. Hitting it resets the
/// map (losing future edge continuity for old streams, never nodes) —
/// simpler than an LRU and adequate for edge-triggered volumes.
const MAX_TRACKED: usize = 512;

#[derive(Debug, thiserror::Error)]
pub enum ObservationMirrorError {
    #[error("graph backend error: {0}")]
    Backend(String),
}

/// One mirrorable observation: the node body plus the optional stream the
/// node chains into.
struct Observation {
    /// Key for the `transition` chain ("service:ollama", "connectivity",
    /// "pressure:rss", …); `None` for unchained observations.
    stream: Option<String>,
    body: serde_json::Value,
}

/// Bus-event → episodic-graph mirror. Construct once, then feed every
/// [`BrainEvent`] through [`Self::mirror`]; non-observation events are
/// ignored.
pub struct ObservationGraphMirror {
    graph: Arc<dyn EpisodicGraph>,
    namespace: String,
    embedder: NodeEmbedder,
    /// stream key → id of the stream's most recent node.
    streams: Mutex<HashMap<String, String>>,
    /// correlation id → id of that flow's most recent node.
    correlations: Mutex<HashMap<Uuid, String>>,
}

impl ObservationGraphMirror {
    pub fn new(graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph,
            namespace: "personal".to_string(),
            embedder: NodeEmbedder::default(),
            streams: Mutex::new(HashMap::new()),
            correlations: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    /// Wire the namespace residency policy: when this mirror's namespace is
    /// `local_only` and the embedder is remote, mirrored nodes get the
    /// deterministic fallback vector instead of a remote embed.
    pub fn with_residency(mut self, residency: brain::ResidencyPolicy) -> Self {
        self.embedder.set_residency(residency);
        self
    }

    /// Wire the embedding write-path. Mirrored nodes are embedded through
    /// `embedder` and stored in the shared `vectors` store's `graph_vec`
    /// collection (sized to `embedding_dim`) — this is what lets
    /// observations participate in ANN recall, not just FTS.
    pub fn with_embedding(
        mut self,
        embedder: Option<Arc<Embedder>>,
        vectors: RuVectorStore,
        embedding_dim: usize,
    ) -> Self {
        self.embedder
            .set_embedding(embedder, vectors, embedding_dim);
        self
    }

    /// Mirror `ev` into the graph if it is an observation event. Returns the
    /// new node's id, or `None` for event kinds this mirror ignores.
    pub async fn mirror(&self, ev: &BrainEvent) -> Result<Option<String>, ObservationMirrorError> {
        let Some(obs) = observation_for(ev) else {
            return Ok(None);
        };
        let mut node = Node::new(
            NodeKind::new("observation"),
            obs.body,
            self.namespace.clone(),
            None,
        );
        self.embedder.embed_and_link(&mut node).await;
        self.graph
            .add_node(&node)
            .map_err(|e| ObservationMirrorError::Backend(e.to_string()))?;

        if let Some(stream) = obs.stream {
            if let Some(prev) = remember(&self.streams, stream, node.id.clone()) {
                self.graph
                    .add_edge(&Edge::new(&prev, &node.id, EdgeKind::new("transition")))
                    .map_err(|e| ObservationMirrorError::Backend(e.to_string()))?;
            }
        }
        if let Some(prev) = remember(&self.correlations, ev.id(), node.id.clone()) {
            self.graph
                .add_edge(&Edge::new(&prev, &node.id, EdgeKind::new("correlated")))
                .map_err(|e| ObservationMirrorError::Backend(e.to_string()))?;
        }
        Ok(Some(node.id))
    }
}

/// Record `value` under `key`, returning the previous value if one was
/// tracked. Resets the map at [`MAX_TRACKED`] entries.
fn remember<K: std::hash::Hash + Eq>(
    map: &Mutex<HashMap<K, String>>,
    key: K,
    value: String,
) -> Option<String> {
    let mut map = map.lock().expect("observation mirror map poisoned");
    if map.len() >= MAX_TRACKED && !map.contains_key(&key) {
        map.clear();
    }
    map.insert(key, value)
}

/// Map a bus event to its graph projection; `None` for kinds that are not
/// operational observations (pipeline chatter, audit, terminal lifecycles —
/// the latter have their own mirror).
fn observation_for(ev: &BrainEvent) -> Option<Observation> {
    let correlation_id = ev.id().to_string();
    match ev {
        BrainEvent::ResourcePressure {
            gauge,
            value,
            threshold,
            severity,
            ..
        } => Some(Observation {
            stream: Some(format!("pressure:{gauge}")),
            body: serde_json::json!({
                "observation": "resource_pressure",
                "gauge": gauge,
                "value": value,
                "threshold": threshold,
                "severity": severity,
                "summary": format!(
                    "resource pressure {severity}: {gauge} at {value:.0} crossed ceiling {threshold:.0}"
                ),
                "correlation_id": correlation_id,
            }),
        }),
        BrainEvent::ServiceHealthChanged {
            service,
            target,
            healthy,
            detail,
            ..
        } => {
            let summary = if *healthy {
                format!("service {service} recovered ({target})")
            } else {
                format!("service {service} went down ({target}): {detail}")
            };
            Some(Observation {
                stream: Some(format!("service:{service}")),
                body: serde_json::json!({
                    "observation": "service_health",
                    "service": service,
                    "target": target,
                    "healthy": healthy,
                    "detail": detail,
                    "summary": summary,
                    "correlation_id": correlation_id,
                }),
            })
        }
        BrainEvent::ReflexFired {
            trigger_id,
            payload,
            ..
        } => Some(Observation {
            stream: Some(format!("reflex:{trigger_id}")),
            body: serde_json::json!({
                "observation": "reflex_fired",
                "trigger_id": trigger_id,
                "payload": payload,
                "summary": format!("reflex {trigger_id} fired"),
                "correlation_id": correlation_id,
            }),
        }),
        BrainEvent::ConnectivityChanged {
            state,
            previous,
            detail,
            ..
        } => Some(Observation {
            stream: Some("connectivity".to_string()),
            body: serde_json::json!({
                "observation": "connectivity",
                "state": state,
                "previous": previous,
                "detail": detail,
                "summary": format!("connectivity {state} (was {previous}): {detail}"),
                "correlation_id": correlation_id,
            }),
        }),
        BrainEvent::PowerStateChanged {
            state,
            previous,
            detail,
            ..
        } => Some(Observation {
            stream: Some("power".to_string()),
            body: serde_json::json!({
                "observation": "power",
                "state": state,
                "previous": previous,
                "detail": detail,
                "summary": format!("power {state} (was {previous}): {detail}"),
                "correlation_id": correlation_id,
            }),
        }),
        BrainEvent::BaselineDrift {
            from,
            to,
            added,
            removed,
            changed,
            keys,
            ..
        } => Some(Observation {
            stream: Some("baseline".to_string()),
            body: serde_json::json!({
                "observation": "baseline_drift",
                "from": from,
                "to": to,
                "added": added,
                "removed": removed,
                "changed": changed,
                "keys": keys,
                "summary": format!(
                    "baseline drift {from} → {to}: {changed} changed, {added} added, {removed} removed ({})",
                    keys.join(", ")
                ),
                "correlation_id": correlation_id,
            }),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use hippocampus::SqliteGraph;
    use storage::SqlitePool;

    fn mirror() -> (ObservationGraphMirror, Arc<dyn EpisodicGraph>) {
        let graph: Arc<dyn EpisodicGraph> =
            Arc::new(SqliteGraph::new(SqlitePool::open_memory().unwrap()));
        (ObservationGraphMirror::new(graph.clone()), graph)
    }

    fn pressure(id: Uuid) -> BrainEvent {
        BrainEvent::ResourcePressure {
            id,
            gauge: "rss".into(),
            value: 2304.0,
            threshold: 2048.0,
            severity: "warn".into(),
            ts: Utc::now(),
        }
    }

    fn service(id: Uuid, healthy: bool) -> BrainEvent {
        BrainEvent::ServiceHealthChanged {
            id,
            service: "ollama".into(),
            target: "http://localhost:11434/api/tags".into(),
            healthy,
            detail: if healthy {
                String::new()
            } else {
                "connection refused".into()
            },
            ts: Utc::now(),
        }
    }

    #[tokio::test]
    async fn pressure_event_lands_as_fts_searchable_node() {
        let (m, graph) = mirror();
        let node_id = m
            .mirror(&pressure(Uuid::new_v4()))
            .await
            .unwrap()
            .expect("pressure is an observation");

        let node = graph.get_node(&node_id).unwrap().expect("node stored");
        assert_eq!(node.kind.as_str(), "observation");
        assert_eq!(
            node.body.get("observation").and_then(|v| v.as_str()),
            Some("resource_pressure")
        );

        // The DoD query: a pressure event is findable by content.
        let hits = graph.search_text("pressure rss", 5, None).unwrap();
        assert_eq!(hits.len(), 1, "FTS should surface the pressure node");
        assert_eq!(hits[0].id, node_id);
    }

    #[tokio::test]
    async fn service_down_then_up_chains_a_transition_edge() {
        let (m, graph) = mirror();
        let down = m
            .mirror(&service(Uuid::new_v4(), false))
            .await
            .unwrap()
            .unwrap();
        let up = m
            .mirror(&service(Uuid::new_v4(), true))
            .await
            .unwrap()
            .unwrap();

        let nb = graph.neighbors(&down).unwrap();
        assert_eq!(nb.len(), 1, "down node points at its recovery");
        let (edge, dst) = &nb[0];
        assert_eq!(edge.kind.as_str(), "transition");
        assert_eq!(dst.id, up);
        assert_eq!(
            dst.body.get("healthy").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn shared_correlation_id_links_observations() {
        let (m, graph) = mirror();
        let id = Uuid::new_v4();
        // A reflex firing and a pressure crossing in the same signal flow.
        let reflex = m
            .mirror(&BrainEvent::ReflexFired {
                id,
                trigger_id: "cron:nightly-backup".into(),
                payload: serde_json::json!({"entry": "nightly-backup"}),
                ts: Utc::now(),
            })
            .await
            .unwrap()
            .unwrap();
        let pressed = m.mirror(&pressure(id)).await.unwrap().unwrap();

        let correlated: Vec<_> = graph
            .neighbors(&reflex)
            .unwrap()
            .into_iter()
            .filter(|(e, _)| e.kind.as_str() == "correlated")
            .collect();
        assert_eq!(correlated.len(), 1);
        assert_eq!(correlated[0].1.id, pressed);
    }

    #[tokio::test]
    async fn non_observation_events_are_ignored() {
        let (m, graph) = mirror();
        let out = m
            .mirror(&BrainEvent::Error {
                id: Uuid::new_v4(),
                source: "test".into(),
                message: "boom".into(),
                ts: Utc::now(),
            })
            .await
            .unwrap();
        assert!(out.is_none());
        assert!(graph.list_all_nodes().unwrap().is_empty());
    }

    #[tokio::test]
    async fn connectivity_and_power_chain_their_own_streams() {
        let (m, graph) = mirror();
        let offline = m
            .mirror(&BrainEvent::ConnectivityChanged {
                id: Uuid::new_v4(),
                state: "offline".into(),
                previous: "online".into(),
                detail: "2 of 2 endpoints unreachable".into(),
                ts: Utc::now(),
            })
            .await
            .unwrap()
            .unwrap();
        // A power flip in between must not break the connectivity chain.
        m.mirror(&BrainEvent::PowerStateChanged {
            id: Uuid::new_v4(),
            state: "battery".into(),
            previous: "external".into(),
            detail: "battery at 47%".into(),
            ts: Utc::now(),
        })
        .await
        .unwrap()
        .unwrap();
        let online = m
            .mirror(&BrainEvent::ConnectivityChanged {
                id: Uuid::new_v4(),
                state: "online".into(),
                previous: "offline".into(),
                detail: String::new(),
                ts: Utc::now(),
            })
            .await
            .unwrap()
            .unwrap();

        let nb = graph.neighbors(&offline).unwrap();
        assert_eq!(nb.len(), 1);
        assert_eq!(nb[0].0.kind.as_str(), "transition");
        assert_eq!(nb[0].1.id, online);
    }

    #[tokio::test]
    async fn baseline_drift_lands_with_keys_in_summary() {
        let (m, graph) = mirror();
        let node_id = m
            .mirror(&BrainEvent::BaselineDrift {
                id: Uuid::new_v4(),
                from: "baseline v3 (2026-06-12 09:14)".into(),
                to: "current live state".into(),
                added: 1,
                removed: 0,
                changed: 1,
                keys: vec!["llm.model".into(), "adapter.http".into()],
                ts: Utc::now(),
            })
            .await
            .unwrap()
            .unwrap();
        let node = graph.get_node(&node_id).unwrap().unwrap();
        let summary = node.body.get("summary").and_then(|v| v.as_str()).unwrap();
        assert!(summary.contains("llm.model"), "summary: {summary}");
        let hits = graph.search_text("baseline drift", 5, None).unwrap();
        assert_eq!(hits.len(), 1);
    }
}
