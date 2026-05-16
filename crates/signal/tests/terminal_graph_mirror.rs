//! PR-4Bd acceptance — a terminal session lifecycle (open → close)
//! must leave exactly three nodes and two edges in the graph.

#![cfg(unix)]

use std::sync::Arc;

use brainos_signal::terminal_graph_mirror::HippocampusTerminalSink;
use hippocampus::{EpisodicGraph, SqliteGraph};
use storage::SqlitePool;
use terminal::{
    pb::{OpenRequest, PtySize},
    TerminalBridge,
};

#[tokio::test]
async fn terminal_lifecycle_writes_three_nodes_and_two_edges() {
    let graph: Arc<dyn EpisodicGraph> =
        Arc::new(SqliteGraph::new(SqlitePool::open_memory().unwrap()));
    let sink = Arc::new(HippocampusTerminalSink::new(graph.clone()));
    let bridge = TerminalBridge::new().with_graph_sink(sink);
    let svc = bridge.svc();

    // Open a tiny shell. /bin/sh is universally available on unix.
    let handle = svc
        .open_via_pipeline(
            OpenRequest {
                program: "/bin/sh".to_string(),
                args: vec!["-c".into(), "exit 0".into()],
                env: Default::default(),
                cwd: String::new(),
                initial_size: Some(PtySize {
                    rows: 24,
                    cols: 80,
                    pixel_width: 0,
                    pixel_height: 0,
                }),
                client_id: String::new(),
                set_controlling_tty: false,
            },
            None,
        )
        .await
        .expect("open");

    // Close it. Don't care about the exit status — just that the
    // lifecycle completes.
    let _ = svc
        .close_via_pipeline(&handle.session_id)
        .await
        .expect("close");

    // Inspect the graph: three nodes, two edges.
    let nodes = graph.list_all_nodes().expect("list_all_nodes");
    assert_eq!(
        nodes.len(),
        3,
        "expected 3 nodes (tool_call + open_event + close_event), got {}: {:?}",
        nodes.len(),
        nodes.iter().map(|n| n.kind.as_str()).collect::<Vec<_>>()
    );
    let tool_calls: Vec<_> = nodes
        .iter()
        .filter(|n| n.kind.as_str() == "tool_call")
        .collect();
    let events: Vec<_> = nodes
        .iter()
        .filter(|n| n.kind.as_str() == "terminal_event")
        .collect();
    assert_eq!(tool_calls.len(), 1, "exactly one tool_call node");
    assert_eq!(events.len(), 2, "two terminal_event nodes (open + close)");

    // Edges: tool_call → open_event, open_event → close_event. Walk
    // from the tool_call to verify the chain.
    let tool_call = tool_calls[0];
    let nb = graph.neighbors(&tool_call.id).unwrap();
    assert_eq!(nb.len(), 1, "tool_call has exactly one outgoing edge");
    let (open_edge, open_event) = &nb[0];
    assert_eq!(open_edge.kind.as_str(), "causal_produced");
    assert_eq!(open_event.kind.as_str(), "terminal_event");

    let nb2 = graph.neighbors(&open_event.id).unwrap();
    assert_eq!(nb2.len(), 1, "open_event has exactly one outgoing edge");
    let (close_edge, close_event) = &nb2[0];
    assert_eq!(close_edge.kind.as_str(), "causal_produced");
    assert_eq!(close_event.kind.as_str(), "terminal_event");

    // The full chain (3 ids, 2 hops) is reachable via path().
    let path = graph
        .path(&tool_call.id, &close_event.id, 5)
        .unwrap()
        .expect("path through 3-node chain");
    assert_eq!(path.len(), 3);

    // Body sanity: open event payload notes the session id and phase.
    let body = &open_event.body;
    assert_eq!(body.get("phase").and_then(|v| v.as_str()), Some("open"));
    assert_eq!(
        body.get("session_id").and_then(|v| v.as_str()),
        Some(handle.session_id.as_str())
    );
    let close_body = &close_event.body;
    assert_eq!(
        close_body.get("phase").and_then(|v| v.as_str()),
        Some("close")
    );
}

#[tokio::test]
async fn bridge_without_graph_sink_skips_mirror() {
    // Confirm the mirror is opt-in — an unwired bridge runs the
    // lifecycle with no graph side-effects (no graph attached, so
    // there's nothing to inspect; this test just guards against a
    // regression where calling open without a sink panics).
    let bridge = TerminalBridge::new();
    let svc = bridge.svc();
    let handle = svc
        .open_via_pipeline(
            OpenRequest {
                program: "/bin/sh".to_string(),
                args: vec!["-c".into(), "exit 0".into()],
                env: Default::default(),
                cwd: String::new(),
                initial_size: None,
                client_id: String::new(),
                set_controlling_tty: false,
            },
            None,
        )
        .await
        .expect("open without sink");
    let _ = svc.close_via_pipeline(&handle.session_id).await;
}
