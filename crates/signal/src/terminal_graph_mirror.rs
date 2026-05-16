//! Bridge implementation that mirrors terminal lifecycles into a
//! [`hippocampus::EpisodicGraph`].
//!
//! Lives in `signal` because this crate is the natural assembly point
//! for `terminal` + `hippocampus` — keeping the adapter here means
//! `terminal` stays storage-free and `hippocampus` stays
//! terminal-free.
//!
//! Emission per session lifecycle:
//! - **Open** writes two nodes (`tool_call` for the invocation,
//!   `terminal_event` for the open side) plus a `causal_produced`
//!   edge between them.
//! - **Close** writes one node (`terminal_event` for the close side)
//!   plus a second `causal_produced` edge from the open event to the
//!   close event.
//!
//! Net: three nodes + two edges per session.

use std::sync::Arc;

use async_trait::async_trait;
use hippocampus::{Edge, EdgeKind, EpisodicGraph, Node, NodeKind};
use identity::Principal;
use terminal::{MirrorError, TerminalGraphHandles, TerminalGraphSink};

/// `TerminalGraphSink` impl backed by an [`EpisodicGraph`].
pub struct HippocampusTerminalSink {
    graph: Arc<dyn EpisodicGraph>,
    namespace: String,
}

impl HippocampusTerminalSink {
    pub fn new(graph: Arc<dyn EpisodicGraph>) -> Self {
        Self {
            graph,
            namespace: "personal".to_string(),
        }
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }
}

fn principal_summary(p: Option<&Principal>) -> serde_json::Value {
    match p {
        Some(p) => serde_json::json!({
            "agent_id": p.agent_id,
            "tier": p.tier.to_string(),
        }),
        None => serde_json::Value::Null,
    }
}

#[async_trait]
impl TerminalGraphSink for HippocampusTerminalSink {
    async fn record_open(
        &self,
        session_id: &str,
        program: &str,
        args: &[String],
        cwd: Option<&str>,
        principal: Option<&Principal>,
    ) -> Result<TerminalGraphHandles, MirrorError> {
        let tool_call = Node::new(
            NodeKind::new("tool_call"),
            serde_json::json!({
                "verb": "terminal.open",
                "program": program,
                "args": args,
                "cwd": cwd,
                "session_id": session_id,
                "principal": principal_summary(principal),
            }),
            self.namespace.clone(),
            None,
        );
        let open_event = Node::new(
            NodeKind::new("terminal_event"),
            serde_json::json!({
                "phase": "open",
                "session_id": session_id,
                "program": program,
            }),
            self.namespace.clone(),
            None,
        );
        self.graph
            .add_node(&tool_call)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_node(&open_event)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_edge(&Edge::new(
                &tool_call.id,
                &open_event.id,
                EdgeKind::new("causal_produced"),
            ))
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        Ok(TerminalGraphHandles {
            tool_call_node_id: tool_call.id,
            open_event_node_id: open_event.id,
        })
    }

    async fn record_close(
        &self,
        handles: &TerminalGraphHandles,
        session_id: &str,
        exit_code: i32,
        was_killed: bool,
    ) -> Result<(), MirrorError> {
        let close_event = Node::new(
            NodeKind::new("terminal_event"),
            serde_json::json!({
                "phase": "close",
                "session_id": session_id,
                "exit_code": exit_code,
                "was_killed": was_killed,
            }),
            self.namespace.clone(),
            None,
        );
        self.graph
            .add_node(&close_event)
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        self.graph
            .add_edge(&Edge::new(
                &handles.open_event_node_id,
                &close_event.id,
                EdgeKind::new("causal_produced"),
            ))
            .map_err(|e| MirrorError::Backend(e.to_string()))?;
        Ok(())
    }
}
