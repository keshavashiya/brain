//! MCP egress-scope enforcement: a tool call outside the consented scope
//! fails closed *and* leaves an audit row.
//!
//! Drives the public [`SignalProcessor::process`] surface through the same
//! capability-kernel path as the Phase 3 acceptance test (`/tool` →
//! `Intent::ToolCall` → router → `ToolRoute::Mcp` → `MCPHost::call`), but with
//! a scope-limited [`mcphost::InMemoryMcpHost`] and a recording audit trail so
//! the fail-closed + audit-row DoD is asserted end to end.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};

/// Recording audit trail: captures every `record` so the test can assert a
/// scope-denied row landed. The non-`record` methods are unused here.
#[derive(Default)]
struct RecordingAuditTrail {
    entries: Mutex<Vec<audit::AuditEntry>>,
}

#[async_trait]
impl audit::AuditTrail for RecordingAuditTrail {
    async fn record(&self, entry: audit::AuditEntry) -> Result<String, audit::AuditError> {
        let id = entry.id.clone();
        self.entries.lock().unwrap().push(entry);
        Ok(id)
    }

    async fn query(
        &self,
        _spec: audit::AuditQuerySpec,
    ) -> Result<Vec<audit::AuditEntry>, audit::AuditError> {
        Ok(self.entries.lock().unwrap().clone())
    }

    async fn summarize(
        &self,
        _window: chrono::Duration,
    ) -> Result<audit::schema::AuditSummary, audit::AuditError> {
        Ok(audit::schema::AuditSummary {
            total_entries: 0,
            by_outcome: Default::default(),
            by_tier: Default::default(),
            by_source: Default::default(),
            avg_duration_ms: None,
        })
    }

    async fn rollback(
        &self,
        _entry_id: &str,
    ) -> Result<Option<audit::RollbackPlan>, audit::AuditError> {
        Ok(None)
    }

    async fn prune(&self, _older_than: chrono::Duration) -> Result<usize, audit::AuditError> {
        Ok(0)
    }
}

async fn make_processor() -> SignalProcessor {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    std::mem::forget(temp);
    processor
}

fn text(resp: SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn out_of_scope_mcp_call_fails_closed_and_audits() {
    let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
    let router: Arc<dyn intent::IntentRouter> =
        Arc::new(intent::DefaultIntentRouter::new(registry.clone()));

    // Mount a server whose consented scope allows only `read_*` tools. The
    // `write_file` tool is registered for routing but is *outside* the scope.
    let host: Arc<dyn mcphost::MCPHost> = Arc::new(mcphost::InMemoryMcpHost::new());
    host.mount_with_scopes(
        "fs".into(),
        mcphost::stdio_cfg("mcp-fs", vec![]),
        mcphost::ServerScopes {
            allowed_tools: vec!["read_*".into()],
            ..Default::default()
        },
    )
    .await
    .unwrap();
    registry
        .register(intent::ToolDescriptor {
            tool_id: "mcp:fs:write_file".into(),
            source: intent::ToolSource::McpServer {
                server: "fs".into(),
            },
            verb: intent::Verb::new("mcp", "write_file"),
            description: "write a file".into(),
            input_schema: serde_json::json!({ "type": "object" }),
            output_schema: None,
            capabilities: vec![],
            annotations: intent::ToolAnnotations::default(),
            usage: intent::ToolUsage::default(),
            embedding: None,
        })
        .await
        .unwrap();

    let audit_trail = Arc::new(RecordingAuditTrail::default());
    let processor = make_processor()
        .await
        .with_tool_registry(registry.clone())
        .with_intent_router(router)
        .with_mcp_host(host)
        .with_audit_trail(audit_trail.clone());

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool mcp.write_file {}",
        ))
        .await
        .unwrap();

    // Fail closed: the response surfaces the scope block, not a tool result.
    let body = text(resp);
    assert!(
        body.contains("outside the egress scope") || body.contains("scope"),
        "expected a scope-denied message, got: {body}"
    );

    // Audit row: a blocked entry was written, naming the tool.
    let entries = audit_trail.entries.lock().unwrap();
    assert!(
        entries.iter().any(|e| {
            e.request.contains("mcp:fs:write_file")
                && e.decision.contains("scope")
                && matches!(e.outcome, audit::AuditOutcome::Failure)
        }),
        "expected a scope-denied audit row, got: {:?}",
        entries
            .iter()
            .map(|e| (&e.request, &e.decision))
            .collect::<Vec<_>>()
    );
}
