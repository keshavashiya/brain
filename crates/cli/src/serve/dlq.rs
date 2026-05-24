//! Dead-letter queue drain task for `cmd_serve`.
//!
//! Exhausted MCP retries land in the DLQ via the `ResilientMcpHost`
//! decorator. Once per hour, walk the most-recent batch and try to
//! replay each entry against the live MCP host. Entries whose replay
//! returns a non-error outcome are purged; everything else stays in
//! the queue for the next cycle, so a flaky-but-eventually-working
//! tool drains on its own. Skips the cycle if either the DLQ or the
//! MCP host isn't wired — both are required to make a replay decision.

use std::sync::Arc;

/// Spawn the DLQ drain loop if both the queue and an MCP host are
/// wired. No-op when either is unset — caller doesn't need to check.
pub(super) fn spawn_dlq_drain(
    processor: Arc<signal::SignalProcessor>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    if processor.dlq().is_none() || processor.mcp_host().is_none() {
        return;
    }
    let p = processor.clone();
    set.spawn(async move {
        let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(3600));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            let (Some(dlq), Some(mcp_host)) = (p.dlq().cloned(), p.mcp_host().cloned()) else {
                continue;
            };
            let (replayed, still_failing) = drain_dlq_batch(&dlq, &mcp_host, 50).await;
            tracing::info!(
                replayed = replayed.len(),
                still_failing = still_failing.len(),
                "DLQ drain cycle complete"
            );
        }
    });
    tracing::info!("DLQ drain scheduled (every 60min, batch=50)");
}

/// Drain up to `limit` entries from the DLQ by replaying each one
/// through the MCP host. Returns `(replayed_ok_ids, still_failing_ids)`
/// after the batch — successful entries are purged from the queue.
///
/// An entry replays when:
/// - its `tool_id` parses as `mcp:{server}:{tool}`, and
/// - `mcp_host.call(server, tool, args)` returns `Ok(outcome)` with
///   `is_error == false`.
///
/// Anything else (parse failure, transport error, `is_error: true`)
/// leaves the entry in the queue for the next cycle.
async fn drain_dlq_batch(
    dlq: &Arc<dyn ::resilience::DeadLetterQueue>,
    mcp_host: &Arc<dyn mcphost::MCPHost>,
    limit: usize,
) -> (Vec<String>, Vec<String>) {
    let entries = match dlq.list_recent(limit).await {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "DLQ list_recent failed");
            return (Vec::new(), Vec::new());
        }
    };

    let mut succeeded = Vec::new();
    let mut still_failing = Vec::new();
    for entry in entries {
        let parts: Vec<&str> = entry.tool_id.splitn(3, ':').collect();
        if parts.len() != 3 || parts[0] != "mcp" {
            tracing::debug!(tool_id = %entry.tool_id, "skipping non-mcp DLQ entry");
            still_failing.push(entry.id);
            continue;
        }
        let (server, tool) = (parts[1], parts[2]);
        let args: serde_json::Value =
            serde_json::from_str(&entry.request_json).unwrap_or(serde_json::Value::Null);

        match mcp_host.call(server, tool, args).await {
            Ok(outcome) if !outcome.is_error => succeeded.push(entry.id),
            Ok(_) | Err(_) => still_failing.push(entry.id),
        }
    }

    if !succeeded.is_empty() {
        match dlq.purge(&succeeded).await {
            Ok(n) => tracing::debug!(purged = n, "DLQ purge complete"),
            Err(e) => tracing::warn!(error = %e, "DLQ purge failed; entries stay in queue"),
        }
    }

    (succeeded, still_failing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub MCP host whose `call` returns a fixed outcome. Only `call`
    /// is exercised by `drain_dlq_batch`; the other trait methods are
    /// no-ops so the stub stays minimal.
    struct StubMcpHost {
        respond_with_error: bool,
    }

    #[async_trait::async_trait]
    impl mcphost::MCPHost for StubMcpHost {
        async fn mount(
            &self,
            _name: String,
            _cfg: mcphost::ServerConfig,
        ) -> Result<(), mcphost::McpHostError> {
            Ok(())
        }
        async fn unmount(&self, _name: &str) -> Result<(), mcphost::McpHostError> {
            Ok(())
        }
        async fn list_servers(&self) -> Vec<mcphost::ServerStatus> {
            Vec::new()
        }
        async fn list_all_tools(&self) -> Vec<mcphost::ToolDescriptor> {
            Vec::new()
        }
        async fn call(
            &self,
            server: &str,
            tool: &str,
            _args: serde_json::Value,
        ) -> Result<mcphost::CallOutcome, mcphost::McpHostError> {
            Ok(mcphost::CallOutcome {
                server: server.to_string(),
                tool: tool.to_string(),
                is_error: self.respond_with_error,
                content: serde_json::Value::Null,
                elapsed_ms: 0,
            })
        }
    }

    fn seed_entry(tool_id: &str, payload: &str) -> ::resilience::DlqEntry {
        ::resilience::DlqEntry {
            id: uuid::Uuid::new_v4().to_string(),
            tool_id: tool_id.to_string(),
            request_json: payload.to_string(),
            error_message: "exhausted".to_string(),
            attempts: 3,
            dlq_at: chrono::Utc::now(),
        }
    }

    // Successful replay must purge the entry from the queue. A second
    // drain finds nothing — the queue is empty.
    #[tokio::test]
    async fn drain_dlq_replays_and_purges_successful_entries() {
        let dlq: Arc<dyn ::resilience::DeadLetterQueue> =
            Arc::new(::resilience::InMemoryDlq::new());
        let host: Arc<dyn mcphost::MCPHost> = Arc::new(StubMcpHost {
            respond_with_error: false,
        });
        dlq.enqueue(seed_entry("mcp:srv:echo", r#"{"hello":1}"#))
            .await
            .unwrap();
        assert_eq!(dlq.len().await.unwrap(), 1);

        let (ok, fail) = drain_dlq_batch(&dlq, &host, 50).await;
        assert_eq!(ok.len(), 1);
        assert_eq!(fail.len(), 0);
        assert_eq!(dlq.len().await.unwrap(), 0);
    }

    // `is_error: true` keeps the entry around — only outcomes the
    // tool itself deems successful should be retired.
    #[tokio::test]
    async fn drain_dlq_keeps_entries_when_replay_returns_is_error() {
        let dlq: Arc<dyn ::resilience::DeadLetterQueue> =
            Arc::new(::resilience::InMemoryDlq::new());
        let host: Arc<dyn mcphost::MCPHost> = Arc::new(StubMcpHost {
            respond_with_error: true,
        });
        dlq.enqueue(seed_entry("mcp:srv:flake", "null"))
            .await
            .unwrap();

        let (ok, fail) = drain_dlq_batch(&dlq, &host, 50).await;
        assert!(ok.is_empty(), "no successful replays expected");
        assert_eq!(fail.len(), 1);
        assert_eq!(
            dlq.len().await.unwrap(),
            1,
            "is_error entries stay in queue"
        );
    }

    // A non-`mcp:` tool id can't be replayed via the MCP host — the
    // drainer leaves it in place rather than dropping it silently.
    #[tokio::test]
    async fn drain_dlq_leaves_non_mcp_entries_alone() {
        let dlq: Arc<dyn ::resilience::DeadLetterQueue> =
            Arc::new(::resilience::InMemoryDlq::new());
        let host: Arc<dyn mcphost::MCPHost> = Arc::new(StubMcpHost {
            respond_with_error: false,
        });
        dlq.enqueue(seed_entry("native:scheduler:tick", "null"))
            .await
            .unwrap();

        let (ok, fail) = drain_dlq_batch(&dlq, &host, 50).await;
        assert!(ok.is_empty());
        assert_eq!(fail.len(), 1);
        assert_eq!(dlq.len().await.unwrap(), 1);
    }
}
