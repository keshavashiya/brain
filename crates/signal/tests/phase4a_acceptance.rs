//! Phase 4 Track A acceptance — drives a programmable `MCPHost` through
//! the full resilience stack (`ResilientMcpHost`) and verifies each
//! layer fires under the conditions it's responsible for.
//!
//! Each test owns its own decorator stack so behaviors don't leak
//! across runs.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use mcphost::{
    CallOutcome, MCPHost, McpHostError, ResilienceConfig, ResilientMcpHost, ServerConfig,
    ServerStatus, ToolDescriptor,
};
use resilience::{
    BreakerConfig, BreakerRegistry, DeadLetterQueue, InMemoryDlq, LoopDetector, LoopDetectorConfig,
    RetryConfig,
};
use tokio::sync::Mutex;

/// `MCPHost` that returns failures for the first N calls per tool, then
/// succeeds. Used to drive the retry path through a transient-failure
/// recovery without timing-sensitive plumbing.
struct ScriptedMcpHost {
    fail_first_n: AtomicU32,
    calls: AtomicU32,
}

impl ScriptedMcpHost {
    fn new(fail_first_n: u32) -> Self {
        Self {
            fail_first_n: AtomicU32::new(fail_first_n),
            calls: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl MCPHost for ScriptedMcpHost {
    async fn mount(&self, _name: String, _cfg: ServerConfig) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn unmount(&self, _name: &str) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn list_servers(&self) -> Vec<ServerStatus> {
        Vec::new()
    }
    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        Vec::new()
    }
    async fn call(
        &self,
        server: &str,
        tool: &str,
        _args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_first_n.load(Ordering::SeqCst) > 0 {
            self.fail_first_n.fetch_sub(1, Ordering::SeqCst);
            return Err(McpHostError::Transport("scripted transient".into()));
        }
        Ok(CallOutcome {
            server: server.to_string(),
            tool: tool.to_string(),
            is_error: false,
            content: serde_json::json!({"ok": true}),
            elapsed_ms: 1,
        })
    }
}

/// Always-fails host for exhaustion and breaker tests.
struct AlwaysFailingMcpHost {
    calls: AtomicU32,
}

impl AlwaysFailingMcpHost {
    fn new() -> Self {
        Self {
            calls: AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl MCPHost for AlwaysFailingMcpHost {
    async fn mount(&self, _name: String, _cfg: ServerConfig) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn unmount(&self, _name: &str) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn list_servers(&self) -> Vec<ServerStatus> {
        Vec::new()
    }
    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        Vec::new()
    }
    async fn call(
        &self,
        _server: &str,
        _tool: &str,
        _args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(McpHostError::Transport("always fails".into()))
    }
}

/// Slow host for the timeout test — waits longer than the deadline.
struct SlowMcpHost {
    delay: Duration,
    calls: AtomicU32,
}

#[async_trait]
impl MCPHost for SlowMcpHost {
    async fn mount(&self, _: String, _: ServerConfig) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn unmount(&self, _: &str) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn list_servers(&self) -> Vec<ServerStatus> {
        Vec::new()
    }
    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        Vec::new()
    }
    async fn call(
        &self,
        _: &str,
        _: &str,
        _: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        Ok(CallOutcome {
            server: String::new(),
            tool: String::new(),
            is_error: false,
            content: serde_json::Value::Null,
            elapsed_ms: 0,
        })
    }
}

/// Records every call's args so the loop-detector test can assert
/// short-circuit before reaching this layer.
struct CountingMcpHost {
    calls: Mutex<Vec<serde_json::Value>>,
}

impl CountingMcpHost {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl MCPHost for CountingMcpHost {
    async fn mount(&self, _: String, _: ServerConfig) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn unmount(&self, _: &str) -> Result<(), McpHostError> {
        Ok(())
    }
    async fn list_servers(&self) -> Vec<ServerStatus> {
        Vec::new()
    }
    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        Vec::new()
    }
    async fn call(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        self.calls.lock().await.push(args);
        Ok(CallOutcome {
            server: server.to_string(),
            tool: tool.to_string(),
            is_error: false,
            content: serde_json::Value::Null,
            elapsed_ms: 0,
        })
    }
}

fn retry_fast(max_attempts: u32) -> RetryConfig {
    RetryConfig {
        max_attempts,
        base_delay: Duration::from_millis(1),
        max_delay: Duration::from_millis(5),
        jitter_factor: 0.0,
    }
}

#[tokio::test]
async fn retry_recovers_from_transient_failures() {
    let inner = Arc::new(ScriptedMcpHost::new(2));
    let host =
        ResilientMcpHost::new(inner.clone() as Arc<dyn MCPHost>).with_config(ResilienceConfig {
            timeout: Some(Duration::from_secs(2)),
            retry: Some(retry_fast(5)),
        });
    let out = host
        .call("echo", "ping", serde_json::json!({}))
        .await
        .expect("retry should recover");
    assert!(!out.is_error);
    // Three attempts: two scripted failures + the successful retry.
    assert_eq!(inner.calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn breaker_trips_after_repeated_failures() {
    let inner = Arc::new(AlwaysFailingMcpHost::new());
    let breakers = Arc::new(BreakerRegistry::new(BreakerConfig {
        failure_threshold: 2,
        open_duration: Duration::from_secs(30),
        half_open_required_successes: 1,
    }));
    // Retry off so each call counts as exactly one breaker failure.
    let host =
        ResilientMcpHost::new(inner.clone() as Arc<dyn MCPHost>).with_breakers(breakers.clone());

    // Two failures trip the breaker.
    assert!(host.call("e", "p", serde_json::json!({})).await.is_err());
    assert!(host.call("e", "p", serde_json::json!({})).await.is_err());
    // Now the breaker should fast-fail without invoking the inner host.
    let before = inner.calls.load(Ordering::SeqCst);
    let err = host
        .call("e", "p", serde_json::json!({}))
        .await
        .unwrap_err();
    assert_eq!(
        inner.calls.load(Ordering::SeqCst),
        before,
        "breaker should short-circuit before inner call"
    );
    assert!(format!("{err}").contains("breaker open"));
}

#[tokio::test]
async fn dlq_captures_exhausted_failure() {
    let inner = Arc::new(AlwaysFailingMcpHost::new());
    let dlq: Arc<dyn DeadLetterQueue> = Arc::new(InMemoryDlq::new());
    let host = ResilientMcpHost::new(inner as Arc<dyn MCPHost>)
        .with_config(ResilienceConfig {
            timeout: Some(Duration::from_secs(2)),
            retry: Some(retry_fast(3)),
        })
        .with_dlq(dlq.clone());

    let err = host
        .call("e", "ping", serde_json::json!({"x": 1}))
        .await
        .unwrap_err();
    assert!(matches!(err, McpHostError::Transport(_)));

    let entries = dlq.list_recent(10).await.unwrap();
    assert_eq!(entries.len(), 1);
    let entry = &entries[0];
    assert_eq!(entry.tool_id, "mcp:e:ping");
    assert_eq!(entry.attempts, 3);
    assert!(entry.error_message.contains("exhausted"));
    assert_eq!(entry.request_json, r#"{"x":1}"#);
}

#[tokio::test]
async fn loop_detector_short_circuits_repeats() {
    let inner = Arc::new(CountingMcpHost::new());
    let detector = Arc::new(LoopDetector::new(LoopDetectorConfig {
        window: 8,
        threshold: 2,
    }));
    let host = ResilientMcpHost::new(inner.clone() as Arc<dyn MCPHost>)
        .with_loop_detector(detector)
        .with_principal("acceptance");

    let args = serde_json::json!({"x": 1});
    // First two pass under threshold (count 1, 2 — not yet >2).
    host.call("e", "p", args.clone()).await.unwrap();
    host.call("e", "p", args.clone()).await.unwrap();
    // Third trips (count 3 > threshold 2).
    let err = host.call("e", "p", args.clone()).await.unwrap_err();
    assert!(format!("{err}").contains("loop detected"));
    // Inner host was only hit twice — the loop short-circuited before
    // the third call reached the transport.
    assert_eq!(inner.calls.lock().await.len(), 2);
}

#[tokio::test]
async fn timeout_aborts_slow_call_and_lands_in_dlq() {
    let inner = Arc::new(SlowMcpHost {
        delay: Duration::from_millis(200),
        calls: AtomicU32::new(0),
    });
    let dlq: Arc<dyn DeadLetterQueue> = Arc::new(InMemoryDlq::new());
    let host = ResilientMcpHost::new(inner as Arc<dyn MCPHost>)
        .with_config(ResilienceConfig {
            timeout: Some(Duration::from_millis(20)),
            retry: None,
        })
        .with_dlq(dlq.clone());

    let err = host.call("e", "slow", serde_json::json!({})).await;
    assert!(err.is_err(), "expected timeout");
    let entries = dlq.list_recent(5).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert!(entries[0].error_message.contains("timeout"));
}

#[tokio::test]
async fn full_stack_composes_layers_in_documented_order() {
    // Full stack: loop detector + breaker + retry + DLQ. Exercise the
    // happy path through every layer (one transient failure then
    // success), then assert DLQ stays empty.
    let inner = Arc::new(ScriptedMcpHost::new(1));
    let breakers = Arc::new(BreakerRegistry::new(BreakerConfig::default()));
    let detector = Arc::new(LoopDetector::new(LoopDetectorConfig::default()));
    let dlq: Arc<dyn DeadLetterQueue> = Arc::new(InMemoryDlq::new());

    let host = ResilientMcpHost::new(inner.clone() as Arc<dyn MCPHost>)
        .with_config(ResilienceConfig {
            timeout: Some(Duration::from_secs(2)),
            retry: Some(retry_fast(4)),
        })
        .with_breakers(breakers.clone())
        .with_loop_detector(detector)
        .with_dlq(dlq.clone())
        .with_principal("acceptance");

    let out = host
        .call("e", "p", serde_json::json!({"x": 1}))
        .await
        .expect("retry should recover");
    assert!(!out.is_error);
    assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    assert_eq!(dlq.len().await.unwrap(), 0, "happy path must not DLQ");
    // After one success the breaker remains closed.
    use intent::BreakerCheck;
    assert!(!breakers.is_open("mcp:e:p").await);
}
