//! End-to-end test: per-tool circuit breakers wired through the public
//! [`SignalProcessor::process`] surface. Drives `/tool` against a failing
//! MCP host until the breaker trips, then verifies the next call falls
//! through to `HumanConfirm` because the router excludes Open tools.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use chrono::Utc;
use mcphost::{
    CallOutcome, MCPHost, McpHostError, MountedServer, ServerConfig, ServerStatus, ToolDescriptor,
};
use resilience::{BreakerConfig, BreakerRegistry};
use tokio::sync::RwLock;

/// `MCPHost` that registers one tool on mount and then *always* fails the
/// `call`. Used to drive the breaker through the closed → open transition
/// via real dispatch.
struct FailingMcpHost {
    mounted: RwLock<Option<MountedServer>>,
    registry: Arc<dyn intent::ToolRegistry>,
}

impl FailingMcpHost {
    fn new(registry: Arc<dyn intent::ToolRegistry>) -> Self {
        Self {
            mounted: RwLock::new(None),
            registry,
        }
    }
}

#[async_trait]
impl MCPHost for FailingMcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        let tool = ToolDescriptor {
            server: name.clone(),
            name: "broken".into(),
            description: Some("Always fails".into()),
            input_schema: serde_json::json!({ "type": "object" }),
        };
        *self.mounted.write().await = Some(MountedServer {
            name: name.clone(),
            config: cfg,
            mounted_at: Utc::now(),
            info: None,
            tools: vec![tool.clone()],
            scopes: mcphost::ServerScopes::default(),
        });
        self.registry
            .register(intent::ToolDescriptor {
                tool_id: format!("mcp:{name}:{}", tool.name),
                source: intent::ToolSource::McpServer {
                    server: name.clone(),
                },
                verb: intent::Verb::new("mcp", tool.name.clone()),
                description: tool.description.clone().unwrap_or_default(),
                input_schema: tool.input_schema.clone(),
                output_schema: None,
                capabilities: vec![],
                annotations: intent::ToolAnnotations::default(),
                usage: intent::ToolUsage::default(),
                embedding: None,
            })
            .await
            .unwrap();
        Ok(())
    }

    async fn unmount(&self, _name: &str) -> Result<(), McpHostError> {
        Ok(())
    }

    async fn list_servers(&self) -> Vec<ServerStatus> {
        self.mounted
            .read()
            .await
            .as_ref()
            .map(|m| {
                vec![ServerStatus {
                    name: m.name.clone(),
                    mounted_at: m.mounted_at,
                    tool_count: m.tools.len(),
                    info: m.info.clone(),
                    quarantined: false,
                    scopes: m.scopes.clone(),
                }]
            })
            .unwrap_or_default()
    }

    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        self.mounted
            .read()
            .await
            .as_ref()
            .map(|m| m.tools.clone())
            .unwrap_or_default()
    }

    async fn call(
        &self,
        server: &str,
        tool: &str,
        _args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        Err(McpHostError::Transport(format!(
            "induced failure on {server}:{tool}"
        )))
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
async fn repeated_failures_trip_the_breaker_and_router_excludes_the_tool() {
    let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
    let breakers = Arc::new(BreakerRegistry::new(BreakerConfig {
        failure_threshold: 2,
        open_duration: Duration::from_secs(10),
        half_open_required_successes: 1,
    }));
    let router: Arc<dyn intent::IntentRouter> = Arc::new(
        intent::DefaultIntentRouter::new(registry.clone())
            .with_breakers(breakers.clone() as Arc<dyn intent::BreakerCheck>),
    );
    let host: Arc<dyn MCPHost> = Arc::new(FailingMcpHost::new(registry.clone()));
    let processor = make_processor()
        .await
        .with_tool_registry(registry.clone())
        .with_intent_router(router)
        .with_breaker_registry(breakers.clone())
        .with_mcp_host(host);

    // Mount: FailingMcpHost registers `mcp:bad:broken` in the registry.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-mount bad stdio bad-binary",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Mounted MCP server 'bad'"));

    // First failure — breaker still closed, response carries the
    // transport error.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool mcp.broken",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("induced failure"));

    // Second failure — breaker trips to Open.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool mcp.broken",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("induced failure"));
    use intent::BreakerCheck;
    assert!(
        breakers.is_open("mcp:bad:broken").await,
        "breaker must be open after threshold failures",
    );

    // Third invocation — router excludes the Open tool, no other
    // candidates → HumanConfirm.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool mcp.broken",
        ))
        .await
        .unwrap();
    let t = text(resp);
    assert!(
        t.contains("No tool registered"),
        "expected HumanConfirm fall-through, got: {t}"
    );
}
