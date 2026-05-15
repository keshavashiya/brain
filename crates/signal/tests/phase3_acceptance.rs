//! v1.0.0 Phase 3 acceptance test.
//!
//! Drives the full capability-kernel flow through the public
//! [`SignalProcessor::process`] surface:
//!
//! `/tool mcp.echo {"text":"hi"}` → classifier emits
//! `Intent::ToolCall(IntentToken)` → `DefaultIntentRouter::resolve` →
//! `ToolRoute::Mcp` → `MCPHost::call` → rendered response.
//!
//! Uses a tiny in-process `EchoMcpHost` so the test stays self-contained;
//! the real rmcp transport is already covered by
//! `crates/mcphost/tests/http_round_trip.rs`.

use std::sync::Arc;

use async_trait::async_trait;
use brain_core::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalResponse, SignalSource};
use chrono::Utc;
use mcphost::{
    CallOutcome, MCPHost, McpHostError, MountedServer, ServerConfig, ServerStatus, ToolDescriptor,
};
use tokio::sync::RwLock;

/// In-process [`MCPHost`] that exposes a single `echo` tool. Used as a
/// stand-in for a real rmcp transport so the acceptance suite can run
/// without spawning a child process or an HTTP server.
#[derive(Default)]
struct EchoMcpHost {
    mounted: RwLock<Option<MountedServer>>,
    registry: Option<Arc<dyn intent::ToolRegistry>>,
}

impl EchoMcpHost {
    fn with_registry(registry: Arc<dyn intent::ToolRegistry>) -> Self {
        Self {
            mounted: RwLock::new(None),
            registry: Some(registry),
        }
    }
}

#[async_trait]
impl MCPHost for EchoMcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        let tool = ToolDescriptor {
            server: name.clone(),
            name: "echo".into(),
            description: Some("Echo back the provided text".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"],
            }),
        };
        let record = MountedServer {
            name: name.clone(),
            config: cfg,
            mounted_at: Utc::now(),
            info: None,
            tools: vec![tool.clone()],
        };
        *self.mounted.write().await = Some(record);
        if let Some(reg) = &self.registry {
            reg.register(intent::ToolDescriptor {
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
                embedding: None,
            })
            .await
            .unwrap();
        }
        Ok(())
    }

    async fn unmount(&self, name: &str) -> Result<(), McpHostError> {
        let mut guard = self.mounted.write().await;
        if guard.as_ref().map(|m| m.name.as_str()) != Some(name) {
            return Err(McpHostError::NotMounted(name.to_string()));
        }
        *guard = None;
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
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let guard = self.mounted.read().await;
        let Some(record) = guard.as_ref() else {
            return Err(McpHostError::NotMounted(server.to_string()));
        };
        if record.name != server {
            return Err(McpHostError::NotMounted(server.to_string()));
        }
        if tool != "echo" {
            return Err(McpHostError::Transport(format!("unknown tool '{tool}'")));
        }
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        Ok(CallOutcome {
            server: server.to_string(),
            tool: tool.to_string(),
            is_error: false,
            content: serde_json::json!([{ "type": "text", "text": format!("echo: {text}") }]),
            elapsed_ms: 1,
        })
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
async fn tool_slash_resolves_through_router_to_mcp_call() {
    let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
    let router: Arc<dyn intent::IntentRouter> =
        Arc::new(intent::DefaultIntentRouter::new(registry.clone()));
    let host: Arc<dyn MCPHost> = Arc::new(EchoMcpHost::with_registry(registry.clone()));
    let processor = make_processor()
        .await
        .with_tool_registry(registry.clone())
        .with_intent_router(router)
        .with_mcp_host(host);

    // Mount the echo server via the existing /mcp-mount slash. EchoMcpHost
    // registers its tool in the shared registry as part of `mount`.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-mount echo stdio echo-binary",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Mounted MCP server 'echo'"));
    assert_eq!(registry.list().await.len(), 1);

    // /tool mcp.echo {"text":"hello"} — full Phase 3 path:
    // classifier → ToolCall → router.resolve → ToolRoute::Mcp → host.call.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            r#"/tool mcp.echo {"text":"hello"}"#,
        ))
        .await
        .unwrap();
    let t = text(resp);
    assert!(t.contains("mcp:echo:echo"), "{t}");
    assert!(t.contains("(ok"), "{t}");
    assert!(t.contains("echo: hello"), "{t}");
}

#[tokio::test]
async fn tool_slash_with_no_candidate_renders_human_confirm() {
    let registry: Arc<dyn intent::ToolRegistry> = Arc::new(intent::InMemoryToolRegistry::new());
    let router: Arc<dyn intent::IntentRouter> =
        Arc::new(intent::DefaultIntentRouter::new(registry.clone()));
    let processor = make_processor()
        .await
        .with_tool_registry(registry)
        .with_intent_router(router);

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool memory.store",
        ))
        .await
        .unwrap();
    let t = text(resp);
    assert!(t.contains("memory.store"), "{t}");
    assert!(t.contains("No tool registered"), "{t}");
}

#[tokio::test]
async fn tool_slash_without_router_uses_placeholder() {
    let processor = make_processor().await;
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/tool fs.read",
        ))
        .await
        .unwrap();
    let t = text(resp);
    assert!(t.contains("Capability router not configured"), "{t}");
    assert!(t.contains("fs.read"), "{t}");
}
