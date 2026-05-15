//! Round-trip integration test: spin up an in-process rmcp Streamable HTTP
//! server, mount it via [`RmcpHost`], and exercise `list_servers` /
//! `list_all_tools` / `call` / `refresh_tools` end-to-end.

use std::sync::Arc;
use std::time::Duration;

use brainos_mcphost::{MCPHost, RmcpHost, ServerConfig};
use observe::{BrainEvent, BroadcastObserver, Observer};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    },
    ServerHandler,
};
use tokio_util::sync::CancellationToken;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct EchoArgs {
    text: String,
}

#[derive(Debug, Clone)]
struct EchoServer {
    tool_router: ToolRouter<Self>,
}

impl EchoServer {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl EchoServer {
    #[tool(description = "Echo back the provided text")]
    fn echo(&self, Parameters(EchoArgs { text }): Parameters<EchoArgs>) -> String {
        format!("echo: {text}")
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EchoServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}

struct ServerHandle {
    url: String,
    cancel: CancellationToken,
}

async fn spawn_server() -> ServerHandle {
    let ct = CancellationToken::new();
    let service: StreamableHttpService<EchoServer, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(EchoServer::new()),
            Default::default(),
            StreamableHttpServerConfig::default()
                .with_sse_keep_alive(None)
                .with_cancellation_token(ct.child_token()),
        );

    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let ct_serve = ct.clone();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move { ct_serve.cancelled_owned().await })
            .await;
    });

    // The listener is bound but the axum runtime needs a tick to accept.
    tokio::time::sleep(Duration::from_millis(20)).await;

    ServerHandle {
        url: format!("http://{addr}/mcp"),
        cancel: ct,
    }
}

#[tokio::test]
async fn http_mount_round_trip() {
    let server = spawn_server().await;
    let host = RmcpHost::new();

    host.mount(
        "echo".into(),
        ServerConfig::StreamableHttp {
            url: server.url.clone(),
            oauth: None,
        },
    )
    .await
    .expect("mount should succeed against in-process MCP server");

    let servers = host.list_servers().await;
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].name, "echo");
    assert_eq!(servers[0].tool_count, 1);

    let tools = host.list_all_tools().await;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let outcome = host
        .call("echo", "echo", serde_json::json!({"text": "hello"}))
        .await
        .expect("tool call must succeed");
    assert!(!outcome.is_error);
    assert!(outcome.content.to_string().contains("echo: hello"));

    let changed = host
        .refresh_tools("echo")
        .await
        .expect("refresh should succeed");
    assert!(!changed, "tool catalog must be stable across refreshes");

    host.unmount("echo").await.expect("unmount must succeed");
    assert!(host.list_servers().await.is_empty());

    server.cancel.cancel();
}

#[tokio::test]
async fn refresh_with_no_mount_errors() {
    let host = RmcpHost::new();
    let err = host
        .refresh_tools("ghost")
        .await
        .expect_err("refresh on missing mount must fail");
    assert!(
        matches!(err, brainos_mcphost::McpHostError::NotMounted(_)),
        "unexpected: {err:?}"
    );
}

#[tokio::test]
async fn rug_pull_does_not_fire_on_unchanged_catalog() {
    // Observer captures BrainEvents. After a refresh against an unchanged
    // catalog, no `Error` event with source = "mcphost" should appear.
    let server = spawn_server().await;
    let observer = BroadcastObserver::new();
    let mut rx = observer.subscribe();

    let host = RmcpHost::new().with_observer(observer.clone() as Arc<dyn Observer>);
    host.mount(
        "echo".into(),
        ServerConfig::StreamableHttp {
            url: server.url.clone(),
            oauth: None,
        },
    )
    .await
    .unwrap();

    let changed = host.refresh_tools("echo").await.unwrap();
    assert!(!changed);

    // Drain with a short timeout — no error events must be pending.
    let drained = tokio::time::timeout(Duration::from_millis(50), async { rx.recv().await }).await;
    if let Ok(Ok(BrainEvent::Error { source, .. })) = drained {
        assert_ne!(
            source, "mcphost",
            "no mcphost error should be published on a clean refresh"
        );
    }

    host.unmount("echo").await.unwrap();
    server.cancel.cancel();
}
