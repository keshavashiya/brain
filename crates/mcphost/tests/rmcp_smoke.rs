//! Smoke tests for the `rmcp`-backed [`brainos_mcphost::RmcpHost`].
//!
//! End-to-end coverage against a real MCP server is intentionally not in
//! the unit test suite — that requires either a sibling Node/Python
//! binary (flaky on CI) or a hand-rolled fake protocol server. These
//! tests cover the error paths around the rmcp surface; full
//! mount → list_tools → call coverage belongs in an acceptance suite
//! that owns the test server lifecycle.

use brainos_mcphost::{stdio_cfg, MCPHost, McpHostError, RmcpHost};

#[tokio::test]
async fn mount_nonexistent_command_errors_cleanly() {
    let host = RmcpHost::new();
    // Pick a command that almost certainly isn't on PATH. The exact
    // failure mode (spawn fail vs initialize fail) depends on the OS:
    // accept either Transport or Initialize.
    let err = host
        .mount(
            "ghost".into(),
            stdio_cfg("/var/empty/does-not-exist-7777", vec![]),
        )
        .await
        .expect_err("expected mount failure");
    assert!(
        matches!(
            err,
            McpHostError::Transport(_) | McpHostError::Initialize(_)
        ),
        "unexpected error variant: {err:?}"
    );
    // Failed mount must leave the registry empty.
    assert!(host.list_servers().await.is_empty());
}

#[tokio::test]
async fn unmount_unknown_server_returns_not_mounted() {
    let host = RmcpHost::new();
    let err = host
        .unmount("never-mounted")
        .await
        .expect_err("expected unmount failure");
    assert!(matches!(err, McpHostError::NotMounted(_)));
}

#[tokio::test]
async fn call_unknown_server_returns_not_mounted() {
    let host = RmcpHost::new();
    let err = host
        .call("nope", "any", serde_json::json!({}))
        .await
        .expect_err("expected call failure");
    assert!(matches!(err, McpHostError::NotMounted(_)));
}

#[tokio::test]
async fn http_mount_with_invalid_scheme_errors() {
    let host = RmcpHost::new();
    let err = host
        .mount(
            "bad".into(),
            brainos_mcphost::ServerConfig::StreamableHttp {
                url: "ftp://example.invalid/mcp".into(),
                oauth: None,
            },
        )
        .await
        .expect_err("non-HTTP scheme should reject");
    assert!(
        matches!(err, McpHostError::Transport(_)),
        "unexpected: {err:?}"
    );
}

#[tokio::test]
async fn http_mount_with_oauth_but_no_vault_errors() {
    let host = RmcpHost::new();
    let err = host
        .mount(
            "needs-auth".into(),
            brainos_mcphost::ServerConfig::StreamableHttp {
                url: "https://example.invalid/mcp".into(),
                oauth: Some(brainos_mcphost::OAuthConfig {
                    resource: "https://example.invalid/mcp".into(),
                    client_id: None,
                    authorization_server: None,
                }),
            },
        )
        .await
        .expect_err("OAuth mount without a vault should fail");
    assert!(matches!(err, McpHostError::Auth(_)), "unexpected: {err:?}");
}
