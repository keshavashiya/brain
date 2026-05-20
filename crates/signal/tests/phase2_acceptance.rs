//! v1.0.0 Phase 2 acceptance test.
//!
//! Wires both the Terminal Bridge (`crates/adapters/terminal`) and the MCP
//! host (`crates/mcphost`) into a single [`SignalProcessor`] and drives the
//! end-to-end slash flows through the public `process()` entry point. This
//! is the canonical "does Phase 2 still work" check — every PR in the
//! Phase-3+ stack must keep this green.

#![cfg(unix)]

use std::sync::Arc;

use brain::BrainConfig;
use brainos_signal::{ResponseContent, Signal, SignalProcessor, SignalSource};
use mcphost::{InMemoryMcpHost, MCPHost};
use terminal::TerminalBridge;

async fn make_processor() -> SignalProcessor {
    let temp = tempfile::tempdir().unwrap();
    let mut config = BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let processor = SignalProcessor::new(config).await.unwrap();
    std::mem::forget(temp);
    processor
}

fn text(resp: brainos_signal::SignalResponse) -> String {
    match resp.response {
        ResponseContent::Text(t) => t,
        other => panic!("expected Text, got {other:?}"),
    }
}

#[tokio::test]
async fn unwired_mcp_intents_report_not_configured() {
    let processor = make_processor().await;
    for cmd in ["/mcp-list", "/mcp-mount fs stdio mcp-fs", "/mcp-unmount fs"] {
        let resp = processor
            .process(Signal::new(SignalSource::Cli, "cli", "user", cmd))
            .await
            .unwrap();
        let t = text(resp);
        assert!(
            t.contains("not configured"),
            "{cmd}: expected 'not configured', got: {t}"
        );
    }
}

#[tokio::test]
async fn mcp_mount_list_unmount_round_trip() {
    let host: Arc<dyn MCPHost> = Arc::new(InMemoryMcpHost::new());
    let processor = make_processor().await.with_mcp_host(host.clone());

    // List when empty.
    let resp = processor
        .process(Signal::new(SignalSource::Cli, "cli", "user", "/mcp-list"))
        .await
        .unwrap();
    let listed = text(resp);
    assert!(listed.contains("No mounted MCP servers"), "got: {listed}");

    // Mount one stdio server.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-mount fs stdio mcp-fs --root /tmp",
        ))
        .await
        .unwrap();
    let mounted_msg = text(resp);
    assert!(
        mounted_msg.contains("Mounted MCP server 'fs'"),
        "got: {mounted_msg}"
    );
    assert_eq!(host.list_servers().await.len(), 1);

    // List shows it.
    let resp = processor
        .process(Signal::new(SignalSource::Cli, "cli", "user", "/mcp-list"))
        .await
        .unwrap();
    let listed = text(resp);
    assert!(listed.contains("1 mounted MCP server"), "got: {listed}");
    assert!(listed.contains("fs"), "got: {listed}");

    // Unmount.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-unmount fs",
        ))
        .await
        .unwrap();
    let unmounted = text(resp);
    assert!(
        unmounted.contains("Unmounted MCP server 'fs'"),
        "got: {unmounted}"
    );
    assert_eq!(host.list_servers().await.len(), 0);
}

#[tokio::test]
async fn mcp_mount_with_unknown_transport_reports_error() {
    let host: Arc<dyn MCPHost> = Arc::new(InMemoryMcpHost::new());
    let processor = make_processor().await.with_mcp_host(host);

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-mount weird telnet host:9999",
        ))
        .await
        .unwrap();
    let t = text(resp);
    assert!(t.contains("unknown transport 'telnet'"), "got: {t}");
}

#[tokio::test]
async fn terminal_and_mcp_share_one_processor() {
    // The Phase 2 acceptance: a single `SignalProcessor` wires both motor-
    // cortex backends. Drive a terminal session lifecycle and an MCP mount
    // through the same instance.
    let bridge = Arc::new(TerminalBridge::new());
    let host: Arc<dyn MCPHost> = Arc::new(InMemoryMcpHost::new());
    let processor = make_processor()
        .await
        .with_terminal_bridge(bridge.clone())
        .with_mcp_host(host.clone());

    // Terminal: open → list → close.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/terminal-open /bin/sh -c sleep_for_test",
        ))
        .await
        .unwrap();
    let open_msg = text(resp);
    assert!(
        open_msg.starts_with("Opened terminal session "),
        "{open_msg}"
    );
    let session_id = open_msg
        .strip_prefix("Opened terminal session ")
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap()
        .to_string();
    assert_eq!(bridge.sessions().len().await, 1);

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/terminal-list",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains(&session_id));

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            format!("/terminal-close {session_id}"),
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Closed terminal session"));
    assert_eq!(bridge.sessions().len().await, 0);

    // MCP: mount → list → unmount, in the same processor.
    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-mount fs stdio mcp-fs",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Mounted MCP server 'fs'"));

    let resp = processor
        .process(Signal::new(SignalSource::Cli, "cli", "user", "/mcp-list"))
        .await
        .unwrap();
    assert!(text(resp).contains("1 mounted MCP server"));

    let resp = processor
        .process(Signal::new(
            SignalSource::Cli,
            "cli",
            "user",
            "/mcp-unmount fs",
        ))
        .await
        .unwrap();
    assert!(text(resp).contains("Unmounted MCP server 'fs'"));
}
