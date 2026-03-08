use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use grpcadapter::{
    memory_proto::{memory_service_server::MemoryService, SearchRequest},
    MemoryServiceImpl,
};
use serde_json::json;
use signal::SignalProcessor;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tonic::Request;

fn random_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[tokio::test]
#[ignore = "Requires local TCP listener permissions in the runtime environment"]
async fn test_cross_protocol_memory_parity() {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain_core::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();

    let processor = Arc::new(SignalProcessor::new(config).await.unwrap());

    let http_port = random_port();
    let ws_port = random_port();
    let mcp_port = random_port();

    let http_task = tokio::spawn(httpadapter::serve(
        processor.clone(),
        "127.0.0.1",
        http_port,
    ));
    let ws_task = tokio::spawn(wsadapter::serve(processor.clone(), "127.0.0.1", ws_port));
    let mcp_task = tokio::spawn(mcp::serve_http(processor.clone(), "127.0.0.1", mcp_port));

    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let client = reqwest::Client::new();

    // 1) Store via HTTP
    let http_store = client
        .post(format!("http://127.0.0.1:{http_port}/v1/signals"))
        .bearer_auth("demokey123")
        .json(&json!({
            "content": "Remember project uses bun",
            "namespace": "work",
            "source": "http",
            "channel": "http",
            "sender": "tester"
        }))
        .send()
        .await
        .unwrap();
    assert!(http_store.status().is_success());

    // 2) Store via WebSocket
    let (mut ws, _) = connect_async(format!("ws://127.0.0.1:{ws_port}"))
        .await
        .unwrap();
    ws.send(Message::Text(r#"{"api_key":"demokey123"}"#.into()))
        .await
        .unwrap();
    let auth = ws
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap()
        .to_string();
    assert!(auth.contains("\"status\":\"authenticated\""));
    ws.send(Message::Text(
        r#"{"source":"ws","sender":"wsclient","namespace":"work","content":"Remember platform uses cargo"}"#
            .into(),
    ))
    .await
    .unwrap();
    let ws_resp = ws
        .next()
        .await
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap()
        .to_string();
    assert!(ws_resp.contains("\"status\":\"Ok\""));

    // 3) Verify via gRPC adapter service
    let grpc = MemoryServiceImpl::new(processor.clone());
    let grpc_search = grpc
        .search(Request::new(SearchRequest {
            query: "uses".to_string(),
            top_k: 10,
            namespace: "work".to_string(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(
        !grpc_search.facts.is_empty(),
        "gRPC search should see facts stored by HTTP/WS"
    );

    // 4) Verify via MCP HTTP transport
    let mcp_search = client
        .post(format!("http://127.0.0.1:{mcp_port}/mcp"))
        .header("x-api-key", "demokey123")
        .json(&json!({
            "jsonrpc":"2.0",
            "id": 1,
            "method":"tools/call",
            "params":{
                "name":"memory_search",
                "arguments":{
                    "query":"uses",
                    "namespace":"work",
                    "top_k":10
                }
            }
        }))
        .send()
        .await
        .unwrap();
    assert!(mcp_search.status().is_success());
    let mcp_json: serde_json::Value = mcp_search.json().await.unwrap();
    let mcp_text = mcp_json["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_default();
    assert!(mcp_text.contains("uses"));

    // 5) Verify namespace parity via HTTP list endpoint
    let facts_resp = client
        .get(format!(
            "http://127.0.0.1:{http_port}/v1/memory/facts?namespace=work"
        ))
        .bearer_auth("demokey123")
        .send()
        .await
        .unwrap();
    assert!(facts_resp.status().is_success());
    let facts_json: serde_json::Value = facts_resp.json().await.unwrap();
    let facts = facts_json.as_array().unwrap();
    assert!(facts.len() >= 2, "expected at least 2 work facts");

    http_task.abort();
    ws_task.abort();
    mcp_task.abort();
}
