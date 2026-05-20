//! gRPC adapter unit tests — exercise the service impls in-process without a
//! real listener.

use std::sync::Arc;

use tonic::Request;
use uuid::Uuid;

use signal::SignalSource;

use crate::agent_proto::{
    agent_service_server::AgentService, ConnectRequest, ReceiveRequest,
    SignalRequest as AgentSignalRequest,
};
use crate::helpers::{non_empty, response_to_string};
use crate::memory_proto::{
    memory_service_server::MemoryService, GetFactsRequest, SearchRequest,
    SignalRequest as MemorySignalRequest,
};
use crate::state::{AgentServiceImpl, MemoryServiceImpl};

// Helper: build a SignalProcessor backed by a temp directory.
async fn make_processor() -> Arc<signal::SignalProcessor> {
    let temp = tempfile::tempdir().unwrap();
    let mut config = brain::BrainConfig::default();
    config.brain.data_dir = temp.path().to_str().unwrap().to_string();
    let proc = signal::SignalProcessor::new(config).await.unwrap();
    // Keep temp alive by leaking (fine for tests)
    std::mem::forget(temp);
    Arc::new(proc)
}

#[tokio::test]
async fn test_memory_service_get_facts_empty() {
    let processor = make_processor().await;
    let svc = MemoryServiceImpl::new(processor);

    let req = Request::new(GetFactsRequest {
        subject: String::new(),
        namespace: String::new(),
    });
    let resp = svc.get_facts(req).await.unwrap();
    assert!(resp.into_inner().facts.is_empty());
}

#[tokio::test]
async fn test_memory_service_get_facts_with_subject_filter() {
    let processor = make_processor().await;
    let svc = MemoryServiceImpl::new(processor);

    let req = Request::new(GetFactsRequest {
        subject: "rust".to_string(),
        namespace: String::new(),
    });
    let resp = svc.get_facts(req).await.unwrap();
    // No facts stored yet — result should be empty
    assert!(resp.into_inner().facts.is_empty());
}

#[tokio::test]
async fn test_memory_service_search_empty() {
    let processor = make_processor().await;
    let svc = MemoryServiceImpl::new(processor);

    let req = Request::new(SearchRequest {
        query: "what is Rust".to_string(),
        top_k: 5,
        namespace: String::new(),
    });
    let resp = svc.search(req).await.unwrap();
    // No facts in empty store
    assert!(resp.into_inner().facts.is_empty());
}

#[tokio::test]
async fn test_memory_service_search_default_top_k() {
    let processor = make_processor().await;
    let svc = MemoryServiceImpl::new(processor);

    // top_k = 0 should default to 10 internally without panicking
    let req = Request::new(SearchRequest {
        query: "test".to_string(),
        top_k: 0,
        namespace: String::new(),
    });
    let resp = svc.search(req).await.unwrap();
    assert!(resp.into_inner().facts.is_empty());
}

#[tokio::test]
async fn test_agent_service_connect() {
    let processor = make_processor().await;
    let svc = AgentServiceImpl::new(processor);

    let req = Request::new(ConnectRequest {
        agent_id: "testagent".to_string(),
        agent_type: "assistant".to_string(),
    });
    let resp = svc.connect(req).await.unwrap();
    let inner = resp.into_inner();
    assert!(inner.accepted);
    assert!(!inner.session_id.is_empty());
    // session_id should be a valid UUID
    assert!(Uuid::parse_str(&inner.session_id).is_ok());
}

#[tokio::test]
async fn test_agent_service_send_signal() {
    let processor = make_processor().await;
    let svc = AgentServiceImpl::new(processor);

    let req = Request::new(AgentSignalRequest {
        source: "grpc".to_string(),
        channel: "test".to_string(),
        sender: "testagent".to_string(),
        content: "Remember that Rust is fast".to_string(),
        metadata: std::collections::HashMap::new(),
        namespace: String::new(),
        agent: String::new(),
        session_id: String::new(),
    });
    let resp = svc.send_signal(req).await.unwrap();
    let inner = resp.into_inner();
    // Signal was processed — should have a valid UUID and an "Ok" status
    assert!(!inner.signal_id.is_empty());
    assert!(Uuid::parse_str(&inner.signal_id).is_ok());
    assert_eq!(inner.status, "Ok");
}

#[tokio::test]
async fn test_memory_stream_signals() {
    let processor = make_processor().await;
    let svc = MemoryServiceImpl::new(processor);

    let req = Request::new(MemorySignalRequest {
        source: "grpc".to_string(),
        channel: "test".to_string(),
        sender: "testclient".to_string(),
        content: "Remember that Brain is the central AI OS".to_string(),
        metadata: std::collections::HashMap::new(),
        namespace: String::new(),
        agent: String::new(),
        session_id: String::new(),
    });
    let resp = svc.stream_signals(req).await.unwrap();
    let mut stream = resp.into_inner();

    use tokio_stream::StreamExt;
    let first = stream.next().await;
    // Should receive exactly one event
    assert!(first.is_some());
    let event = first.unwrap().unwrap();
    assert!(!event.signal_id.is_empty());
    assert_eq!(event.status, "Ok");
}

#[tokio::test]
async fn test_agent_receive_signals_sends_connected_event() {
    let processor = make_processor().await;
    let svc = AgentServiceImpl::new(processor);

    let session_id = Uuid::new_v4().to_string();
    let req = Request::new(ReceiveRequest {
        session_id: session_id.clone(),
    });
    let resp = svc.receive_signals(req).await.unwrap();
    let mut stream = resp.into_inner();

    use tokio_stream::StreamExt;
    let first = stream.next().await;
    assert!(first.is_some());
    let update = first.unwrap().unwrap();
    assert_eq!(update.event_type, "connected");
    assert!(update.content.contains(&session_id));
}

#[tokio::test]
async fn test_agent_receive_signals_fanout_after_send_signal() {
    use tokio::time::{timeout, Duration};
    use tokio_stream::StreamExt;

    let processor = make_processor().await;
    let agent_svc = AgentServiceImpl::new(processor.clone());

    let recv_req = Request::new(ReceiveRequest {
        session_id: Uuid::new_v4().to_string(),
    });
    let recv_resp = agent_svc.receive_signals(recv_req).await.unwrap();
    let mut stream = recv_resp.into_inner();

    // Initial connected event
    let connected = stream.next().await.unwrap().unwrap();
    assert_eq!(connected.event_type, "connected");

    let send_req = Request::new(AgentSignalRequest {
        source: "grpc".to_string(),
        channel: "test".to_string(),
        sender: "testagent".to_string(),
        content: "Remember that fanout works".to_string(),
        metadata: std::collections::HashMap::new(),
        namespace: "personal".to_string(),
        agent: String::new(),
        session_id: String::new(),
    });
    let send_resp = agent_svc.send_signal(send_req).await.unwrap().into_inner();
    assert_eq!(send_resp.status, "Ok");

    let next = timeout(Duration::from_secs(2), stream.next())
        .await
        .expect("expected fanout event within timeout");
    let update = next.expect("stream closed").expect("stream error");
    assert_eq!(update.event_type, "processed");
    assert!(update.content.contains("fanout works"));
}

#[test]
fn test_parse_source() {
    assert_eq!(
        SignalSource::parse(Some("grpc"), SignalSource::Grpc),
        SignalSource::Grpc
    );
    assert_eq!(
        SignalSource::parse(Some(""), SignalSource::Grpc),
        SignalSource::Grpc
    );
    assert_eq!(
        SignalSource::parse(Some("http"), SignalSource::Grpc),
        SignalSource::Http
    );
    assert_eq!(
        SignalSource::parse(Some("cli"), SignalSource::Grpc),
        SignalSource::Cli
    );
    assert_eq!(
        SignalSource::parse(Some("ws"), SignalSource::Grpc),
        SignalSource::WebSocket
    );
    assert_eq!(
        SignalSource::parse(Some("mcp"), SignalSource::Grpc),
        SignalSource::Mcp
    );
    assert_eq!(
        SignalSource::parse(Some("unknown"), SignalSource::Grpc),
        SignalSource::Grpc
    );
}

#[test]
fn test_non_empty() {
    assert_eq!(non_empty(String::new()), None);
    assert_eq!(non_empty("hello".to_string()), Some("hello".to_string()));
}

#[test]
fn test_response_to_string() {
    assert_eq!(
        response_to_string(signal::ResponseContent::Text("hello".to_string())),
        "hello"
    );
    assert_eq!(
        response_to_string(signal::ResponseContent::Error("err".to_string())),
        "err"
    );
    let json = serde_json::json!({"key": "val"});
    let s = response_to_string(signal::ResponseContent::Json(json));
    assert!(s.contains("key"));
}
