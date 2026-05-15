//! Principal threading + per-session audit events.
//!
//! Drives the trait directly; the auth gate keys off `Request` metadata,
//! not the gRPC transport. The streaming RPCs and end-to-end metadata
//! propagation are covered by the in-process TCP harness in `tests/io.rs`.

#![cfg(unix)]

use std::sync::Arc;
use std::time::Duration;

use brain_core::ApiKeyConfig;
use brainos_terminal::{
    pb::{terminal_session_server::TerminalSession, OpenRequest, PtySize, SessionHandle},
    TerminalAuth, TerminalBridge,
};
use identity::{ConfigIdentityStore, IdentityConfig, PrincipalConfig, Tier};
use observe::{BrainEvent, BroadcastObserver, Observer};
use tonic::Request;

fn open_request(program: &str, args: Vec<String>) -> OpenRequest {
    OpenRequest {
        program: program.to_string(),
        args,
        env: Default::default(),
        cwd: String::new(),
        initial_size: Some(PtySize {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        }),
        set_controlling_tty: false,
        client_id: String::new(),
    }
}

fn with_api_key<T>(mut req: Request<T>, key: &str) -> Request<T> {
    req.metadata_mut()
        .insert("x-api-key", key.parse().expect("ascii key"));
    req
}

fn store_for(agent_id: &str, scopes: Vec<&str>, tier: Tier) -> Arc<ConfigIdentityStore> {
    let cfg = IdentityConfig {
        user_id: "keshav".into(),
        principals: vec![PrincipalConfig {
            agent_id: agent_id.into(),
            scopes: scopes.into_iter().map(String::from).collect(),
            tier,
            path_allowlist: Vec::new(),
        }],
    };
    Arc::new(ConfigIdentityStore::from_config(cfg))
}

fn api_key(key: &str, agent_id: &str) -> ApiKeyConfig {
    ApiKeyConfig {
        key: key.into(),
        name: "test".into(),
        permissions: vec!["read".into(), "write".into()],
        agent_id: Some(agent_id.into()),
    }
}

// ── Back-compat: no auth wired ───────────────────────────────────────────────

#[tokio::test]
async fn open_without_auth_succeeds_and_carries_no_principal() {
    let bridge = TerminalBridge::new();
    let svc = bridge.svc();
    let session_id = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "sleep 5".into()],
        )))
        .await
        .expect("open without auth")
        .into_inner()
        .session_id;

    let meta = bridge.sessions().meta(&session_id).await.expect("meta");
    assert!(meta.principal.is_none());

    let _ = svc
        .close(Request::new(SessionHandle { session_id }))
        .await
        .unwrap();
}

// ── Observer wiring (auth optional) ──────────────────────────────────────────

#[tokio::test]
async fn observer_receives_opened_and_closed_events() {
    let observer = BroadcastObserver::new();
    let mut rx = observer.subscribe();

    let bridge = TerminalBridge::new().with_observer(observer.clone());
    let svc = bridge.svc();

    let session_id = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "printf hi".into()],
        )))
        .await
        .expect("open")
        .into_inner()
        .session_id;

    // First event must be Opened with our session_id.
    let opened = recv_one(&mut rx).await.expect("opened");
    match opened {
        BrainEvent::TerminalSessionOpened {
            session_id: ev_sid,
            program,
            principal,
            ..
        } => {
            assert_eq!(ev_sid, session_id);
            assert_eq!(program, "/bin/sh");
            assert!(principal.is_none(), "no auth wired → no principal");
        }
        other => panic!("expected TerminalSessionOpened, got {other:?}"),
    }

    let ack = svc
        .close(Request::new(SessionHandle {
            session_id: session_id.clone(),
        }))
        .await
        .expect("close")
        .into_inner();

    let closed = recv_one(&mut rx).await.expect("closed");
    match closed {
        BrainEvent::TerminalSessionClosed {
            session_id: ev_sid,
            exit_code,
            ..
        } => {
            assert_eq!(ev_sid, session_id);
            assert_eq!(exit_code, ack.exit_code);
        }
        other => panic!("expected TerminalSessionClosed, got {other:?}"),
    }
}

// ── Full auth path ───────────────────────────────────────────────────────────

#[tokio::test]
async fn open_with_valid_api_key_threads_principal_through_meta_and_event() {
    let observer = BroadcastObserver::new();
    let mut rx = observer.subscribe();

    let bridge = TerminalBridge::new()
        .with_auth(TerminalAuth::new(
            store_for("claude-code", vec!["terminal.*"], Tier::Execute),
            vec![api_key("k1", "claude-code")],
        ))
        .with_observer(observer.clone());
    let svc = bridge.svc();

    let session_id = svc
        .open(with_api_key(
            Request::new(open_request(
                "/bin/sh",
                vec!["-c".into(), "printf hi".into()],
            )),
            "k1",
        ))
        .await
        .expect("open with auth")
        .into_inner()
        .session_id;

    // Principal made it into SessionMeta.
    let meta = bridge.sessions().meta(&session_id).await.expect("meta");
    let p = meta.principal.expect("principal present");
    assert_eq!(p.agent_id.0, "claude-code");
    assert_eq!(p.user_id.0, "keshav");

    // …and into the Opened event.
    let opened = recv_one(&mut rx).await.expect("opened");
    match opened {
        BrainEvent::TerminalSessionOpened {
            principal: Some(summary),
            ..
        } => {
            assert_eq!(summary.agent_id, "claude-code");
            assert_eq!(summary.user_id, "keshav");
        }
        other => panic!("expected Opened with principal, got {other:?}"),
    }

    let _ = svc
        .close(with_api_key(
            Request::new(SessionHandle { session_id }),
            "k1",
        ))
        .await
        .expect("close");
}

#[tokio::test]
async fn open_missing_api_key_returns_unauthenticated() {
    let bridge = TerminalBridge::new().with_auth(TerminalAuth::new(
        store_for("claude-code", vec!["terminal.*"], Tier::Execute),
        vec![api_key("k1", "claude-code")],
    ));
    let svc = bridge.svc();

    let err = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "sleep 1".into()],
        )))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn open_unknown_api_key_returns_unauthenticated() {
    let bridge = TerminalBridge::new().with_auth(TerminalAuth::new(
        store_for("claude-code", vec!["terminal.*"], Tier::Execute),
        vec![api_key("k1", "claude-code")],
    ));
    let svc = bridge.svc();

    let err = svc
        .open(with_api_key(
            Request::new(open_request("/bin/sh", vec!["-c".into(), "sleep 1".into()])),
            "wrong-key",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unauthenticated);
}

#[tokio::test]
async fn open_insufficient_tier_returns_permission_denied() {
    // Principal is Read-tier — below Execute, so check returns Deny.
    let bridge = TerminalBridge::new().with_auth(TerminalAuth::new(
        store_for("readonly", vec!["terminal.*"], Tier::Read),
        vec![api_key("k1", "readonly")],
    ));
    let svc = bridge.svc();

    let err = svc
        .open(with_api_key(
            Request::new(open_request("/bin/sh", vec!["-c".into(), "sleep 1".into()])),
            "k1",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

#[tokio::test]
async fn open_missing_scope_escalates_to_permission_denied() {
    // Right tier, wrong scope — store returns EscalateToUser, which the
    // terminal bridge maps to PermissionDenied (no ConfirmationEngine here).
    let bridge = TerminalBridge::new().with_auth(TerminalAuth::new(
        store_for("scoped", vec!["fs.*"], Tier::Execute),
        vec![api_key("k1", "scoped")],
    ));
    let svc = bridge.svc();

    let err = svc
        .open(with_api_key(
            Request::new(open_request("/bin/sh", vec!["-c".into(), "sleep 1".into()])),
            "k1",
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::PermissionDenied);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async fn recv_one(rx: &mut tokio::sync::broadcast::Receiver<BrainEvent>) -> Option<BrainEvent> {
    tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .ok()?
        .ok()
}
