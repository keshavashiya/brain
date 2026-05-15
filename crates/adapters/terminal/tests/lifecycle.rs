//! PR12 lifecycle integration tests for `TerminalSvc`.
//!
//! Exercises the trait directly (bypassing tonic transport) — full gRPC
//! transport coverage lands with the Phase 2 acceptance test in PR20.

use std::time::Duration;

use brainos_terminal::{
    pb::{terminal_session_server::TerminalSession, OpenRequest, PtySize, SessionHandle},
    TerminalBridge,
};
use tokio_stream::StreamExt;
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

#[tokio::test]
async fn close_unknown_session_returns_not_found() {
    let svc = TerminalBridge::new().svc();
    let err = svc
        .close(Request::new(SessionHandle {
            session_id: "does-not-exist".into(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn attach_unknown_session_returns_not_found() {
    let svc = TerminalBridge::new().svc();
    let err = svc
        .attach(Request::new(SessionHandle {
            session_id: "does-not-exist".into(),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[cfg(unix)]
#[tokio::test]
async fn open_attach_close_full_lifecycle() {
    // Spawns `/bin/sh -c 'printf hello-from-brain && exit 0'`, attaches to
    // the PTY, drains the broadcast until EOF, then closes and verifies
    // the exit code. Cross-platform tests come with PR13/PR20.

    let bridge = TerminalBridge::new();
    let svc = bridge.svc();

    let open_resp = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "printf hello-from-brain".into()],
        )))
        .await
        .expect("open should succeed");
    let session_id = open_resp.into_inner().session_id;
    assert!(!session_id.is_empty());
    assert_eq!(bridge.sessions().len().await, 1);

    let attach_resp = svc
        .attach(Request::new(SessionHandle {
            session_id: session_id.clone(),
        }))
        .await
        .expect("attach should succeed");
    let mut stream = attach_resp.into_inner();

    // Collect bytes until EOF chunk (or a generous timeout cap).
    let mut received = Vec::<u8>::new();
    let mut saw_eof = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while let Ok(Some(item)) = tokio::time::timeout_at(deadline, stream.next()).await {
        let chunk = item.expect("attach stream item should be Ok");
        if chunk.eof {
            saw_eof = true;
            break;
        }
        received.extend_from_slice(&chunk.data);
    }
    assert!(saw_eof, "expected EOF chunk after child exits");
    let s = String::from_utf8_lossy(&received);
    assert!(
        s.contains("hello-from-brain"),
        "expected payload in PTY output, got: {s:?}"
    );

    let close_resp = svc
        .close(Request::new(SessionHandle {
            session_id: session_id.clone(),
        }))
        .await
        .expect("close should succeed");
    let ack = close_resp.into_inner();
    // Child exited on its own, not by kill.
    assert!(!ack.was_killed, "child exited naturally; was_killed=false");
    assert_eq!(ack.exit_code, 0, "child should exit 0");
    assert_eq!(bridge.sessions().len().await, 0);
}

#[cfg(unix)]
#[tokio::test]
async fn close_kills_a_long_running_child() {
    let bridge = TerminalBridge::new();
    let svc = bridge.svc();

    let open_resp = svc
        .open(Request::new(open_request(
            "/bin/sh",
            // Sleep long enough that we'll certainly be the ones to kill it.
            vec!["-c".into(), "sleep 30".into()],
        )))
        .await
        .expect("open should succeed");
    let session_id = open_resp.into_inner().session_id;

    let close_resp = svc
        .close(Request::new(SessionHandle {
            session_id: session_id.clone(),
        }))
        .await
        .expect("close should succeed");
    let ack = close_resp.into_inner();
    assert!(ack.was_killed, "long-running child must be killed by Close");
    assert_eq!(bridge.sessions().len().await, 0);
}

#[cfg(unix)]
#[tokio::test]
async fn registry_meta_reflects_open() {
    let bridge = TerminalBridge::new();
    let svc = bridge.svc();

    let session_id = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "sleep 5".into()],
        )))
        .await
        .expect("open")
        .into_inner()
        .session_id;

    let meta = bridge
        .sessions()
        .meta(&session_id)
        .await
        .expect("meta should exist");
    assert_eq!(meta.program, "/bin/sh");
    assert_eq!(meta.args, vec!["-c".to_string(), "sleep 5".to_string()]);
    assert_eq!(meta.size.rows, 24);
    assert_eq!(meta.size.cols, 80);

    // Clean up so the test runner doesn't hang on lingering children.
    let _ = svc
        .close(Request::new(SessionHandle { session_id }))
        .await
        .unwrap();
}
