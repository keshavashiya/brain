//! PR13 I/O integration tests: Send / Resize / Signal + bidi Interact.
//!
//! Resize and Signal are unary, so we drive them through the trait directly.
//! Send (client-streaming) and Interact (bidi) need a real
//! `tonic::Streaming<T>`, so those tests stand up an in-process tonic
//! server/client pair over a localhost TCP loopback (`TcpIncoming` +
//! `Endpoint::connect`). All cfg(unix) — Windows ConPTY coverage comes
//! with the Phase 2 acceptance in PR20.

#![cfg(unix)]

use std::time::Duration;

use brainos_terminal::{
    pb::{
        client_frame, server_frame, terminal_session_client::TerminalSessionClient,
        terminal_session_server::TerminalSession, ClientFrame, InputChunk, OpenRequest, PtySize,
        ResizeRequest, SessionHandle, Sig, SignalRequest,
    },
    TerminalBridge,
};
use futures::stream;
use tokio::{net::TcpListener, sync::mpsc};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::transport::{server::TcpIncoming, Channel, Endpoint, Server};
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

// ── Direct-trait tests for unary RPCs ─────────────────────────────────────────

#[tokio::test]
async fn resize_succeeds_on_open_session() {
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

    svc.resize(Request::new(ResizeRequest {
        session_id: session_id.clone(),
        size: Some(PtySize {
            rows: 50,
            cols: 132,
            pixel_width: 0,
            pixel_height: 0,
        }),
    }))
    .await
    .expect("resize");

    assert_eq!(bridge.sessions().len().await, 1);

    let _ = svc
        .close(Request::new(SessionHandle { session_id }))
        .await
        .unwrap();
}

#[tokio::test]
async fn resize_unknown_session_returns_not_found() {
    let svc = TerminalBridge::new().svc();
    let err = svc
        .resize(Request::new(ResizeRequest {
            session_id: "missing".into(),
            size: Some(PtySize {
                rows: 1,
                cols: 1,
                pixel_width: 0,
                pixel_height: 0,
            }),
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

#[tokio::test]
async fn signal_sigterm_terminates_long_running_child() {
    let bridge = TerminalBridge::new();
    let svc = bridge.svc();
    let session_id = svc
        .open(Request::new(open_request(
            "/bin/sh",
            vec!["-c".into(), "sleep 30".into()],
        )))
        .await
        .expect("open")
        .into_inner()
        .session_id;

    svc.signal(Request::new(SignalRequest {
        session_id: session_id.clone(),
        signal: Sig::Sigterm as i32,
    }))
    .await
    .expect("signal");

    // Close should observe the already-exited child rather than killing it
    // itself: `was_killed=false` because the signal RPC already did the deed.
    // Give the kill a moment to take effect before Close.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let ack = svc
        .close(Request::new(SessionHandle {
            session_id: session_id.clone(),
        }))
        .await
        .expect("close")
        .into_inner();
    assert!(
        !ack.was_killed,
        "child should be dead before Close; got was_killed=true"
    );
}

#[tokio::test]
async fn signal_unspecified_rejected() {
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

    let err = svc
        .signal(Request::new(SignalRequest {
            session_id: session_id.clone(),
            signal: Sig::Unspecified as i32,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);

    let _ = svc
        .close(Request::new(SessionHandle { session_id }))
        .await
        .unwrap();
}

#[tokio::test]
async fn signal_unknown_session_returns_not_found() {
    let svc = TerminalBridge::new().svc();
    let err = svc
        .signal(Request::new(SignalRequest {
            session_id: "missing".into(),
            signal: Sig::Sigterm as i32,
        }))
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);
}

// ── In-process gRPC harness for streaming RPCs ────────────────────────────────

/// Stand up the `TerminalSession` server on a localhost TCP socket and
/// return a connected client. Tonic's `TcpIncoming` handles the
/// `Connected`/`AsyncRead`/`AsyncWrite` plumbing for us, so the test
/// harness stays small. The returned bridge handle lets the test inspect
/// registry state in parallel with the gRPC client.
async fn in_process_client() -> (TerminalSessionClient<Channel>, TerminalBridge) {
    let bridge = TerminalBridge::new();
    let server_bridge = bridge.clone();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let incoming = TcpIncoming::from(listener);

    tokio::spawn(async move {
        let _ = Server::builder()
            .add_service(server_bridge.into_server())
            .serve_with_incoming(incoming)
            .await;
    });

    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");

    (TerminalSessionClient::new(channel), bridge)
}

#[tokio::test]
async fn send_writes_input_and_round_trips_via_attach() {
    let (mut client, bridge) = in_process_client().await;

    // Open `cat` via the unary RPC, then drive Send via the streaming RPC.
    let session_id = client
        .open(open_request("/bin/cat", vec![]))
        .await
        .expect("open")
        .into_inner()
        .session_id;

    // Subscribe to the attach stream before sending input.
    let mut attach = client
        .attach(SessionHandle {
            session_id: session_id.clone(),
        })
        .await
        .expect("attach")
        .into_inner();

    // Send one InputChunk through the client-streaming RPC.
    let payload = b"hello-pr13\n".to_vec();
    let input_stream = stream::iter(vec![InputChunk {
        session_id: session_id.clone(),
        data: payload.clone(),
    }]);
    let ack = client.send(input_stream).await.expect("send").into_inner();
    assert_eq!(ack.bytes_written, payload.len() as u64);

    // Drain attach until we see the echo.
    let mut received = Vec::<u8>::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while let Ok(Some(item)) = tokio::time::timeout_at(deadline, attach.next()).await {
        let chunk = item.expect("attach item ok");
        if chunk.eof {
            break;
        }
        received.extend_from_slice(&chunk.data);
        if String::from_utf8_lossy(&received).contains("hello-pr13") {
            break;
        }
    }
    let s = String::from_utf8_lossy(&received);
    assert!(s.contains("hello-pr13"), "expected echo, got: {s:?}");

    let _ = client
        .close(SessionHandle { session_id })
        .await
        .expect("close");
    assert_eq!(bridge.sessions().len().await, 0);
}

#[tokio::test]
async fn send_rejects_chunk_missing_session_id() {
    let (mut client, _bridge) = in_process_client().await;
    let input_stream = stream::iter(vec![InputChunk {
        session_id: String::new(),
        data: b"oops".to_vec(),
    }]);
    let err = client.send(input_stream).await.unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn interact_full_lifecycle() {
    let (mut client, bridge) = in_process_client().await;

    let (in_tx, in_rx) = mpsc::channel::<ClientFrame>(8);
    let in_stream = ReceiverStream::new(in_rx);

    let mut out = client
        .interact(in_stream)
        .await
        .expect("interact")
        .into_inner();

    // 1. Open.
    in_tx
        .send(ClientFrame {
            k: Some(client_frame::K::Open(open_request("/bin/cat", vec![]))),
        })
        .await
        .unwrap();

    // 2. Expect Handle.
    let handle_frame = next_frame(&mut out).await.expect("handle frame");
    let session_id = match handle_frame.k {
        Some(server_frame::K::Handle(h)) => h.session_id,
        other => panic!("expected Handle, got {other:?}"),
    };
    assert!(!session_id.is_empty());

    // 3. Send Input (per-frame session_id can be empty: bound from Open).
    in_tx
        .send(ClientFrame {
            k: Some(client_frame::K::Input(InputChunk {
                session_id: String::new(),
                data: b"echo-via-interact\n".to_vec(),
            })),
        })
        .await
        .unwrap();

    // 4. Drain until we see both Ack and the echo in Output frames.
    let mut saw_ack = false;
    let mut received = Vec::<u8>::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while !(saw_ack && String::from_utf8_lossy(&received).contains("echo-via-interact")) {
        let frame = match tokio::time::timeout_at(deadline, next_frame(&mut out)).await {
            Ok(Some(f)) => f,
            _ => break,
        };
        match frame.k {
            Some(server_frame::K::Ack(_)) => saw_ack = true,
            Some(server_frame::K::Output(o)) => received.extend_from_slice(&o.data),
            Some(server_frame::K::Error(e)) => panic!("unexpected Error frame: {e}"),
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    assert!(saw_ack, "never saw Ack for the Input frame");
    assert!(
        String::from_utf8_lossy(&received).contains("echo-via-interact"),
        "never saw echo: {:?}",
        String::from_utf8_lossy(&received)
    );

    // 5. Close.
    in_tx
        .send(ClientFrame {
            k: Some(client_frame::K::Close(SessionHandle {
                session_id: String::new(),
            })),
        })
        .await
        .unwrap();

    let mut saw_closed = false;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while let Ok(Some(frame)) = tokio::time::timeout_at(deadline, next_frame(&mut out)).await {
        if let Some(server_frame::K::Closed(_)) = frame.k {
            saw_closed = true;
            break;
        }
    }
    assert!(saw_closed, "never saw Closed frame after Close request");
    // Give the server task one tick to drop the registry entry.
    for _ in 0..20 {
        if bridge.sessions().is_empty().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(bridge.sessions().len().await, 0);
}

#[tokio::test]
async fn interact_input_before_open_errors() {
    let (mut client, _bridge) = in_process_client().await;

    let (in_tx, in_rx) = mpsc::channel::<ClientFrame>(4);
    let in_stream = ReceiverStream::new(in_rx);
    let mut out = client
        .interact(in_stream)
        .await
        .expect("interact")
        .into_inner();

    in_tx
        .send(ClientFrame {
            k: Some(client_frame::K::Input(InputChunk {
                session_id: String::new(),
                data: b"x".to_vec(),
            })),
        })
        .await
        .unwrap();

    let frame = tokio::time::timeout(Duration::from_secs(1), next_frame(&mut out))
        .await
        .expect("frame within timeout")
        .expect("frame");
    match frame.k {
        Some(server_frame::K::Error(msg)) => {
            assert!(msg.contains("Input before Open"), "got: {msg}");
        }
        other => panic!("expected Error frame, got {other:?}"),
    }
}

#[tokio::test]
async fn interact_disconnect_without_close_cleans_session() {
    let (mut client, bridge) = in_process_client().await;

    let (in_tx, in_rx) = mpsc::channel::<ClientFrame>(4);
    let in_stream = ReceiverStream::new(in_rx);
    let mut out = client
        .interact(in_stream)
        .await
        .expect("interact")
        .into_inner();

    in_tx
        .send(ClientFrame {
            k: Some(client_frame::K::Open(open_request(
                "/bin/sh",
                vec!["-c".into(), "sleep 30".into()],
            ))),
        })
        .await
        .unwrap();
    let _ = next_frame(&mut out).await.expect("Handle frame");
    assert_eq!(bridge.sessions().len().await, 1);

    // Drop input → stream ends → server-side dispatch loop exits and
    // tears down the bound session.
    drop(in_tx);

    for _ in 0..30 {
        if bridge.sessions().is_empty().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        bridge.sessions().is_empty().await,
        "Interact stream end should tear down the bound session"
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async fn next_frame(
    stream: &mut tonic::Streaming<brainos_terminal::pb::ServerFrame>,
) -> Option<brainos_terminal::pb::ServerFrame> {
    let item = stream.next().await?;
    Some(item.expect("server frame should be Ok"))
}
