//! Tonic implementation of the `brain.terminal.v1.TerminalSession` service.
//!
//! Open / Close / Attach / Send / Resize / Signal + bidi `Interact`. The
//! split RPCs delegate to crate-private helpers (`open_inner`,
//! `close_inner`, `write_input_inner`, …) which `Interact` also drives
//! directly, so the two surfaces share one code path per operation.

use std::{io::Read, sync::Arc, time::SystemTime};

use bytes::Bytes;
use chrono::Utc;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};
use tracing::warn;
use uuid::Uuid;

use crate::{
    pb::{
        self, client_frame, server_frame, terminal_session_server::TerminalSession, ClientFrame,
        CloseAck, InputChunk, OpenRequest, OutputChunk, ResizeAck, ResizeRequest, SendAck,
        ServerFrame, SessionHandle, Sig, SignalAck, SignalRequest,
    },
    session::{Session, IN_MPSC_CAPACITY, OUT_BROADCAST_CAPACITY},
    types::{SessionMeta, TermSize},
    SessionRegistry,
};

/// Buffer used by the PTY reader thread between `Read` syscalls. 8 KiB
/// matches the typical PTY line discipline kernel buffer and avoids
/// pathological many-tiny-chunks behavior on slow producers.
const PTY_READ_BUFFER_SIZE: usize = 8 * 1024;

/// Channel buffer for the per-attach / per-interact output stream pump.
/// Independent from the broadcast capacity — this is just the in-process
/// mpsc that hands frames to tonic's outbound encoder.
const STREAM_OUT_BUFFER: usize = 64;

/// Tonic service implementation. Cheap to clone (everything inside is
/// `Arc`-ed) so tonic can spawn one per concurrent RPC.
#[derive(Clone)]
pub struct TerminalSvc {
    registry: Arc<SessionRegistry>,
}

impl TerminalSvc {
    pub fn new(registry: Arc<SessionRegistry>) -> Self {
        Self { registry }
    }

    pub fn registry(&self) -> &Arc<SessionRegistry> {
        &self.registry
    }
}

fn term_size_from_pb(pb: Option<pb::PtySize>) -> TermSize {
    match pb {
        Some(s) => TermSize {
            rows: s.rows as u16,
            cols: s.cols as u16,
            pixel_width: s.pixel_width as u16,
            pixel_height: s.pixel_height as u16,
        },
        None => TermSize::default(),
    }
}

fn to_pty_size(s: TermSize) -> PtySize {
    PtySize {
        rows: s.rows,
        cols: s.cols,
        pixel_width: s.pixel_width,
        pixel_height: s.pixel_height,
    }
}

fn timestamp_now() -> Option<prost_types::Timestamp> {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?;
    Some(prost_types::Timestamp {
        seconds: now.as_secs() as i64,
        nanos: now.subsec_nanos() as i32,
    })
}

// ── Crate-private helpers (used by both split RPCs and Interact) ──────────────

impl TerminalSvc {
    async fn open_inner(&self, r: OpenRequest) -> Result<SessionHandle, Status> {
        let size = term_size_from_pb(r.initial_size);

        let pty = native_pty_system();
        let pair = pty
            .openpty(to_pty_size(size))
            .map_err(|e| Status::internal(format!("openpty: {e}")))?;

        let mut cmd = if r.program.is_empty() {
            CommandBuilder::new_default_prog()
        } else {
            CommandBuilder::new(&r.program)
        };
        for a in &r.args {
            cmd.arg(a);
        }
        for (k, v) in &r.env {
            cmd.env(k, v);
        }
        if !r.cwd.is_empty() {
            cmd.cwd(&r.cwd);
        }

        let child = pair
            .slave
            .spawn_command(cmd)
            .map_err(|e| Status::internal(format!("spawn: {e}")))?;
        drop(pair.slave);

        let master = Arc::new(Mutex::new(pair.master));
        let (out_tx, out_anchor) = broadcast::channel::<Bytes>(OUT_BROADCAST_CAPACITY);
        let (in_tx, mut in_rx) = mpsc::channel::<Bytes>(IN_MPSC_CAPACITY);

        // PTY reader → broadcast. The pump owns the *only* `Sender`;
        // see `Session::out_anchor` for the EOF-propagation rationale.
        {
            let reader_res = master.lock().await.try_clone_reader();
            let reader = reader_res.map_err(|e| Status::internal(format!("clone_reader: {e}")))?;
            tokio::task::spawn_blocking(move || pump_reader(reader, out_tx));
        }

        // mpsc → PTY writer. Lives on a blocking task until `in_rx` closes
        // (which happens when the last `in_tx` clone is dropped — i.e.
        // session removal).
        {
            let writer_res = master.lock().await.take_writer();
            let writer = writer_res.map_err(|e| Status::internal(format!("take_writer: {e}")))?;
            tokio::task::spawn_blocking(move || {
                let mut writer = writer;
                while let Some(chunk) = in_rx.blocking_recv() {
                    use std::io::Write;
                    if writer.write_all(&chunk).is_err() {
                        break;
                    }
                    let _ = writer.flush();
                }
            });
        }

        let session_id = Uuid::new_v4().to_string();
        let meta = SessionMeta {
            session_id: session_id.clone(),
            program: r.program,
            args: r.args,
            cwd: if r.cwd.is_empty() { None } else { Some(r.cwd) },
            opened_at: Utc::now(),
            client_id: if r.client_id.is_empty() {
                None
            } else {
                Some(r.client_id)
            },
            size,
        };

        let session = Arc::new(Session {
            meta,
            out_anchor,
            in_tx,
            master,
            child: Arc::new(Mutex::new(child)),
        });
        self.registry.insert(session).await;

        Ok(SessionHandle { session_id })
    }

    async fn close_inner(&self, id: &str) -> Result<CloseAck, Status> {
        let session = self
            .registry
            .remove(&id.to_string())
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}'")))?;

        let mut child = session.child.lock().await;
        let already_exited = matches!(child.try_wait(), Ok(Some(_)));
        let was_killed = if already_exited {
            false
        } else {
            child.kill().is_ok()
        };
        let exit_code = child.wait().map(|s| s.exit_code() as i32).unwrap_or(-1);

        Ok(CloseAck {
            exit_code,
            was_killed,
        })
    }

    async fn lookup(&self, id: &str) -> Result<Arc<Session>, Status> {
        self.registry
            .get(&id.to_string())
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}'")))
    }

    /// Buffer one input chunk into the session's PTY writer pump. Returns the
    /// number of bytes that were accepted into the in-process queue. A full
    /// queue (slow PTY consumer) backpressures the caller.
    async fn write_input_inner(&self, id: &str, data: Bytes) -> Result<u64, Status> {
        let session = self.lookup(id).await?;
        let len = data.len() as u64;
        session
            .in_tx
            .send(data)
            .await
            .map_err(|_| Status::aborted("session writer closed"))?;
        Ok(len)
    }

    async fn resize_inner(&self, id: &str, size: TermSize) -> Result<(), Status> {
        let session = self.lookup(id).await?;
        let result = session.master.lock().await.resize(to_pty_size(size));
        result.map_err(|e| Status::internal(format!("resize: {e}")))
    }

    /// Routes a `Sig` to the underlying PTY/child.
    ///
    /// - `Sigint` → `\x03` (line discipline interprets, portable on Unix and
    ///   ConPTY on Windows).
    /// - `Sigquit` → `\x1c`.
    /// - `Sigterm` / `Sighup` / `Sigkill` → `child.kill()` (best-effort;
    ///   true SIGHUP support is not exposed by portable-pty).
    /// - `Unspecified` → no-op (returns `InvalidArgument`).
    async fn signal_inner(&self, id: &str, sig: Sig) -> Result<(), Status> {
        let session = self.lookup(id).await?;
        match sig {
            Sig::Sigint => session
                .in_tx
                .send(Bytes::from_static(b"\x03"))
                .await
                .map_err(|_| Status::aborted("session writer closed")),
            Sig::Sigquit => session
                .in_tx
                .send(Bytes::from_static(b"\x1c"))
                .await
                .map_err(|_| Status::aborted("session writer closed")),
            Sig::Sigterm | Sig::Sighup | Sig::Sigkill => session
                .child
                .lock()
                .await
                .kill()
                .map_err(|e| Status::internal(format!("kill: {e}"))),
            Sig::Unspecified => Err(Status::invalid_argument("signal must not be UNSPECIFIED")),
        }
    }
}

// ── Tonic trait implementation ────────────────────────────────────────────────

#[tonic::async_trait]
impl TerminalSession for TerminalSvc {
    type AttachStream = ReceiverStream<Result<OutputChunk, Status>>;
    type InteractStream = ReceiverStream<Result<ServerFrame, Status>>;

    async fn open(&self, req: Request<OpenRequest>) -> Result<Response<SessionHandle>, Status> {
        Ok(Response::new(self.open_inner(req.into_inner()).await?))
    }

    async fn close(&self, req: Request<SessionHandle>) -> Result<Response<CloseAck>, Status> {
        let id = req.into_inner().session_id;
        Ok(Response::new(self.close_inner(&id).await?))
    }

    async fn attach(
        &self,
        req: Request<SessionHandle>,
    ) -> Result<Response<Self::AttachStream>, Status> {
        let id = req.into_inner().session_id;
        let session = self.lookup(&id).await?;

        let mut rx = session.out_anchor.resubscribe();
        let (tx, out) = mpsc::channel::<Result<OutputChunk, Status>>(STREAM_OUT_BUFFER);

        tokio::spawn(async move {
            let mut seq: u64 = 0;
            loop {
                match rx.recv().await {
                    Ok(bytes) => {
                        seq += 1;
                        let chunk = OutputChunk {
                            data: bytes.to_vec(),
                            ts: timestamp_now(),
                            seq,
                            eof: false,
                        };
                        if tx.send(Ok(chunk)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        let _ = tx
                            .send(Ok(OutputChunk {
                                data: Vec::new(),
                                ts: timestamp_now(),
                                seq: seq + 1,
                                eof: true,
                            }))
                            .await;
                        break;
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            session = %id,
                            dropped = n,
                            "attach stream lagged — bumped subscriber"
                        );
                        let _ = tx
                            .send(Err(Status::resource_exhausted(format!(
                                "attach lagged: {n} chunks dropped"
                            ))))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(out)))
    }

    async fn send(&self, req: Request<Streaming<InputChunk>>) -> Result<Response<SendAck>, Status> {
        let mut stream = req.into_inner();
        let mut total: u64 = 0;
        while let Some(chunk_res) = stream.next().await {
            let chunk = chunk_res?;
            if chunk.session_id.is_empty() {
                return Err(Status::invalid_argument("input chunk missing session_id"));
            }
            total += self
                .write_input_inner(&chunk.session_id, Bytes::from(chunk.data))
                .await?;
        }
        Ok(Response::new(SendAck {
            bytes_written: total,
        }))
    }

    async fn resize(&self, req: Request<ResizeRequest>) -> Result<Response<ResizeAck>, Status> {
        let r = req.into_inner();
        let size = term_size_from_pb(r.size);
        self.resize_inner(&r.session_id, size).await?;
        Ok(Response::new(ResizeAck {}))
    }

    async fn signal(&self, req: Request<SignalRequest>) -> Result<Response<SignalAck>, Status> {
        let r = req.into_inner();
        let sig = Sig::try_from(r.signal).unwrap_or(Sig::Unspecified);
        self.signal_inner(&r.session_id, sig).await?;
        Ok(Response::new(SignalAck {}))
    }

    /// Bidirectional perf path: one stream for the whole session lifetime.
    ///
    /// Frame contract:
    /// - First client frame **must** be `Open`. The server replies with
    ///   `Handle` once the session is created, then begins emitting `Output`
    ///   frames continuously from the PTY broadcast.
    /// - Subsequent client frames: `Input`, `Resize`, `Signal`, or `Close`.
    /// - Server frames are interleaved: `Output` from the PTY, plus `Ack`
    ///   for inputs, `Closed` on a clean close, and `Error` on any failure.
    ///   `Error` does not always terminate — only `Close` (or stream end)
    ///   does — but the output forwarder always emits an `Output{eof=true}`
    ///   when the PTY broadcast closes, mirroring `Attach`.
    async fn interact(
        &self,
        req: Request<Streaming<ClientFrame>>,
    ) -> Result<Response<Self::InteractStream>, Status> {
        let mut input = req.into_inner();
        let (tx, out) = mpsc::channel::<Result<ServerFrame, Status>>(STREAM_OUT_BUFFER);
        let svc = self.clone();

        tokio::spawn(async move {
            let mut bound_id: Option<String> = None;
            let mut output_task: Option<tokio::task::JoinHandle<()>> = None;

            while let Some(frame_res) = input.next().await {
                let frame = match frame_res {
                    Ok(f) => f,
                    Err(_) => break,
                };
                let Some(k) = frame.k else {
                    continue;
                };

                match k {
                    client_frame::K::Open(open_req) => {
                        if bound_id.is_some() {
                            let _ = tx
                                .send(Ok(error_frame("Interact: session already opened")))
                                .await;
                            continue;
                        }
                        match svc.open_inner(open_req).await {
                            Ok(handle) => {
                                bound_id = Some(handle.session_id.clone());
                                output_task = Some(spawn_output_forwarder(
                                    svc.clone(),
                                    handle.session_id.clone(),
                                    tx.clone(),
                                ));
                                let _ = tx
                                    .send(Ok(ServerFrame {
                                        k: Some(server_frame::K::Handle(handle)),
                                    }))
                                    .await;
                            }
                            Err(s) => {
                                let _ = tx.send(Ok(error_frame(s.message()))).await;
                                break;
                            }
                        }
                    }

                    client_frame::K::Input(chunk) => {
                        let target = pick_id(&bound_id, &chunk.session_id);
                        if let Some(id) = target {
                            match svc.write_input_inner(&id, Bytes::from(chunk.data)).await {
                                Ok(n) => {
                                    let _ = tx
                                        .send(Ok(ServerFrame {
                                            k: Some(server_frame::K::Ack(SendAck {
                                                bytes_written: n,
                                            })),
                                        }))
                                        .await;
                                }
                                Err(s) => {
                                    let _ = tx.send(Ok(error_frame(s.message()))).await;
                                }
                            }
                        } else {
                            let _ = tx
                                .send(Ok(error_frame("Interact: Input before Open")))
                                .await;
                        }
                    }

                    client_frame::K::Resize(r) => {
                        let target = pick_id(&bound_id, &r.session_id);
                        if let Some(id) = target {
                            let size = term_size_from_pb(r.size);
                            if let Err(s) = svc.resize_inner(&id, size).await {
                                let _ = tx.send(Ok(error_frame(s.message()))).await;
                            }
                        }
                    }

                    client_frame::K::Signal(sg) => {
                        let target = pick_id(&bound_id, &sg.session_id);
                        if let Some(id) = target {
                            let sig = Sig::try_from(sg.signal).unwrap_or(Sig::Unspecified);
                            if let Err(s) = svc.signal_inner(&id, sig).await {
                                let _ = tx.send(Ok(error_frame(s.message()))).await;
                            }
                        }
                    }

                    client_frame::K::Close(handle) => {
                        let target = pick_id(&bound_id, &handle.session_id);
                        if let Some(id) = target {
                            match svc.close_inner(&id).await {
                                Ok(ack) => {
                                    let _ = tx
                                        .send(Ok(ServerFrame {
                                            k: Some(server_frame::K::Closed(ack)),
                                        }))
                                        .await;
                                }
                                Err(s) => {
                                    let _ = tx.send(Ok(error_frame(s.message()))).await;
                                }
                            }
                        }
                        break;
                    }
                }
            }

            // Client closed the stream without Close. Tear down any
            // still-running session bound to this Interact so PTYs don't
            // leak on disconnect (deterministic-teardown contract from §3.1).
            if let Some(id) = bound_id {
                if svc.registry.get(&id).await.is_some() {
                    let _ = svc.close_inner(&id).await;
                }
            }
            if let Some(t) = output_task {
                t.abort();
            }
        });

        Ok(Response::new(ReceiverStream::new(out)))
    }
}

fn error_frame(msg: impl Into<String>) -> ServerFrame {
    ServerFrame {
        k: Some(server_frame::K::Error(msg.into())),
    }
}

/// Prefer the per-frame `session_id` if non-empty (lets clients address
/// other sessions over one Interact stream, though the typical pattern is
/// to bind once via Open and leave the field empty thereafter).
fn pick_id(bound: &Option<String>, per_frame: &str) -> Option<String> {
    if !per_frame.is_empty() {
        Some(per_frame.to_string())
    } else {
        bound.clone()
    }
}

/// Spawn the per-Interact PTY → `ServerFrame::Output` forwarder. Mirrors
/// the Attach loop but wraps chunks in `ServerFrame` instead of yielding
/// `OutputChunk` directly.
fn spawn_output_forwarder(
    svc: TerminalSvc,
    session_id: String,
    tx: mpsc::Sender<Result<ServerFrame, Status>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let Some(session) = svc.registry.get(&session_id).await else {
            let _ = tx
                .send(Ok(error_frame(format!(
                    "Interact: session '{session_id}' vanished"
                ))))
                .await;
            return;
        };
        let mut rx = session.out_anchor.resubscribe();
        let mut seq: u64 = 0;
        loop {
            match rx.recv().await {
                Ok(bytes) => {
                    seq += 1;
                    let chunk = OutputChunk {
                        data: bytes.to_vec(),
                        ts: timestamp_now(),
                        seq,
                        eof: false,
                    };
                    let frame = ServerFrame {
                        k: Some(server_frame::K::Output(chunk)),
                    };
                    if tx.send(Ok(frame)).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => {
                    let _ = tx
                        .send(Ok(ServerFrame {
                            k: Some(server_frame::K::Output(OutputChunk {
                                data: Vec::new(),
                                ts: timestamp_now(),
                                seq: seq + 1,
                                eof: true,
                            })),
                        }))
                        .await;
                    break;
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!(
                        session = %session_id,
                        dropped = n,
                        "interact stream lagged — bumped subscriber"
                    );
                    let _ = tx
                        .send(Ok(error_frame(format!(
                            "interact lagged: {n} chunks dropped"
                        ))))
                        .await;
                    break;
                }
            }
        }
    })
}

/// Blocking PTY reader pump. Runs on its own `spawn_blocking` task.
/// Sends raw bytes into `out_tx` until the reader hits EOF or a fatal
/// error. Dropping `out_tx` here is what closes the broadcast for every
/// subscriber — that's how child-exit propagates to clients.
fn pump_reader(mut reader: Box<dyn Read + Send>, out_tx: broadcast::Sender<Bytes>) {
    let mut buf = vec![0u8; PTY_READ_BUFFER_SIZE];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => {
                // `send` returns Err only when there are zero receivers.
                // We keep pumping in case a new subscriber attaches later;
                // the broadcast keeps the most recent `OUT_BROADCAST_CAPACITY`
                // chunks for them either way.
                let _ = out_tx.send(Bytes::copy_from_slice(&buf[..n]));
            }
            Err(_) => break,
        }
    }
}
