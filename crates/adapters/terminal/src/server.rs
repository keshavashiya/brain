//! Tonic implementation of the `brain.terminal.v1.TerminalSession` service.
//!
//! PR12 lands `Open` / `Close` / `Attach`. The remaining RPCs (`Send`,
//! `Resize`, `Signal`, `Interact`) compile but return `Status::unimplemented`
//! until PR13.

use std::{io::Read, sync::Arc, time::SystemTime};

use bytes::Bytes;
use chrono::Utc;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};
use tracing::warn;
use uuid::Uuid;

use crate::{
    pb::{
        self, terminal_session_server::TerminalSession, ClientFrame, CloseAck, InputChunk,
        OpenRequest, OutputChunk, ResizeAck, ResizeRequest, SendAck, ServerFrame, SessionHandle,
        SignalAck, SignalRequest,
    },
    session::{Session, IN_MPSC_CAPACITY, OUT_BROADCAST_CAPACITY},
    types::{SessionMeta, TermSize},
    SessionRegistry,
};

/// Buffer used by the PTY reader thread between `Read` syscalls. 8 KiB
/// matches the typical PTY line discipline kernel buffer and avoids
/// pathological many-tiny-chunks behavior on slow producers.
const PTY_READ_BUFFER_SIZE: usize = 8 * 1024;

/// Channel buffer for the per-attach output stream pump. Independent from
/// the broadcast capacity — this is just the in-process mpsc that hands
/// frames to tonic's outbound encoder.
const ATTACH_STREAM_BUFFER: usize = 64;

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

#[tonic::async_trait]
impl TerminalSession for TerminalSvc {
    type AttachStream = ReceiverStream<Result<OutputChunk, Status>>;
    type InteractStream = ReceiverStream<Result<ServerFrame, Status>>;

    async fn open(&self, req: Request<OpenRequest>) -> Result<Response<SessionHandle>, Status> {
        let r = req.into_inner();
        let size = term_size_from_pb(r.initial_size);

        // Spawn the PTY pair + child process.
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
        // Drop the slave; child owns its end and we don't need the handle.
        drop(pair.slave);

        let master = Arc::new(Mutex::new(pair.master));
        let (out_tx, out_anchor) = broadcast::channel::<Bytes>(OUT_BROADCAST_CAPACITY);
        let (in_tx, mut in_rx) = mpsc::channel::<Bytes>(IN_MPSC_CAPACITY);

        // PTY reader → broadcast. portable_pty's reader is blocking, so run
        // it on a blocking task to keep the tokio runtime healthy.
        //
        // The pump owns the *only* `Sender`. The session stores the matched
        // `Receiver` as a resubscribe anchor (see `Session::out_anchor`).
        // When the PTY hits EOF, the pump returns and drops the sender —
        // every subscriber then observes `RecvError::Closed`, which is how
        // child-exit propagates back to attached clients.
        {
            let reader_res = master.lock().await.try_clone_reader();
            let reader = reader_res.map_err(|e| Status::internal(format!("clone_reader: {e}")))?;
            tokio::task::spawn_blocking(move || pump_reader(reader, out_tx));
        }

        // mpsc → PTY writer. Same blocking-task pattern. Held alive so PR13
        // Send/Interact has a writer end immediately on Open. Drops on Close.
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

        Ok(Response::new(SessionHandle { session_id }))
    }

    async fn close(&self, req: Request<SessionHandle>) -> Result<Response<CloseAck>, Status> {
        let id = req.into_inner().session_id;
        let session = self
            .registry
            .remove(&id)
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}'")))?;

        let mut child = session.child.lock().await;

        // Was the child already done when we got here? Determines
        // whether the proto `was_killed` flag should be set.
        let already_exited = matches!(child.try_wait(), Ok(Some(_)));

        let was_killed = if already_exited {
            false
        } else {
            // Best-effort kill — already-dead is fine.
            child.kill().is_ok()
        };

        let exit_code = child.wait().map(|s| s.exit_code() as i32).unwrap_or(-1);

        Ok(Response::new(CloseAck {
            exit_code,
            was_killed,
        }))
    }

    async fn attach(
        &self,
        req: Request<SessionHandle>,
    ) -> Result<Response<Self::AttachStream>, Status> {
        let id = req.into_inner().session_id;
        let session = self
            .registry
            .get(&id)
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}'")))?;

        let mut rx = session.out_anchor.resubscribe();
        let (tx, out) = mpsc::channel::<Result<OutputChunk, Status>>(ATTACH_STREAM_BUFFER);

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
                        // Producer is gone — PTY reader exited (process
                        // ended, or session was closed). Signal EOF
                        // explicitly so the client gets a clean stream end.
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

    // ── Deferred to PR13 ──────────────────────────────────────────────────

    async fn send(
        &self,
        _req: Request<Streaming<InputChunk>>,
    ) -> Result<Response<SendAck>, Status> {
        Err(Status::unimplemented("Send: PR13"))
    }

    async fn resize(&self, _req: Request<ResizeRequest>) -> Result<Response<ResizeAck>, Status> {
        Err(Status::unimplemented("Resize: PR13"))
    }

    async fn signal(&self, _req: Request<SignalRequest>) -> Result<Response<SignalAck>, Status> {
        Err(Status::unimplemented("Signal: PR13"))
    }

    async fn interact(
        &self,
        _req: Request<Streaming<ClientFrame>>,
    ) -> Result<Response<Self::InteractStream>, Status> {
        Err(Status::unimplemented("Interact: PR13"))
    }
}

/// Blocking PTY reader pump. Runs on its own `spawn_blocking` task.
/// Sends raw bytes into `out_tx` until the reader hits EOF or a fatal
/// error. Dropping `out_tx` here signals the broadcast `Closed` arm to
/// every active `Attach` subscriber, which is how EOF propagates to clients.
fn pump_reader(mut reader: Box<dyn Read + Send>, out_tx: broadcast::Sender<Bytes>) {
    let mut buf = vec![0u8; PTY_READ_BUFFER_SIZE];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => {
                if out_tx.send(Bytes::copy_from_slice(&buf[..n])).is_err() {
                    // No subscribers AND nobody holding the channel — fine.
                    // Broadcast::send returns Err only when there are zero
                    // active receivers; we keep pumping in case a new
                    // Attach arrives. But the broadcast keeps the buffered
                    // tail, so we don't lose data either way.
                }
            }
            Err(_) => break,
        }
    }
}
