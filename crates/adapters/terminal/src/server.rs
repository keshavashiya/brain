//! Tonic implementation of the `brain.terminal.v1.TerminalSession` service.
//!
//! Open / Close / Attach / Send / Resize / Signal + bidi `Interact`. The
//! split RPCs delegate to crate-private helpers (`open_inner`,
//! `close_inner`, `write_input_inner`, …) which `Interact` also drives
//! directly, so the two surfaces share one code path per operation.

use std::{
    collections::VecDeque,
    io::Read,
    sync::{Arc, Mutex as StdMutex},
    time::SystemTime,
};

use bytes::Bytes;
use chrono::Utc;
use identity::{AgentHint, AuthorizationRequest, CheckOutcome, Principal, Tier};
use observe::{BrainEvent, Observer, PrincipalSummary};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tonic::{Request, Response, Status, Streaming};
use tracing::warn;
use uuid::Uuid;

use crate::{
    graph::TerminalGraphSink,
    pb::{
        self, client_frame, server_frame, terminal_session_server::TerminalSession, ClientFrame,
        CloseAck, InputChunk, OpenRequest, OutputChunk, ResizeAck, ResizeRequest, SendAck,
        ServerFrame, SessionHandle, Sig, SignalAck, SignalRequest,
    },
    session::{Session, IN_MPSC_CAPACITY, OUT_BROADCAST_CAPACITY},
    types::{SessionMeta, TermSize},
    SessionRegistry, TerminalAuth,
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
    auth: Option<TerminalAuth>,
    observer: Option<Arc<dyn Observer>>,
    graph_sink: Option<Arc<dyn TerminalGraphSink>>,
}

impl TerminalSvc {
    pub fn new(
        registry: Arc<SessionRegistry>,
        auth: Option<TerminalAuth>,
        observer: Option<Arc<dyn Observer>>,
        graph_sink: Option<Arc<dyn TerminalGraphSink>>,
    ) -> Self {
        Self {
            registry,
            auth,
            observer,
            graph_sink,
        }
    }

    pub fn registry(&self) -> &Arc<SessionRegistry> {
        &self.registry
    }

    /// Pipeline-friendly Open: same path as the gRPC `Open` RPC but skips
    /// metadata-based authorization (the caller, typically `SignalProcessor`,
    /// has already gated via `IdentityStore::check`). The `principal` arg
    /// flows straight into [`SessionMeta`] and the audit event.
    pub async fn open_via_pipeline(
        &self,
        request: OpenRequest,
        principal: Option<Principal>,
    ) -> Result<SessionHandle, Status> {
        self.open_inner(request, principal).await
    }

    /// Pipeline-friendly Close: same path as the gRPC `Close` RPC, no
    /// auth metadata.
    pub async fn close_via_pipeline(&self, id: &str) -> Result<CloseAck, Status> {
        self.close_inner(id).await
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

fn principal_summary(p: &Principal) -> PrincipalSummary {
    PrincipalSummary {
        user_id: p.user_id.0.clone(),
        agent_id: p.agent_id.0.clone(),
    }
}

/// Read the api-key from either `x-api-key` or `authorization` (with
/// optional `Bearer` prefix). Mirrors the pattern in `brainos-grpcadapter`.
fn api_key_from_metadata<T>(req: &Request<T>) -> Option<String> {
    let metadata = req.metadata();
    if let Some(v) = metadata.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(v.to_string());
    }
    let authz = metadata
        .get("authorization")
        .and_then(|v| v.to_str().ok())?;
    let key = brain::auth::extract_bearer_from_value(authz).unwrap_or(authz);
    Some(key.to_string())
}

// ── Crate-private helpers (used by both split RPCs and Interact) ──────────────

impl TerminalSvc {
    /// Resolve a [`Principal`] from request metadata (api-key → agent_id →
    /// `IdentityStore::principal_for`) **and** authorize the action.
    ///
    /// - No auth wiring on the bridge → `Ok(None)` (gate skipped, back-compat).
    /// - Auth wired, no api-key → `Status::unauthenticated`.
    /// - Auth wired, api-key unknown → `Status::unauthenticated`.
    /// - Auth wired, principal resolved, check returns `Allow` → `Ok(Some(p))`.
    /// - Auth wired, principal resolved, check returns `Deny`/`EscalateToUser`
    ///   → `Status::permission_denied` with the carried reason.
    ///
    /// The `verb_action` argument names the RPC (`open`, `attach`, `send`,
    /// `resize`, `signal`, `close`). The `verb_ns` is always `terminal`.
    /// Terminal RPCs require `Tier::Execute` — spawning processes / driving
    /// a running shell is qualitatively the same as `shell.exec`.
    async fn authorize<T>(
        &self,
        req: &Request<T>,
        verb_action: &str,
        modifiers: serde_json::Value,
    ) -> Result<Option<Principal>, Status> {
        let Some(auth) = self.auth.as_ref() else {
            return Ok(None);
        };
        let key =
            api_key_from_metadata(req).ok_or_else(|| Status::unauthenticated("missing api-key"))?;
        let agent_id = auth
            .api_keys
            .iter()
            .find(|k| k.key == key)
            .and_then(|k| k.agent_id.clone())
            .ok_or_else(|| Status::unauthenticated("unknown api-key"))?;
        let principal = auth
            .identity
            .principal_for(&AgentHint::AgentId(agent_id.into()))
            .await
            .map_err(|e| Status::unauthenticated(format!("principal lookup: {e}")))?;
        let authz_req =
            AuthorizationRequest::new("terminal", verb_action).with_modifiers(modifiers);
        match auth
            .identity
            .check(&principal, &authz_req, Tier::Execute)
            .await
        {
            CheckOutcome::Allow => Ok(Some(principal)),
            CheckOutcome::EscalateToUser { reason } => Err(Status::permission_denied(format!(
                "terminal.{verb_action} requires user confirmation: {reason}"
            ))),
            CheckOutcome::Deny { reason } => Err(Status::permission_denied(format!(
                "terminal.{verb_action} denied: {reason}"
            ))),
        }
    }

    async fn publish(&self, ev: BrainEvent) {
        if let Some(obs) = self.observer.as_ref() {
            // Treat publish failures (closed bus) as informational; the
            // bridge should not fail an RPC because no one is listening.
            let _ = obs.publish(ev).await;
        }
    }

    async fn open_inner(
        &self,
        r: OpenRequest,
        principal: Option<Principal>,
    ) -> Result<SessionHandle, Status> {
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
        let replay: Arc<StdMutex<VecDeque<Bytes>>> = Arc::new(StdMutex::new(
            VecDeque::with_capacity(OUT_BROADCAST_CAPACITY),
        ));

        // PTY reader → replay + broadcast. The pump owns the *only* `Sender`;
        // see `Session::out_anchor` for the EOF-propagation rationale. The
        // pump appends to `replay` *and* publishes on the broadcast inside
        // one critical section — see `Session::attach_snapshot` for the
        // race this closes.
        {
            let reader_res = master.lock().await.try_clone_reader();
            let reader = reader_res.map_err(|e| Status::internal(format!("clone_reader: {e}")))?;
            let replay_for_pump = replay.clone();
            tokio::task::spawn_blocking(move || pump_reader(reader, out_tx, replay_for_pump));
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

        let session_uuid = Uuid::new_v4();
        let session_id = session_uuid.to_string();
        let cwd = if r.cwd.is_empty() { None } else { Some(r.cwd) };
        let meta = SessionMeta {
            session_id: session_id.clone(),
            program: r.program,
            args: r.args,
            cwd,
            opened_at: Utc::now(),
            client_id: if r.client_id.is_empty() {
                None
            } else {
                Some(r.client_id)
            },
            size,
            principal: principal.clone(),
        };

        // Snapshot the audit-event payload before moving `meta` into the
        // session; this avoids cloning the whole struct on the hot path.
        let event = BrainEvent::TerminalSessionOpened {
            id: session_uuid,
            session_id: session_id.clone(),
            program: meta.program.clone(),
            args: meta.args.clone(),
            cwd: meta.cwd.clone(),
            principal: principal.as_ref().map(principal_summary),
            ts: Utc::now(),
        };

        // Graph mirror: write the two open-side nodes and the
        // `tool_call → terminal_event(open)` edge. Failures are logged
        // and the lifecycle proceeds — a missing graph row must never
        // block a live PTY.
        let graph_handles = if let Some(sink) = &self.graph_sink {
            let principal_ref = principal.as_ref();
            match sink
                .record_open(
                    &session_id,
                    &meta.program,
                    &meta.args,
                    meta.cwd.as_deref(),
                    principal_ref,
                )
                .await
            {
                Ok(h) => Some(h),
                Err(e) => {
                    warn!(session_id = %session_id, error = %e, "graph mirror record_open failed");
                    None
                }
            }
        } else {
            None
        };

        let session = Arc::new(Session {
            meta,
            out_anchor,
            replay,
            in_tx,
            master,
            child: Arc::new(Mutex::new(child)),
            graph_handles,
        });
        self.registry.insert(session).await;
        self.publish(event).await;

        Ok(SessionHandle { session_id })
    }

    async fn close_inner(&self, id: &str) -> Result<CloseAck, Status> {
        let session = self
            .registry
            .remove(&id.to_string())
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}' not found")))?;

        let mut child = session.child.lock().await;
        let already_exited = matches!(child.try_wait(), Ok(Some(_)));
        let was_killed = if already_exited {
            false
        } else {
            child.kill().is_ok()
        };
        let exit_code = child.wait().map(|s| s.exit_code() as i32).unwrap_or(-1);
        drop(child);

        // Re-parse the session id so the observe event carries the same
        // `Uuid` we minted at Open (the registry key is its string form).
        let session_uuid = Uuid::parse_str(id).unwrap_or_else(|_| Uuid::new_v4());
        self.publish(BrainEvent::TerminalSessionClosed {
            id: session_uuid,
            session_id: id.to_string(),
            exit_code,
            was_killed,
            principal: session.meta.principal.as_ref().map(principal_summary),
            ts: Utc::now(),
        })
        .await;

        // Graph mirror: write the close-side node + the second
        // `causal_produced` edge. Same fail-soft posture as
        // `record_open`.
        if let (Some(sink), Some(handles)) = (&self.graph_sink, &session.graph_handles) {
            if let Err(e) = sink.record_close(handles, id, exit_code, was_killed).await {
                warn!(session_id = %id, error = %e, "graph mirror record_close failed");
            }
        }

        Ok(CloseAck {
            exit_code,
            was_killed,
        })
    }

    async fn lookup(&self, id: &str) -> Result<Arc<Session>, Status> {
        self.registry
            .get(&id.to_string())
            .await
            .ok_or_else(|| Status::not_found(format!("session '{id}' not found")))
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
        let principal = self
            .authorize(
                &req,
                "open",
                serde_json::json!({"program": req.get_ref().program.as_str()}),
            )
            .await?;
        Ok(Response::new(
            self.open_inner(req.into_inner(), principal).await?,
        ))
    }

    async fn close(&self, req: Request<SessionHandle>) -> Result<Response<CloseAck>, Status> {
        self.authorize(&req, "close", serde_json::Value::Null)
            .await?;
        let id = req.into_inner().session_id;
        Ok(Response::new(self.close_inner(&id).await?))
    }

    async fn attach(
        &self,
        req: Request<SessionHandle>,
    ) -> Result<Response<Self::AttachStream>, Status> {
        self.authorize(&req, "attach", serde_json::Value::Null)
            .await?;
        let id = req.into_inner().session_id;
        let session = self.lookup(&id).await?;

        let (snapshot, mut rx) = session.attach_snapshot();
        let (tx, out) = mpsc::channel::<Result<OutputChunk, Status>>(STREAM_OUT_BUFFER);

        tokio::spawn(async move {
            let mut seq: u64 = 0;
            for bytes in snapshot {
                seq += 1;
                let chunk = OutputChunk {
                    data: bytes.to_vec(),
                    ts: timestamp_now(),
                    seq,
                    eof: false,
                };
                if tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
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
        self.authorize(&req, "send", serde_json::Value::Null)
            .await?;
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
        self.authorize(&req, "resize", serde_json::Value::Null)
            .await?;
        let r = req.into_inner();
        let size = term_size_from_pb(r.size);
        self.resize_inner(&r.session_id, size).await?;
        Ok(Response::new(ResizeAck {}))
    }

    async fn signal(&self, req: Request<SignalRequest>) -> Result<Response<SignalAck>, Status> {
        self.authorize(&req, "signal", serde_json::Value::Null)
            .await?;
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
        // Authorize once at stream start under the umbrella verb
        // `terminal.interact`. Subsequent Open frames inside this stream
        // inherit the resolved principal — we don't re-resolve per frame
        // because the bidi RPC has one set of metadata, set on connect.
        let principal = self
            .authorize(&req, "interact", serde_json::Value::Null)
            .await?;

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
                        match svc.open_inner(open_req, principal.clone()).await {
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
        let (snapshot, mut rx) = session.attach_snapshot();
        let mut seq: u64 = 0;
        for bytes in snapshot {
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
                return;
            }
        }
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
///
/// Each chunk is pushed onto `replay` *and* sent on the broadcast inside
/// one critical section. Attachers take the same lock in
/// `Session::attach_snapshot` to snapshot+resubscribe atomically, which
/// is what stops a fast-exiting child from racing past late attachers.
fn pump_reader(
    mut reader: Box<dyn Read + Send>,
    out_tx: broadcast::Sender<Bytes>,
    replay: Arc<StdMutex<VecDeque<Bytes>>>,
) {
    let mut buf = vec![0u8; PTY_READ_BUFFER_SIZE];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => {
                let chunk = Bytes::copy_from_slice(&buf[..n]);
                let mut guard = replay.lock().expect("replay mutex poisoned");
                if guard.len() == OUT_BROADCAST_CAPACITY {
                    guard.pop_front();
                }
                guard.push_back(chunk.clone());
                // `send` returns Err only when there are zero receivers.
                // We keep pumping in case a new subscriber attaches later;
                // the broadcast keeps the most recent `OUT_BROADCAST_CAPACITY`
                // chunks for them either way.
                let _ = out_tx.send(chunk);
                drop(guard);
            }
            Err(_) => break,
        }
    }
}
