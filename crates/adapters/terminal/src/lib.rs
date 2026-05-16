//! # Brain Terminal Bridge
//!
//! gRPC PTY motor cortex — Brain's adapter for spawning and driving
//! pseudo-terminal sessions across platforms (Unix `openpty`/`forkpty`,
//! Windows ConPTY via `portable-pty`). Each session exposes split RPCs
//! (`Open` / `Close` / `Attach` / `Send` / `Resize` / `Signal`) plus a bidi
//! `Interact` stream for latency-critical use.
//!
//! Default port: **19793**.

use std::{collections::HashMap, sync::Arc};

use identity::IdentityStore;
use observe::Observer;
use tokio::sync::Mutex;

pub mod error;
pub mod graph;
pub mod server;
pub(crate) mod session;
pub mod types;

pub use error::TerminalError;
pub use graph::{MirrorError, TerminalGraphHandles, TerminalGraphSink};
pub use server::TerminalSvc;
pub use types::{SessionId, SessionMeta, TermSize};

use session::Session;

/// Default gRPC bind port for the Terminal Bridge.
pub const DEFAULT_PORT: u16 = 19793;

/// Generated protobuf types and tonic stubs for `brain.terminal.v1`.
pub mod pb {
    tonic::include_proto!("brain.terminal.v1");
}

/// Live PTY session registry. Holds `Arc<Session>` internally; the
/// implementation-detail `Session` shape (master/child handles, channels)
/// is kept crate-private. External callers can only read [`SessionMeta`].
#[derive(Default)]
pub struct SessionRegistry {
    inner: Mutex<HashMap<SessionId, Arc<Session>>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.is_empty()
    }

    pub async fn meta(&self, id: &SessionId) -> Option<SessionMeta> {
        self.inner.lock().await.get(id).map(|s| s.meta.clone())
    }

    pub async fn list(&self) -> Vec<SessionMeta> {
        self.inner
            .lock()
            .await
            .values()
            .map(|s| s.meta.clone())
            .collect()
    }

    pub(crate) async fn get(&self, id: &SessionId) -> Option<Arc<Session>> {
        self.inner.lock().await.get(id).cloned()
    }

    pub(crate) async fn insert(&self, session: Arc<Session>) {
        let id = session.meta.session_id.clone();
        self.inner.lock().await.insert(id, session);
    }

    pub(crate) async fn remove(&self, id: &SessionId) -> Option<Arc<Session>> {
        self.inner.lock().await.remove(id)
    }
}

/// Authentication wiring for the Terminal Bridge. When attached, every RPC
/// resolves a [`identity::Principal`] from request metadata (api-key) and
/// the identity store gates the action. When absent, the bridge runs
/// without authentication — sessions still get a `None` principal in
/// [`SessionMeta`] and audit events, and no gate is enforced.
#[derive(Clone)]
pub struct TerminalAuth {
    pub identity: Arc<dyn IdentityStore>,
    pub api_keys: Arc<Vec<brain_core::ApiKeyConfig>>,
}

impl TerminalAuth {
    pub fn new(identity: Arc<dyn IdentityStore>, api_keys: Vec<brain_core::ApiKeyConfig>) -> Self {
        Self {
            identity,
            api_keys: Arc::new(api_keys),
        }
    }
}

/// Terminal Bridge service handle.
///
/// Holds the [`SessionRegistry`] plus optional [`TerminalAuth`],
/// [`Observer`], and [`TerminalGraphSink`] wiring. Cheap to clone
/// (everything inside is `Arc`-ed).
#[derive(Clone)]
pub struct TerminalBridge {
    sessions: Arc<SessionRegistry>,
    auth: Option<TerminalAuth>,
    observer: Option<Arc<dyn Observer>>,
    graph_sink: Option<Arc<dyn TerminalGraphSink>>,
}

impl TerminalBridge {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(SessionRegistry::new()),
            auth: None,
            observer: None,
            graph_sink: None,
        }
    }

    pub fn sessions(&self) -> &Arc<SessionRegistry> {
        &self.sessions
    }

    /// Attach identity/api-key wiring. Once set, every RPC requires a
    /// resolvable api-key in metadata and a passing
    /// [`IdentityStore::check`] for the corresponding `terminal.*` verb.
    pub fn with_auth(mut self, auth: TerminalAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Attach an [`Observer`] so the bridge can publish
    /// `BrainEvent::TerminalSessionOpened` / `TerminalSessionClosed` on
    /// every session lifecycle transition.
    pub fn with_observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Attach a [`TerminalGraphSink`] so every session lifecycle also
    /// lands `tool_call → terminal_event(open) → terminal_event(close)`
    /// nodes/edges in the episodic graph.
    pub fn with_graph_sink(mut self, sink: Arc<dyn TerminalGraphSink>) -> Self {
        self.graph_sink = Some(sink);
        self
    }

    /// Build the tonic-generated server wrapping a [`TerminalSvc`] backed by
    /// this bridge's registry. Plug into a `tonic::transport::Server::builder`.
    pub fn into_server(self) -> pb::terminal_session_server::TerminalSessionServer<TerminalSvc> {
        pb::terminal_session_server::TerminalSessionServer::new(TerminalSvc::new(
            self.sessions,
            self.auth,
            self.observer,
            self.graph_sink,
        ))
    }

    /// Construct a [`TerminalSvc`] sharing this bridge's wiring, for tests
    /// or callers that want to drive the trait directly without spinning up
    /// a tonic transport.
    pub fn svc(&self) -> TerminalSvc {
        TerminalSvc::new(
            self.sessions.clone(),
            self.auth.clone(),
            self.observer.clone(),
            self.graph_sink.clone(),
        )
    }
}

impl Default for TerminalBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn registry_starts_empty() {
        let bridge = TerminalBridge::new();
        assert!(bridge.sessions().is_empty().await);
        assert_eq!(bridge.sessions().len().await, 0);
    }

    #[test]
    fn default_port_matches_spec() {
        assert_eq!(DEFAULT_PORT, 19793);
    }
}
