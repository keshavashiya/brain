//! # Brain Terminal Bridge
//!
//! Phase 2 of the v1.0.0 plan (`docs/v1.0.0.md` §3.1) — Brain's motor cortex
//! for spawning and driving PTY sessions over gRPC.
//!
//! Default port: **19793** (sibling of the 19792 Memory/Agent service).
//!
//! ## Status
//!
//! - PR11: `.proto` + tonic stubs + portable-pty dep + registry shell.
//! - **PR12 (current):** `Open` / `Close` / `Attach` RPCs implemented
//!   against `portable-pty 0.9` — PTY reader thread → `broadcast<Bytes>`,
//!   mpsc → writer thread (writer is wired but unused until PR13).
//! - PR13: `Send` / `Resize` / `Signal` + bidi `Interact` perf path.
//! - PR14: Principal threading + per-session audit events.
//! - PR15: Thalamus intents (`OpenTerminalSession` / `ListTerminalSessions`
//!   / `CloseTerminalSession`).

use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

pub mod error;
pub mod server;
pub(crate) mod session;
pub mod types;

pub use error::TerminalError;
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

/// Terminal Bridge service handle.
///
/// Holds the [`SessionRegistry`] and constructs a [`TerminalSvc`] tonic
/// server on demand. Cheap to clone (the registry is `Arc`-ed).
#[derive(Clone)]
pub struct TerminalBridge {
    sessions: Arc<SessionRegistry>,
}

impl TerminalBridge {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(SessionRegistry::new()),
        }
    }

    pub fn sessions(&self) -> &Arc<SessionRegistry> {
        &self.sessions
    }

    /// Build the tonic-generated server wrapping a [`TerminalSvc`] backed by
    /// this bridge's registry. Plug into a `tonic::transport::Server::builder`.
    pub fn into_server(self) -> pb::terminal_session_server::TerminalSessionServer<TerminalSvc> {
        pb::terminal_session_server::TerminalSessionServer::new(TerminalSvc::new(self.sessions))
    }

    /// Construct a [`TerminalSvc`] sharing this bridge's registry, for tests
    /// or callers that want to drive the trait directly without spinning up
    /// a tonic transport.
    pub fn svc(&self) -> TerminalSvc {
        TerminalSvc::new(self.sessions.clone())
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
