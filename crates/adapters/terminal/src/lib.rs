//! # Brain Terminal Bridge
//!
//! Phase 2 of the v1.0.0 plan (`docs/v1.0.0.md` §3.1) — Brain's motor cortex
//! for spawning and driving PTY sessions over gRPC.
//!
//! Default port: **19793** (sibling of the 19792 Memory/Agent service).
//!
//! ## Scope of this skeleton (PR11)
//!
//! - `.proto` definition (`proto/terminal.proto`) and generated tonic stubs.
//! - Cross-platform PTY backend dep (`portable-pty 0.9+`, ConPTY on Windows).
//! - Public error / handle / size types and the `TerminalBridge` struct
//!   holding the session registry. Trait-shaped but no RPCs implemented yet.
//!
//! Open/Close/Attach come in PR12; Send/Resize/Signal/Interact in PR13;
//! Principal threading and per-session audit events in PR14;
//! Thalamus intents (`OpenTerminalSession` / `ListTerminalSessions` /
//! `CloseTerminalSession`) in PR15.

use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

pub mod error;
pub mod types;

pub use error::TerminalError;
pub use types::{SessionId, SessionMeta, TermSize};

/// Default gRPC bind port for the Terminal Bridge.
pub const DEFAULT_PORT: u16 = 19793;

/// Generated protobuf types and tonic stubs for `brain.terminal.v1`.
pub mod pb {
    tonic::include_proto!("brain.terminal.v1");
}

/// Live PTY session registry. RPC handlers will mutate this in PR12+.
///
/// Kept opaque on purpose — the concrete `Session` shape (master/child
/// handles, broadcast/mpsc channels) is implementation detail of the
/// RPC layer and lands with PR12.
#[derive(Default)]
pub struct SessionRegistry {
    inner: Mutex<HashMap<SessionId, SessionMeta>>,
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
        self.inner.lock().await.get(id).cloned()
    }

    pub async fn list(&self) -> Vec<SessionMeta> {
        self.inner.lock().await.values().cloned().collect()
    }
}

/// Terminal Bridge service handle.
///
/// PR12 will implement `pb::terminal_session_server::TerminalSession` on this
/// type. For PR11 it carries the registry and any wiring (observer, identity
/// store) the RPC handlers will need.
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
