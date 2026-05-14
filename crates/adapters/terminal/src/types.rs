//! Public, non-PB types for the Terminal Bridge.
//!
//! The protobuf-generated `pb::*` types are tied to gRPC encoding and aren't
//! ergonomic to pass around the rest of the workspace (intents, audit events,
//! observers). These mirrored types are what flows through Brain.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Stable string handle for a live PTY session (UUID v4).
pub type SessionId = String;

/// PTY dimensions. Mirrors `pb::PtySize` but carries `u16` (the native
/// `portable_pty::PtySize` width) instead of proto's `uint32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TermSize {
    pub rows: u16,
    pub cols: u16,
    pub pixel_width: u16,
    pub pixel_height: u16,
}

impl Default for TermSize {
    fn default() -> Self {
        Self {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        }
    }
}

/// Lightweight snapshot of a session for `ListTerminalSessions` and audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: SessionId,
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub opened_at: DateTime<Utc>,
    pub client_id: Option<String>,
    pub size: TermSize,
}
