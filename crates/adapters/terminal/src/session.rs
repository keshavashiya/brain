//! Internal session state for live PTY sessions.
//!
//! The fields here are implementation detail of the gRPC server — outside
//! callers only see [`crate::SessionMeta`] via [`crate::SessionRegistry::meta`]
//! / [`crate::SessionRegistry::list`]. PR13 will start using `in_tx` and
//! `master` for Send/Resize.

use std::sync::Arc;

use bytes::Bytes;
use portable_pty::{Child, MasterPty};
use tokio::sync::{broadcast, mpsc, Mutex};

use crate::types::SessionMeta;

/// Capacity for the per-session output broadcast channel. Late subscribers
/// will get `Lagged` if they fall this many chunks behind; that's surfaced
/// as a stream error to the client rather than silent loss.
pub(crate) const OUT_BROADCAST_CAPACITY: usize = 256;

/// Capacity of the in-process input mpsc. Small bound applies natural
/// backpressure to writers that stream faster than the PTY drains.
pub(crate) const IN_MPSC_CAPACITY: usize = 64;

/// One live PTY session. Held inside [`crate::SessionRegistry`] as `Arc<Session>`
/// so the resubscribe handle / mpsc sender can be cloned out for streaming
/// without holding the registry lock.
///
/// **Note on `out_anchor`:** we deliberately keep *only* a `Receiver`, not a
/// `Sender`. The single `broadcast::Sender<Bytes>` lives on the PTY reader
/// pump task; when the PTY hits EOF, that task drops the sender and every
/// active `Attach` subscriber observes `RecvError::Closed`, which is how
/// end-of-output propagates to clients. If `Session` also held a `Sender`
/// clone, the channel would stay open past the child's exit forever.
pub(crate) struct Session {
    pub(crate) meta: SessionMeta,
    pub(crate) out_anchor: broadcast::Receiver<Bytes>,
    #[allow(dead_code)] // Used by PR13's Send/Resize/Interact.
    pub(crate) in_tx: mpsc::Sender<Bytes>,
    #[allow(dead_code)] // Used by PR13's Resize and future signal handling.
    pub(crate) master: Arc<Mutex<Box<dyn MasterPty + Send>>>,
    pub(crate) child: Arc<Mutex<Box<dyn Child + Send + Sync>>>,
}
