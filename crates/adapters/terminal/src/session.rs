//! Internal session state for live PTY sessions.
//!
//! The fields here are implementation detail of the gRPC server — outside
//! callers only see [`crate::SessionMeta`] via [`crate::SessionRegistry::meta`]
//! / [`crate::SessionRegistry::list`].

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};

use bytes::Bytes;
use portable_pty::{Child, MasterPty};
use tokio::sync::{broadcast, mpsc, Mutex};

use crate::graph::TerminalGraphHandles;
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
    /// Bounded replay of PTY output emitted before each attach. The pump
    /// pushes here under `replay`'s lock and *then* sends on the broadcast,
    /// both inside the same critical section. An attacher takes the same
    /// lock to snapshot the buffer and `resubscribe()` at the broadcast's
    /// current tail — so the snapshot ends exactly where the live
    /// subscription begins. Without this, a fast-exiting child can drop
    /// the broadcast sender before any attacher subscribes, and
    /// `Receiver::resubscribe()` (which positions at the current tail,
    /// not the head) gives the late attacher only EOF.
    pub(crate) replay: Arc<StdMutex<VecDeque<Bytes>>>,
    pub(crate) in_tx: mpsc::Sender<Bytes>,
    pub(crate) master: Arc<Mutex<Box<dyn MasterPty + Send>>>,
    pub(crate) child: Arc<Mutex<Box<dyn Child + Send + Sync>>>,
    /// Graph node handles produced at session open; consulted by
    /// `close_inner` to wire the close-side edge back to the open
    /// event. `None` when the bridge runs without a graph sink.
    pub(crate) graph_handles: Option<TerminalGraphHandles>,
}

impl Session {
    /// Atomic "give me everything so far, then start streaming" handle for
    /// `Attach` / `Interact`. Returns a snapshot of past chunks and a new
    /// subscriber that will receive every chunk produced after the snapshot.
    pub(crate) fn attach_snapshot(&self) -> (Vec<Bytes>, broadcast::Receiver<Bytes>) {
        let buf = self.replay.lock().expect("replay mutex poisoned");
        let snapshot: Vec<Bytes> = buf.iter().cloned().collect();
        let rx = self.out_anchor.resubscribe();
        (snapshot, rx)
    }
}
