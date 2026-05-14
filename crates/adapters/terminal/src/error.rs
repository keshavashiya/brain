//! Terminal Bridge error type.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TerminalError {
    #[error("session '{0}' not found")]
    NotFound(String),

    #[error("PTY backend error: {0}")]
    Pty(String),

    #[error("spawn failed: {0}")]
    Spawn(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("identity gate denied: {0}")]
    Denied(String),
}
