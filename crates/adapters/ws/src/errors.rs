//! WebSocket adapter error type.

#[derive(Debug, thiserror::Error)]
pub enum WsAdapterError {
    #[error("WebSocket error: {0}")]
    Ws(String),
    #[error("Server error: {0}")]
    Server(String),
}
