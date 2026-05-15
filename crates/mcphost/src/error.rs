//! MCP host error taxonomy.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpHostError {
    #[error("server '{0}' is already mounted")]
    AlreadyMounted(String),

    #[error("server '{0}' not mounted")]
    NotMounted(String),

    #[error("initialize failed: {0}")]
    Initialize(String),

    #[error("protocol version negotiation failed: got {0}")]
    Version(String),

    #[error("tool '{0}' not found on server '{1}'")]
    NoTool(String, String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("auth error: {0}")]
    Auth(String),

    #[error("rmcp error: {0}")]
    Rmcp(String),

    #[error("identity gate denied: {0}")]
    Denied(String),
}
