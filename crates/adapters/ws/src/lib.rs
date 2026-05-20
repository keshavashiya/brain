//! # Brain WebSocket Adapter
//!
//! Exposes Brain's signal processing pipeline over WebSocket using tokio-tungstenite.
//!
//! ## Protocol
//! 1. Client connects (WebSocket handshake).
//! 2. Client sends first text frame: `{"api_key":"<key>"}` — authentication.
//! 3. Server replies with `{"status":"authenticated","conn_id":"<uuid>"}` or
//!    `{"status":"error","message":"..."}` then closes.
//! 4. Subsequent text frames are `SignalRequest` JSON; server replies with
//!    `SignalResponse` JSON.
//!
//! ## Authentication
//! The initial handshake message MUST contain a valid `api_key`.
//! If the key is absent or invalid the server sends an error frame and closes.

mod auth;
mod chat_transport;
mod connection;
mod errors;
mod protocol;
mod server;
mod streaming;

pub use errors::WsAdapterError;
pub use protocol::{AuthMessage, AuthResponse, ClientMessage, ConnectionInfo, Connections};
pub use server::serve;

#[cfg(test)]
mod tests;
