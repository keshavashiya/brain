//! Chat commands — interactive and non-interactive conversation modes.
//!
//! Uses WebSocket for communication with the daemon, enabling lower latency
//! and a consistent protocol across all adapters.
//!
//! The module is split by concern: [`signals`] (in-chat slash commands),
//! [`render`] (terminal/markdown rendering), [`frames`] (one-shot WS frame
//! accumulation), [`transport`] (connect/auth/send over WebSocket), and
//! [`reader`] (the interactive two-half loop).

mod frames;
mod reader;
mod render;
mod signals;
mod transport;

#[cfg(test)]
mod tests;

pub(crate) use reader::chat_interactive;
pub(crate) use signals::signal_catalog;
pub(crate) use transport::{chat_non_interactive, command_over_chat};
