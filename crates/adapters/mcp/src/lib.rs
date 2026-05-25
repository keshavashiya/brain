//! # Brain MCP Adapter
//!
//! Exposes Brain's memory tools as an MCP (Model Context Protocol) server.
//!
//! ## Transports
//!
//! The crate ships two transport entry points:
//!
//! - [`serve_http`]: JSON-RPC over HTTP POST on the configured MCP port
//!   (default 19791). The daemon runs this when `adapters.mcp.enabled =
//!   true` is set in config; it is what `brain mcp` proxies into.
//! - [`serve_stdio`]: line-delimited JSON-RPC on stdin/stdout. Exposed
//!   for library consumers that want to surface MCP from their own
//!   process with their own [`signal::SignalProcessor`].
//!
//! The `brain mcp` CLI subcommand intentionally does *not* invoke
//! [`serve_stdio`] directly — it forwards stdin/stdout to the daemon's
//! HTTP MCP port instead, so a single shared `SignalProcessor` (and one
//! passphrase prompt, one ruvector lock, etc.) services every MCP
//! client that spawns the binary.
//!
//! ## Authentication
//! - **HTTP**: `x-api-key: <key>` HTTP header on every request.
//! - **stdio**: either `BRAIN_API_KEY` env var (session-level, recommended for
//!   MCP clients) or `params._meta["x-api-key"]` per request.
//!   If no API keys are configured, auth is skipped.
//!
//! ## Tools
//! - `memory_search`     — semantic search over stored facts/episodes
//! - `memory_store`      — store a structured fact (subject predicate object)
//! - `memory_facts`      — get all facts about a subject
//! - `memory_episodes`   — get recent conversation episodes
//! - `user_profile`      — return user profile / config data
//! - `memory_procedures` — manage learned workflows (list / store / delete)
//!
//! ## MCP client config (stdio transport)
//! ```json
//! {
//!   "mcpServers": {
//!     "brain": {
//!       "command": "brain",
//!       "args": ["mcp"],
//!       "env": {
//!         "BRAIN_API_KEY": "<your-api-key>"
//!       }
//!     }
//!   }
//! }
//! ```

mod errors;
mod protocol;
mod server;
mod tools;
mod transport;

pub use errors::McpError;
pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use server::McpServer;
pub use transport::{serve_http, serve_stdio};

#[cfg(test)]
mod tests;
