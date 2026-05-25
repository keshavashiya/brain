//! # Brain Bridge
//!
//! External service relay — WebSocket client that connects Brain to a
//! remote messaging gateway and relays inbound messages through Brain's
//! signal processing pipeline.
//!
//! ## Protocol
//!
//! 1. [`BridgeClient`] connects to the configured `url` via WebSocket.
//! 2. Each inbound text frame is parsed as a JSON-encoded [`BridgeMessage`].
//!    Unparseable text frames are logged and skipped; the connection stays
//!    open. Binary frames are silently ignored.
//! 3. WebSocket `Ping` frames are answered with `Pong` automatically.
//!    `Close` frames trigger the reconnect loop after the configured
//!    backoff.
//! 4. A caller-supplied `handler` is invoked for each parsed message and
//!    must return a [`BridgeMessage`] response (the response keeps the
//!    original message's `id` if built via [`BridgeMessage::reply`]).
//! 5. The response is serialised and sent back as a text frame. Failure
//!    to send tears down the connection and triggers the reconnect loop;
//!    serialisation failure logs and drops the response only.
//! 6. On disconnect, the client waits `initial_backoff_ms * 2^attempt`
//!    (capped at `max_backoff_ms`) before retrying. With
//!    [`BridgeConfig::max_reconnect_attempts`] set, the loop returns
//!    [`BridgeError::MaxRetriesExceeded`] once the cap is reached; with
//!    `None`, retries continue indefinitely.
//!
//! [`BridgeClient::connect_and_relay_bidirectional`] adds an optional
//! proactive-push channel: each [`BridgeMessage`] received on
//! `proactive_rx` is serialised and pushed to the gateway alongside the
//! normal request/response flow. If the broadcast channel lags, lagged
//! events are logged and dropped; if it closes, the client continues in
//! relay-only mode.
//!
//! ## Usage
//! ```no_run
//! # use brainos_bridge::{BridgeClient, BridgeConfig, BridgeMessage};
//! # #[tokio::main] async fn main() -> anyhow::Result<()> {
//! let client = BridgeClient::new("ws://gateway.example.com/brain", BridgeConfig::default());
//! client.connect_and_relay(|msg| async move {
//!     BridgeMessage::reply(&msg, format!("Echo: {}", msg.content))
//! }).await?;
//! # Ok(())
//! # }
//! ```

use std::{collections::HashMap, future::Future, time::Duration};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Connection failed after {0} attempts")]
    MaxRetriesExceeded(u32),
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// Configuration for [`BridgeClient`] reconnection behaviour.
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Base delay used to compute backoff (milliseconds). Default: 1 000 ms.
    /// The actual wait before retry `n` (counting from 0 after the most
    /// recent successful connection) is
    /// `min(initial_backoff_ms * 2^n, max_backoff_ms)`.
    pub initial_backoff_ms: u64,
    /// Upper bound on the per-retry backoff delay (milliseconds).
    /// Default: 60 000 ms.
    pub max_backoff_ms: u64,
    /// Maximum number of *consecutive* failed reconnect attempts before
    /// the relay loop gives up with [`BridgeError::MaxRetriesExceeded`].
    /// The counter resets to zero on every successful WebSocket connect,
    /// so a flaky link can still recover indefinitely as long as it
    /// occasionally succeeds. `None` (the default) means keep retrying
    /// forever.
    pub max_reconnect_attempts: Option<u32>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            initial_backoff_ms: 1_000,
            max_backoff_ms: 60_000,
            max_reconnect_attempts: None,
        }
    }
}

/// A message exchanged between Brain and the remote gateway.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BridgeMessage {
    /// Unique message ID (UUID v4 string).
    pub id: String,
    /// Text content of the message.
    pub content: String,
    /// Optional source label set by the gateway (transport id, channel
    /// name, sender id — whatever the gateway forwards).
    pub source: Option<String>,
    /// Arbitrary key-value metadata forwarded from the gateway.
    pub metadata: Option<HashMap<String, String>>,
}

impl BridgeMessage {
    /// Create a new outbound message with a fresh UUID.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.into(),
            source: None,
            metadata: None,
        }
    }

    /// Create a reply to `original`, reusing the same `id` for correlation.
    pub fn reply(original: &BridgeMessage, content: impl Into<String>) -> Self {
        Self {
            id: original.id.clone(),
            content: content.into(),
            source: None,
            metadata: None,
        }
    }
}

// ─── BridgeClient ─────────────────────────────────────────────────────────────

/// WebSocket client that relays messages between Brain and a remote gateway.
pub struct BridgeClient {
    url: String,
    config: BridgeConfig,
}

impl BridgeClient {
    /// Create a new client pointing at `url`.
    pub fn new(url: impl Into<String>, config: BridgeConfig) -> Self {
        Self {
            url: url.into(),
            config,
        }
    }

    /// Calculate the backoff duration for a given retry attempt number (0-indexed).
    ///
    /// Uses exponential back-off: `initial * 2^attempt`, capped at `max`.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        // 2^attempt, capped to avoid overflow
        let multiplier = 1u64.checked_shl(attempt.min(62)).unwrap_or(u64::MAX);
        let ms = self
            .config
            .initial_backoff_ms
            .saturating_mul(multiplier)
            .min(self.config.max_backoff_ms);
        Duration::from_millis(ms)
    }

    /// Connect to the remote WebSocket gateway and relay messages indefinitely.
    ///
    /// For each inbound [`BridgeMessage`], `handler` is called and its return
    /// value is sent back as a text frame.  On disconnect the client waits for
    /// the appropriate backoff period then reconnects automatically.
    ///
    /// Returns `Err(BridgeError::MaxRetriesExceeded)` only when
    /// [`BridgeConfig::max_reconnect_attempts`] is set and exceeded.
    pub async fn connect_and_relay<F, Fut>(&self, handler: F) -> Result<(), BridgeError>
    where
        F: Fn(BridgeMessage) -> Fut + Clone,
        Fut: Future<Output = BridgeMessage>,
    {
        self.connect_and_relay_bidirectional(handler, None).await
    }

    /// Connect with optional proactive push channel.
    ///
    /// When `proactive_rx` is provided, proactive notifications are forwarded
    /// to the gateway as outbound `BridgeMessage` frames alongside the normal
    /// request-response relay.
    pub async fn connect_and_relay_bidirectional<F, Fut>(
        &self,
        handler: F,
        mut proactive_rx: Option<tokio::sync::broadcast::Receiver<BridgeMessage>>,
    ) -> Result<(), BridgeError>
    where
        F: Fn(BridgeMessage) -> Fut + Clone,
        Fut: Future<Output = BridgeMessage>,
    {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::{connect_async, tungstenite::Message};

        let mut attempt = 0u32;

        loop {
            // Check retry limit before sleeping
            if let Some(max) = self.config.max_reconnect_attempts {
                if attempt >= max {
                    return Err(BridgeError::MaxRetriesExceeded(attempt));
                }
            }

            // Backoff before retries (not before the first attempt)
            if attempt > 0 {
                let backoff = self.backoff_duration(attempt - 1);
                tracing::info!(
                    url = %self.url,
                    attempt,
                    backoff_ms = backoff.as_millis(),
                    "Reconnecting to bridge gateway"
                );
                tokio::time::sleep(backoff).await;
            }

            tracing::info!(url = %self.url, "Connecting to bridge gateway");

            let ws_stream = match connect_async(&self.url).await {
                Err(e) => {
                    tracing::warn!(url = %self.url, error = %e, "Bridge connection failed");
                    attempt += 1;
                    continue;
                }
                Ok((ws, _response)) => {
                    tracing::info!(url = %self.url, "Bridge connected");
                    attempt = 0; // reset on successful connection
                    ws
                }
            };

            let (mut sink, mut stream) = ws_stream.split();

            loop {
                tokio::select! {
                    ws_msg = stream.next() => {
                        match ws_msg {
                            None => {
                                tracing::warn!(url = %self.url, "Bridge stream ended (EOF)");
                                break;
                            }
                            Some(Err(e)) => {
                                tracing::warn!(url = %self.url, error = %e, "Bridge WebSocket error");
                                break;
                            }
                            Some(Ok(Message::Ping(data))) => {
                                if sink.send(Message::Pong(data)).await.is_err() {
                                    break;
                                }
                            }
                            Some(Ok(Message::Close(_))) => {
                                tracing::info!(url = %self.url, "Bridge connection closed by remote");
                                break;
                            }
                            Some(Ok(Message::Text(text))) => {
                                let msg: BridgeMessage = match serde_json::from_str(&text) {
                                    Ok(m) => m,
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            raw = %text,
                                            "Ignoring unparseable bridge message"
                                        );
                                        continue;
                                    }
                                };

                                let msg_id = msg.id.clone();
                                let response = handler.clone()(msg).await;

                                match serde_json::to_string(&response) {
                                    Ok(payload) => {
                                        if sink.send(Message::Text(payload.into())).await.is_err() {
                                            tracing::warn!(id = %msg_id, "Failed to send bridge response");
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(id = %msg_id, error = %e, "Failed to serialise response");
                                    }
                                }
                            }
                            Some(Ok(_)) => {} // ignore binary frames and pong
                        }
                    }
                    proactive = async {
                        match proactive_rx.as_mut() {
                            Some(rx) => rx.recv().await,
                            None => std::future::pending().await,
                        }
                    } => {
                        match proactive {
                            Ok(msg) => {
                                if let Ok(payload) = serde_json::to_string(&msg) {
                                    if sink.send(Message::Text(payload.into())).await.is_err() {
                                        tracing::warn!("Failed to push proactive to bridge");
                                        break;
                                    }
                                    tracing::debug!(id = %msg.id, "Proactive pushed to bridge");
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!(skipped = n, "Bridge proactive receiver lagged");
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                tracing::info!("Proactive channel closed, bridge continues in relay-only mode");
                                proactive_rx = None;
                            }
                        }
                    }
                }
            }

            attempt += 1;
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_config_defaults() {
        let cfg = BridgeConfig::default();
        assert_eq!(cfg.initial_backoff_ms, 1_000);
        assert_eq!(cfg.max_backoff_ms, 60_000);
        assert!(cfg.max_reconnect_attempts.is_none());
    }

    #[test]
    fn test_backoff_grows_exponentially() {
        let client = BridgeClient::new("ws://unused", BridgeConfig::default());
        // attempt 0: 1000 ms
        assert_eq!(client.backoff_duration(0), Duration::from_millis(1_000));
        // attempt 1: 2000 ms
        assert_eq!(client.backoff_duration(1), Duration::from_millis(2_000));
        // attempt 2: 4000 ms
        assert_eq!(client.backoff_duration(2), Duration::from_millis(4_000));
        // attempt 3: 8000 ms
        assert_eq!(client.backoff_duration(3), Duration::from_millis(8_000));
    }

    #[test]
    fn test_backoff_capped_at_max() {
        let cfg = BridgeConfig {
            initial_backoff_ms: 1_000,
            max_backoff_ms: 5_000,
            max_reconnect_attempts: None,
        };
        let client = BridgeClient::new("ws://unused", cfg);
        // After enough doublings it should saturate at 5000
        assert_eq!(client.backoff_duration(10), Duration::from_millis(5_000));
        assert_eq!(client.backoff_duration(30), Duration::from_millis(5_000));
    }

    #[test]
    fn test_bridge_message_new() {
        let msg = BridgeMessage::new("hello");
        assert!(!msg.id.is_empty());
        assert_eq!(msg.content, "hello");
        assert!(msg.source.is_none());
        assert!(msg.metadata.is_none());
    }

    #[test]
    fn test_bridge_message_reply_shares_id() {
        let original = BridgeMessage::new("what time is it?");
        let reply = BridgeMessage::reply(&original, "It is noon.");
        assert_eq!(
            reply.id, original.id,
            "reply should reuse original message ID"
        );
        assert_eq!(reply.content, "It is noon.");
    }

    #[test]
    fn test_bridge_message_roundtrip_json() {
        let mut meta = HashMap::new();
        meta.insert("channel".to_string(), "#general".to_string());
        let original = BridgeMessage {
            id: "test-id-123".to_string(),
            content: "Deploy to prod?".to_string(),
            source: Some("chat-main".to_string()),
            metadata: Some(meta),
        };
        let json = serde_json::to_string(&original).unwrap();
        let decoded: BridgeMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, original);
    }

    #[tokio::test]
    async fn test_connect_fails_and_hits_max_retries() {
        // Point at an address that will refuse connections immediately.
        let cfg = BridgeConfig {
            initial_backoff_ms: 1, // tiny backoff so the test is fast
            max_backoff_ms: 1,
            max_reconnect_attempts: Some(3),
        };
        let client = BridgeClient::new("ws://127.0.0.1:19999", cfg);

        let result = client
            .connect_and_relay(|msg| async move { BridgeMessage::reply(&msg, "ok") })
            .await;

        assert!(
            matches!(result, Err(BridgeError::MaxRetriesExceeded(3))),
            "expected MaxRetriesExceeded(3), got: {:?}",
            result
        );
    }
}
