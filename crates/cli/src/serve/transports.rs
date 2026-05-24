//! Preset-driven transports + channel-relay wiring for `cmd_serve`.
//!
//! Two complementary delivery surfaces share this module:
//!
//! - **Preset transports** (`channel.transports[]`): YAML-defined
//!   HttpPolled / WebhookInbound / WebhookOutbound transports the
//!   dispatcher uses for outbound proactive messages and that, for
//!   inbound kinds, feed messages back through the signal pipeline
//!   via [`forward_inbound_to_signal`]. The [`wire_preset_transports`]
//!   helper builds, registers, and spawns each entry.
//!
//! - **Channel relays** (`channel.relays[]`): WS bridge endpoints that
//!   adapt remote signals into the pipeline via the small
//!   [`RelayPipelineHandler`] fallback (used by the relay adapter
//!   when no correlation match wins).

use std::sync::Arc;

use async_trait::async_trait;
use bridge::BridgeMessage;
use channel::relay::SignalHandler;

/// Bridge-facing handler that forwards non-correlation messages into the
/// main signal pipeline. Used by each configured relay adapter so replies
/// arriving on any external transport hit the same pipeline as HTTP/WS
/// traffic.
struct RelayPipelineHandler {
    processor: Arc<signal::SignalProcessor>,
    channel_id: String,
    namespace: String,
}

#[async_trait]
impl SignalHandler for RelayPipelineHandler {
    async fn handle(&self, msg: &BridgeMessage) -> String {
        let sender = msg.source.clone().unwrap_or_else(|| "relay".to_string());
        let sig = signal::Signal::new(
            signal::SignalSource::WebSocket,
            &self.channel_id,
            sender,
            &msg.content,
        )
        .with_namespace(&self.namespace);
        match self.processor.process(sig).await {
            Ok(resp) => signal::response_to_text(&resp.response),
            Err(e) => format!("error: {e}"),
        }
    }
}

/// Pipe an inbound transport message into the main signal pipeline with the
/// same shape the relay handler uses, then route the pipeline's response
/// back to the originating channel via the dispatcher. Without the
/// round-trip, replies to inbound Telegram/etc. messages would be silently
/// dropped (the request side has no live WS adapter to write back to).
async fn forward_inbound_to_signal(
    msg: channel::InboundMessage,
    processor: Arc<signal::SignalProcessor>,
    dispatcher: Arc<channel::ChannelDispatcher>,
    channel_id: String,
    namespace: String,
) {
    let sender = msg.user_ref.clone().unwrap_or_else(|| "transport".into());
    let reply_to = msg.reply_to.clone();
    let sig = signal::Signal::new(
        signal::SignalSource::WebSocket,
        &channel_id,
        sender,
        &msg.content,
    )
    .with_namespace(&namespace);
    let resp = match processor.process(sig).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(channel = %channel_id, error = %e, "Transport inbound pipeline error");
            return;
        }
    };

    let body = signal::response_to_text(&resp.response);
    if body.trim().is_empty() {
        return;
    }
    let mut intent = channel::DeliveryIntent::new(
        body,
        channel::DeliveryCategory::Response,
        channel::UrgencyLevel::Normal,
    )
    .with_namespace(&namespace)
    .with_preferred(&channel_id)
    .with_initiation(&channel_id);
    if let Some(rt) = reply_to {
        intent = intent.with_metadata("reply_to", rt);
    }
    if let Err(e) = dispatcher.dispatch(intent).await {
        tracing::warn!(channel = %channel_id, error = %e, "Failed to route inbound response back to source");
    }
}

/// Build preset-driven transports, register each with the dispatcher (so
/// it can actually deliver), spawn polling loops for HttpPolled kinds,
/// and feed inbound messages into the signal pipeline (after the
/// correlator has had a chance to claim them).
pub(super) async fn wire_preset_transports(
    entries: &[brain::config::TransportEntry],
    processor: Arc<signal::SignalProcessor>,
    dispatcher: Arc<channel::ChannelDispatcher>,
    correlator: Arc<channel::ConfirmationCorrelator>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) -> (
    Vec<tokio::sync::oneshot::Sender<()>>,
    std::collections::HashMap<String, Arc<channel::transport::inbound::WebhookInboundTransport>>,
) {
    use channel::transport::inbound::{WebhookInboundConfig, WebhookInboundTransport};
    use channel::transport::outbound::{WebhookOutboundConfig, WebhookOutboundTransport};
    use channel::transport::polled::{HttpPolledConfig, HttpPolledTransport};
    use channel::transport::preset::{self, PresetKind};
    use channel::ChannelTransport;

    let mut shutdown_guards = Vec::new();
    let mut webhook_handlers = std::collections::HashMap::new();

    for entry in entries {
        let preset = match preset::load(&entry.preset) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    transport = %entry.id,
                    preset = %entry.preset,
                    error = %e,
                    "Preset load failed — skipping transport"
                );
                continue;
            }
        };

        match preset.kind {
            PresetKind::HttpPolled => {
                let cfg = HttpPolledConfig::new(
                    &entry.id,
                    &entry.label,
                    preset.clone(),
                    &entry.credential,
                );
                let transport = match HttpPolledTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "HttpPolled init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, polled)",
                    entry.label, entry.preset
                );

                let rx = transport.inbound();
                let proc_clone = processor.clone();
                let disp_clone = dispatcher.clone();
                let chan_id = entry.id.clone();
                let ns = entry.namespace.clone();
                let corr = correlator.clone();
                set.spawn(async move {
                    let mut rx = rx;
                    while let Ok(msg) = rx.recv().await {
                        match corr.process(&msg.content).await {
                            Ok(channel::CorrelationOutcome::NoMatch) => {}
                            Ok(_) => continue,
                            Err(e) => {
                                tracing::debug!(error = %e, "correlator error — forwarding raw");
                            }
                        }
                        forward_inbound_to_signal(
                            msg,
                            proc_clone.clone(),
                            disp_clone.clone(),
                            chan_id.clone(),
                            ns.clone(),
                        )
                        .await;
                    }
                    Ok(())
                });

                let run_transport = transport.clone();
                let (sd_tx, sd_rx) = tokio::sync::oneshot::channel::<()>();
                shutdown_guards.push(sd_tx);
                set.spawn(async move {
                    run_transport.run(sd_rx).await;
                    Ok(())
                });
            }
            PresetKind::WebhookOutbound => {
                let cfg = WebhookOutboundConfig::new(&entry.id, &entry.label, preset.clone())
                    .with_credential(&entry.credential);
                let transport = match WebhookOutboundTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "WebhookOutbound init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, outbound-only)",
                    entry.label, entry.preset
                );
            }
            PresetKind::WebhookInbound => {
                let mut cfg = WebhookInboundConfig::new(&entry.id, &entry.label, preset.clone());
                if !entry.credential.is_empty() {
                    cfg = cfg.with_credential(&entry.credential);
                }
                if let Some(secret) = &entry.signing_secret {
                    cfg = cfg.with_signing_secret(secret);
                }
                let transport = match WebhookInboundTransport::new(cfg) {
                    Ok(t) => Arc::new(t),
                    Err(e) => {
                        tracing::warn!(
                            transport = %entry.id, error = %e, "WebhookInbound init failed"
                        );
                        continue;
                    }
                };
                if let Err(e) = dispatcher
                    .register_transport(transport.clone() as Arc<dyn ChannelTransport>)
                    .await
                {
                    tracing::warn!(
                        transport = %entry.id, error = %e,
                        "dispatcher register failed",
                    );
                }
                println!(
                    "  Transport    → {} (preset: {}, webhook-inbound)",
                    entry.label, entry.preset
                );
                webhook_handlers.insert(entry.id.clone(), transport);
            }
        }
    }

    (shutdown_guards, webhook_handlers)
}

/// Wire each configured `channel.relays[]` entry as a `RelayAdapter`,
/// registering its channel with the router and spawning its bridge
/// loop. Each adapter falls back to [`RelayPipelineHandler`] when no
/// correlator claims an inbound message — that drives the legacy
/// "relay reply hits the pipeline" path.
pub(super) async fn wire_channel_relays(
    relays: &[brain::config::RelayEntry],
    processor: &Arc<signal::SignalProcessor>,
    router: Arc<dyn channel::ChannelRouter>,
    correlator: Arc<channel::ConfirmationCorrelator>,
    prefs: Arc<dyn channel::ChannelPreferenceStore>,
    set: &mut tokio::task::JoinSet<anyhow::Result<()>>,
) {
    for entry in relays {
        let bridge_cfg = bridge::BridgeConfig {
            initial_backoff_ms: entry.initial_backoff_ms,
            max_backoff_ms: entry.max_backoff_ms,
            max_reconnect_attempts: None,
        };
        let relay_cfg = channel::RelayConfig::new(&entry.id, &entry.label, &entry.url)
            .with_namespace(&entry.namespace)
            .with_bridge(bridge_cfg);
        let fallback: Arc<dyn SignalHandler> = Arc::new(RelayPipelineHandler {
            processor: processor.clone(),
            channel_id: entry.id.clone(),
            namespace: entry.namespace.clone(),
        });
        let adapter = Arc::new(channel::RelayAdapter::new(
            relay_cfg,
            router.clone(),
            correlator.clone(),
            prefs.clone(),
            fallback,
        ));
        if let Err(e) = adapter.register_channel().await {
            tracing::warn!(
                channel = %entry.id,
                "Relay channel registration failed (non-fatal): {e}"
            );
        } else {
            println!("  Relay        → {} ({})", entry.label, entry.url);
        }
        let a = adapter.clone();
        set.spawn(async move {
            if let Err(e) = a.run().await {
                tracing::warn!("Relay adapter exited: {e}");
            }
            Ok(())
        });
    }
}
