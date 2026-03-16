//! Bridge command — relay messages between Brain and an external gateway.

pub(crate) async fn cmd_bridge(
    config: &brain_core::BrainConfig,
    gateway_url: &str,
    api_key: Option<&str>,
) -> anyhow::Result<()> {
    use bridge::{BridgeClient, BridgeConfig, BridgeMessage};
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let api_key = api_key
        .map(String::from)
        .or_else(|| config.access.api_keys.first().map(|k| k.key.clone()))
        .ok_or_else(|| anyhow::anyhow!("No API key configured. Add one in ~/.brain/config.yaml"))?;

    let brain_ws_url = format!(
        "ws://{}:{}/ws",
        config.adapters.http.host, config.adapters.ws.port
    );

    println!("Brain Bridge");
    println!("  → Brain:   {}", brain_ws_url);
    println!("  → Gateway: {}", gateway_url);
    println!();

    let bridge_client = BridgeClient::new(gateway_url, BridgeConfig::default());

    bridge_client
        .connect_and_relay(move |msg: BridgeMessage| {
            let brain_ws_url = brain_ws_url.clone();
            let api_key = api_key.clone();
            async move {
                let (ws, _) = tokio_tungstenite::connect_async(&brain_ws_url)
                    .await
                    .expect("Failed to connect to Brain WebSocket");

                let (mut sink, mut stream) = ws.split();

                let signal = serde_json::json!({
                    "api_key": api_key,
                    "content": msg.content,
                    "source": msg.source.clone().unwrap_or_else(|| "bridge".to_string()),
                    "metadata": msg.metadata.clone(),
                });

                sink.send(Message::Text(signal.to_string().into()))
                    .await
                    .expect("Failed to send to Brain");

                let response = stream
                    .next()
                    .await
                    .expect("No response from Brain")
                    .expect("Stream error");

                let response_text = match response {
                    Message::Text(t) => t.to_string(),
                    _ => "{}".to_string(),
                };

                let response_json: serde_json::Value =
                    serde_json::from_str(&response_text).unwrap_or_default();

                let content = response_json
                    .get("response")
                    .and_then(|r| r.as_str())
                    .unwrap_or("");

                BridgeMessage::reply(&msg, content)
            }
        })
        .await?;

    Ok(())
}
