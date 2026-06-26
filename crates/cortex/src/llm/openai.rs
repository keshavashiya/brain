use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use super::{
    build_http_client, ensure_ok, LlmError, LlmProvider, Message, ProposedToolCall, Response,
    ResponseChunk, ToolDef, Usage,
};

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f64,
    max_tokens: Option<i32>,
    stream: bool,
    /// Advertised tools (OpenAI function-calling shape). Omitted entirely
    /// from a plain-text request so behaviour is unchanged when no tools
    /// channel is in play.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    /// `"auto"` lets the model answer in plain text or propose a call. We
    /// never force tool use, so chat stays able to just talk.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'static str>,
}

#[derive(Serialize, Deserialize, Default)]
struct OpenAiMessage {
    role: String,
    /// Optional on the response side: a tool-call turn carries `null`
    /// content. Always `Some` on the request side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    /// Set only on a `role:"tool"` result turn — links the result to the
    /// assistant `tool_calls` entry it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// One advertised tool in the OpenAI request (`{"type":"function", ...}`).
#[derive(Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDef,
}

#[derive(Serialize)]
struct OpenAiFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// A tool call in the response. `function.arguments` is a JSON-encoded
/// string per the OpenAI wire format. The same shape is replayed on the
/// request side for an assistant tool-call turn, where `type` must be
/// `"function"`.
#[derive(Serialize, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default = "function_kind")]
    kind: String,
    function: OpenAiFunctionCall,
}

fn function_kind() -> String {
    "function".to_string()
}

#[derive(Serialize, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiStreamResponse {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// OpenAI-compatible provider (works with OpenAI, OpenRouter, etc.)
pub struct OpenAiProvider {
    client: reqwest::Client,
    /// The configured provider label (preset name like `openrouter`, `groq`,
    /// or `openai_compat`). Surfaced by [`name`] so self-reporting ("which
    /// model am I?") names the actual gateway, not a hardcoded `openai`.
    name: String,
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f64,
    max_tokens: Option<i32>,
}

impl OpenAiProvider {
    pub fn new(
        base_url: &str,
        api_key: Option<&str>,
        model: &str,
        temperature: f64,
        max_tokens: Option<i32>,
    ) -> Result<Self, LlmError> {
        let client = build_http_client(brain::timeouts::LLM_GENERATE)?;
        Ok(Self {
            client,
            name: "openai".to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
            model: model.to_string(),
            temperature,
            max_tokens,
        })
    }

    /// Override the provider label reported by [`name`]. Used by
    /// `create_provider` to stamp the configured preset (`openrouter`,
    /// `groq`, …) so situated self-reporting names the real gateway.
    pub fn with_name(mut self, name: &str) -> Self {
        if !name.trim().is_empty() {
            self.name = name.trim().to_string();
        }
        self
    }

    pub fn openai(api_key: &str, model: &str) -> Result<Self, LlmError> {
        Self::new(
            "https://api.openai.com/v1",
            Some(api_key),
            model,
            0.7,
            Some(4096),
        )
    }

    pub fn openrouter(api_key: &str, model: &str) -> Result<Self, LlmError> {
        Ok(Self::new(
            "https://openrouter.ai/api/v1",
            Some(api_key),
            model,
            0.7,
            Some(4096),
        )?
        .with_name("openrouter"))
    }

    fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
        messages.iter().map(Self::convert_message).collect()
    }

    /// Translate one kernel [`Message`] into the OpenAI wire shape. An
    /// assistant turn that proposed tool calls replays them (with `null`
    /// content when it carried no prose); a [`Role::Tool`] result turn
    /// carries its `tool_call_id`; every other turn is plain content.
    fn convert_message(m: &Message) -> OpenAiMessage {
        let role = m.role.as_wire_str().to_string();
        if !m.tool_calls.is_empty() {
            return OpenAiMessage {
                role,
                content: (!m.content.is_empty()).then(|| m.content.clone()),
                tool_calls: Some(m.tool_calls.iter().map(convert_proposed_call).collect()),
                tool_call_id: None,
            };
        }
        OpenAiMessage {
            role,
            content: Some(m.content.clone()),
            tool_calls: None,
            tool_call_id: m.tool_call_id.clone(),
        }
    }

    /// Translate the kernel's provider-agnostic [`ToolDef`]s into the
    /// OpenAI function-calling request shape.
    fn convert_tools(tools: &[ToolDef]) -> Vec<OpenAiTool> {
        tools
            .iter()
            .map(|t| OpenAiTool {
                kind: "function",
                function: OpenAiFunctionDef {
                    name: sanitize_tool_name(&t.name),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    /// Map a response message's `tool_calls` into provider-agnostic
    /// [`ProposedToolCall`]s, parsing each JSON-string argument blob into a
    /// [`serde_json::Value`] (empty / unparseable args become an empty
    /// object so the caller never has to re-parse or guard for null).
    fn extract_tool_calls(message: &OpenAiMessage) -> Vec<ProposedToolCall> {
        message
            .tool_calls
            .iter()
            .flatten()
            .map(|tc| ProposedToolCall {
                id: tc.id.clone(),
                name: desanitize_tool_name(&tc.function.name),
                arguments: parse_arguments(&tc.function.arguments),
            })
            .collect()
    }

    fn build_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut builder = builder;
        if let Some(key) = &self.api_key {
            builder = builder.header("Authorization", format!("Bearer {}", key));
        }
        builder
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: false,
            tools: None,
            tool_choice: None,
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;
        let resp = ensure_ok(resp).await?;

        let data: OpenAiResponse = resp.json().await?;
        let content = data
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(Response::text(content, convert_usage(data.usage)))
    }

    async fn generate_with_tools(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<Response, LlmError> {
        // No tools to advertise → identical to a plain generate.
        if tools.is_empty() {
            return self.generate(messages).await;
        }

        let url = format!("{}/chat/completions", self.base_url);
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: false,
            tools: Some(Self::convert_tools(tools)),
            // Never force a call — the model may answer in plain text.
            tool_choice: Some("auto"),
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;
        let resp = ensure_ok(resp).await?;

        let data: OpenAiResponse = resp.json().await?;
        let (content, tool_calls) = match data.choices.first() {
            Some(choice) => (
                choice.message.content.clone().unwrap_or_default(),
                Self::extract_tool_calls(&choice.message),
            ),
            None => (String::new(), Vec::new()),
        };

        Ok(Response {
            content,
            usage: convert_usage(data.usage),
            tool_calls,
        })
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        use futures::stream::try_unfold;

        let url = format!("{}/chat/completions", self.base_url);
        let request = OpenAiRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: true,
            tools: None,
            tool_choice: None,
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;
        let resp = ensure_ok(resp).await?;

        let byte_stream = resp.bytes_stream();
        let stream = try_unfold(
            (Box::pin(byte_stream), String::new()),
            |(mut byte_stream, mut buf)| async move {
                use futures::TryStreamExt;

                loop {
                    if let Some(newline_pos) = buf.find('\n') {
                        let line: String = buf[..newline_pos].to_string();
                        buf = buf[newline_pos + 1..].to_string();

                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        if let Some(data) = line.strip_prefix("data: ") {
                            let data = data.trim();
                            if data == "[DONE]" {
                                return Ok(None);
                            }

                            match serde_json::from_str::<OpenAiStreamResponse>(data) {
                                Ok(resp) => {
                                    if let Some(choice) = resp.choices.first() {
                                        let content =
                                            choice.delta.content.clone().unwrap_or_default();
                                        let is_done = choice.finish_reason.is_some();
                                        let chunk = ResponseChunk { content, is_done };
                                        return Ok(Some((chunk, (byte_stream, buf))));
                                    }
                                    continue;
                                }
                                Err(e) => {
                                    return Err(LlmError::InvalidFormat(format!(
                                        "Failed to parse streaming response: {e}"
                                    )));
                                }
                            }
                        }
                        continue;
                    }

                    match byte_stream.try_next().await {
                        Ok(Some(bytes)) => {
                            buf.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => return Ok(None),
                        Err(e) => return Err(LlmError::Http(e)),
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/models", self.base_url);
        match self.build_request(self.client.get(&url)).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// OpenAI-compatible servers on loopback (llama.cpp, LM Studio,
    /// LocalAI) count as local; hosted endpoints do not.
    fn is_local(&self) -> bool {
        brain::url_is_loopback(&self.base_url)
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
        }
        #[derive(Deserialize)]
        struct Models {
            data: Vec<ModelEntry>,
        }

        let url = format!("{}/models", self.base_url);
        let resp = self.build_request(self.client.get(&url)).send().await?;
        let resp = ensure_ok(resp).await?;
        let data: Models = resp.json().await?;
        Ok(data.data.into_iter().map(|m| m.id).collect())
    }

    async fn fetch_context_window(&self) -> Option<usize> {
        // 1. API-based detection: some providers (OpenRouter) advertise
        //    `context_length` per model in their /models response.
        #[derive(Deserialize)]
        struct ModelDetail {
            id: String,
            #[serde(default)]
            context_length: Option<usize>,
        }
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelDetail>,
        }

        let from_api = (async {
            let url = format!("{}/models", self.base_url);
            let resp = self
                .build_request(self.client.get(&url))
                .send()
                .await
                .ok()?;
            let resp = ensure_ok(resp).await.ok()?;
            let data: ModelsResponse = resp.json().await.ok()?;
            let active = self.model();
            // Exact match first.
            for model in &data.data {
                if model.id == active {
                    return model.context_length;
                }
            }
            // Prefix match for OpenRouter model IDs like "openai/gpt-4o"
            // where the config stores just "gpt-4o".
            for model in &data.data {
                if model.id.ends_with(active) || model.id.contains(active) {
                    return model.context_length;
                }
            }
            None
        })
        .await;
        if from_api.is_some() {
            return from_api;
        }

        // 2. Model-name heuristics (covers OpenAI, Groq, DeepSeek, etc.).
        super::known_context_window(self.model())
    }
}

/// Map the wire usage block into the kernel's [`Usage`].
fn convert_usage(usage: Option<OpenAiUsage>) -> Option<Usage> {
    usage.map(|u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    })
}

/// Reverse of [`OpenAiProvider::extract_tool_calls`]: render a kernel
/// [`ProposedToolCall`] back into the OpenAI wire shape for an assistant
/// tool-call turn. `arguments` go back out as a JSON-encoded string.
fn convert_proposed_call(call: &ProposedToolCall) -> OpenAiToolCall {
    OpenAiToolCall {
        id: call.id.clone(),
        kind: function_kind(),
        function: OpenAiFunctionCall {
            name: sanitize_tool_name(&call.name),
            arguments: serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string()),
        },
    }
}

/// OpenAI's function-name grammar is `^[a-zA-Z0-9_-]+$` — no dots. Strict
/// gateways (DeepSeek via OpenCode Zen, OpenAI itself) reject anything else
/// with a `400 invalid_request_error`, while lenient ones (Ollama) accept it.
/// Brain's native tools are dotted namespaces (`net.http`, `memory.store`), so
/// we map `.`→`__` on the wire and reverse it when reading a proposed call
/// back. The pair is a clean inverse as long as a tool name never legitimately
/// contains `__` — which holds for every native verb and the common MCP shape
/// (`create_issue`, single underscores).
fn sanitize_tool_name(name: &str) -> String {
    name.replace('.', "__")
}

/// Inverse of [`sanitize_tool_name`]: restore the kernel's dotted tool name
/// from the wire form so the route resolver matches the advertised descriptor.
fn desanitize_tool_name(name: &str) -> String {
    name.replace("__", ".")
}

/// Parse a tool call's JSON-string `arguments` into a [`serde_json::Value`].
/// An empty or unparseable blob becomes an empty object so callers always
/// get a well-formed args object.
fn parse_arguments(raw: &str) -> serde_json::Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| serde_json::json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_name_sanitize_round_trips_dotted_names() {
        // Dotted native names become OpenAI-legal (`^[a-zA-Z0-9_-]+$`) and map
        // back exactly. Single underscores and hyphens (the common MCP shape)
        // are preserved across the round trip.
        for name in [
            "net.http",
            "memory.store",
            "fs.read",
            "create_issue",
            "do-thing",
        ] {
            let wire = sanitize_tool_name(name);
            assert!(
                wire.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "sanitized name {wire:?} must satisfy OpenAI's function-name grammar"
            );
            assert_eq!(desanitize_tool_name(&wire), name, "round trip for {name:?}");
        }
        assert!(!sanitize_tool_name("net.http").contains('.'));
    }

    #[test]
    fn convert_tools_emits_dot_free_function_names() {
        let tools = vec![ToolDef {
            name: "net.http".to_string(),
            description: "fetch a url".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }];
        let out = OpenAiProvider::convert_tools(&tools);
        assert_eq!(out[0].function.name, "net__http");
    }

    #[test]
    fn extract_tool_calls_restores_dotted_name() {
        let msg = OpenAiMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![OpenAiToolCall {
                id: Some("call_1".to_string()),
                kind: function_kind(),
                function: OpenAiFunctionCall {
                    name: "net__http".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
        };
        let calls = OpenAiProvider::extract_tool_calls(&msg);
        assert_eq!(calls[0].name, "net.http");
    }
}
