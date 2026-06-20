use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use super::{
    build_http_client, ensure_ok, LlmError, LlmProvider, Message, ProposedToolCall, Response,
    ResponseChunk, ToolDef, Usage,
};

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: Option<OllamaOptions>,
    /// Advertised tools. Omitted from a plain-text request so behaviour is
    /// unchanged when no tools channel is in play.
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
}

#[derive(Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

/// One advertised tool in the request (`{"type":"function", ...}` — Ollama
/// mirrors the OpenAI function-calling shape).
#[derive(Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OllamaFunctionDef,
}

#[derive(Serialize)]
struct OllamaFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// A tool call in the response. Unlike OpenAI, Ollama sends
/// `function.arguments` as a JSON *object*, not a string.
#[derive(Serialize, Deserialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f64,
    #[serde(rename = "num_predict")]
    num_predict: i32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: Option<OllamaMessage>,
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

/// Ollama LLM provider.
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    temperature: f64,
    max_tokens: i32,
}

impl OllamaProvider {
    pub fn new(
        base_url: &str,
        model: &str,
        temperature: f64,
        max_tokens: i32,
    ) -> Result<Self, LlmError> {
        let client = build_http_client(brain::timeouts::LLM_GENERATE)?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            temperature,
            max_tokens,
        })
    }

    pub fn default_config() -> Result<Self, LlmError> {
        Self::new("http://localhost:11434", "qwen2.5-coder:7b", 0.7, 4096)
    }

    fn convert_messages(messages: &[Message]) -> Vec<OllamaMessage> {
        messages
            .iter()
            .map(|m| OllamaMessage {
                role: m.role.as_wire_str().to_string(),
                content: m.content.clone(),
                tool_calls: (!m.tool_calls.is_empty())
                    .then(|| m.tool_calls.iter().map(convert_proposed_call).collect()),
            })
            .collect()
    }

    /// Translate the kernel's provider-agnostic [`ToolDef`]s into Ollama's
    /// function-calling request shape.
    fn convert_tools(tools: &[ToolDef]) -> Vec<OllamaTool> {
        tools
            .iter()
            .map(|t| OllamaTool {
                kind: "function",
                function: OllamaFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    /// Map a response message's `tool_calls` into provider-agnostic
    /// [`ProposedToolCall`]s. Ollama supplies no call id and sends
    /// arguments as an object, which we pass through unchanged.
    fn extract_tool_calls(message: &OllamaMessage) -> Vec<ProposedToolCall> {
        message
            .tool_calls
            .iter()
            .flatten()
            .map(|tc| ProposedToolCall {
                id: None,
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl LlmProvider for OllamaProvider {
    async fn generate(&self, messages: &[Message]) -> Result<Response, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            }),
            tools: None,
        };

        let resp = self.client.post(&url).json(&request).send().await?;
        let resp = ensure_ok(resp).await?;

        let data: OllamaResponse = resp.json().await?;
        let usage = usage_from(&data);

        Ok(Response::text(
            data.message.map(|m| m.content).unwrap_or_default(),
            usage,
        ))
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

        let url = format!("{}/api/chat", self.base_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: false,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            }),
            tools: Some(Self::convert_tools(tools)),
        };

        let resp = self.client.post(&url).json(&request).send().await?;
        let resp = ensure_ok(resp).await?;

        let data: OllamaResponse = resp.json().await?;
        let usage = usage_from(&data);
        let (content, mut tool_calls) = match data.message {
            Some(ref m) => (m.content.clone(), Self::extract_tool_calls(m)),
            None => (String::new(), Vec::new()),
        };

        // Fallback for models that emit the call as JSON *text* rather than via
        // Ollama's structured `tool_calls` field (notably qwen2.5-coder, the
        // default local model). Without this the call is never dispatched and
        // the raw JSON leaks to the user. When we recover a call from the
        // content, the content was *only* that call, so we clear it.
        let content = if tool_calls.is_empty() {
            let recovered = tool_calls_from_content(&content, tools);
            if recovered.is_empty() {
                content
            } else {
                tool_calls = recovered;
                String::new()
            }
        } else {
            content
        };

        Ok(Response {
            content,
            usage,
            tool_calls,
        })
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseChunk, LlmError>> + Send>>, LlmError> {
        use futures::stream::try_unfold;

        let url = format!("{}/api/chat", self.base_url);
        let request = OllamaRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            stream: true,
            options: Some(OllamaOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            }),
            tools: None,
        };

        let resp = self.client.post(&url).json(&request).send().await?;
        let resp = ensure_ok(resp).await?;

        let byte_stream = resp.bytes_stream();
        let stream = try_unfold(
            (Box::pin(byte_stream), String::new(), false),
            |(mut byte_stream, mut buf, done)| async move {
                use futures::TryStreamExt;

                if done {
                    return Ok(None);
                }

                loop {
                    if let Some(newline_pos) = buf.find('\n') {
                        let line: String = buf[..newline_pos].to_string();
                        buf = buf[newline_pos + 1..].to_string();

                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<OllamaResponse>(line) {
                            Ok(data) => {
                                let is_done = data.done;
                                let content = data.message.map(|m| m.content).unwrap_or_default();
                                let chunk = ResponseChunk { content, is_done };
                                return Ok(Some((chunk, (byte_stream, buf, is_done))));
                            }
                            Err(e) => {
                                return Err(LlmError::InvalidFormat(format!(
                                    "Failed to parse streaming response: {e}"
                                )));
                            }
                        }
                    }

                    match byte_stream.try_next().await {
                        Ok(Some(bytes)) => {
                            buf.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => {
                            let remaining = buf.trim();
                            if !remaining.is_empty() {
                                if let Ok(data) = serde_json::from_str::<OllamaResponse>(remaining)
                                {
                                    let content =
                                        data.message.map(|m| m.content).unwrap_or_default();
                                    return Ok(Some((
                                        ResponseChunk {
                                            content,
                                            is_done: true,
                                        },
                                        (byte_stream, String::new(), true),
                                    )));
                                }
                            }
                            return Ok(None);
                        }
                        Err(e) => return Err(LlmError::Http(e)),
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    async fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    fn name(&self) -> &str {
        "ollama"
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// Ollama is only "local" when it actually runs on this machine —
    /// a LAN-hosted Ollama still takes content off-box.
    fn is_local(&self) -> bool {
        brain::url_is_loopback(&self.base_url)
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        #[derive(Deserialize)]
        struct Tag {
            name: String,
        }
        #[derive(Deserialize)]
        struct Tags {
            models: Vec<Tag>,
        }

        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await?;
        let resp = ensure_ok(resp).await?;
        let data: Tags = resp.json().await?;
        Ok(data.models.into_iter().map(|m| m.name).collect())
    }

    async fn fetch_context_window(&self) -> Option<usize> {
        // 1. API-based detection via /api/show (works for most Ollama models).
        #[derive(Deserialize)]
        struct ModelInfo {
            #[serde(default)]
            model_info: std::collections::HashMap<String, serde_json::Value>,
        }

        let from_api = (async {
            let url = format!("{}/api/show", self.base_url);
            let body = serde_json::json!({ "model": self.model });
            let resp = self.client.post(&url).json(&body).send().await.ok()?;
            let resp = ensure_ok(resp).await.ok()?;
            let data: ModelInfo = resp.json().await.ok()?;

            // Ollama exposes context length under various keys depending
            // on the backend. Try known patterns.
            for key in &[
                "llama.context_length",
                "gptneox.context_length",
                "llama2.context_length",
            ] {
                if let Some(val) = data.model_info.get(*key) {
                    if let Some(n) = val.as_u64().or_else(|| val.as_f64().map(|f| f as u64)) {
                        let n = n as usize;
                        // Sanity: reject anything below 512 (parse artifact).
                        if n >= 512 {
                            return Some(n);
                        }
                    }
                }
            }
            None
        })
        .await;
        if from_api.is_some() {
            return from_api;
        }

        // 2. Model-name heuristics.
        super::known_context_window(self.model())
    }
}

/// Reverse of [`OllamaProvider::extract_tool_calls`]: render a kernel
/// [`ProposedToolCall`] back into Ollama's request shape for an assistant
/// tool-call turn. Arguments stay an object (Ollama's wire format).
fn convert_proposed_call(call: &ProposedToolCall) -> OllamaToolCall {
    OllamaToolCall {
        function: OllamaFunctionCall {
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        },
    }
}

/// Recover tool calls a model emitted as JSON *text* instead of through
/// Ollama's structured `tool_calls` field. Some local models — notably
/// qwen2.5-coder, the default — reply with `{"name": "...", "arguments":
/// {...}}` in the message content when offered tools; without this they would
/// never dispatch and the raw JSON would leak to the user as the answer.
///
/// Only objects whose `name` matches an *offered* tool are accepted, so a
/// model legitimately answering with JSON is not mistaken for a tool call. A
/// bare object and a JSON array of objects are both handled.
fn tool_calls_from_content(content: &str, tools: &[ToolDef]) -> Vec<ProposedToolCall> {
    let candidate = extract_json_block(content);
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&candidate) else {
        return Vec::new();
    };
    let objects = match value {
        serde_json::Value::Array(items) => items,
        obj @ serde_json::Value::Object(_) => vec![obj],
        _ => return Vec::new(),
    };
    let names: std::collections::HashSet<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    objects
        .into_iter()
        .filter_map(|obj| {
            let name = obj.get("name")?.as_str()?.to_string();
            if !names.contains(name.as_str()) {
                return None;
            }
            // Models name the args field "arguments" or "parameters"; absent
            // means a no-arg call.
            let arguments = obj
                .get("arguments")
                .or_else(|| obj.get("parameters"))
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
            // Some models double-encode the args as a JSON string.
            let arguments = match arguments {
                serde_json::Value::String(s) => {
                    serde_json::from_str(&s).unwrap_or(serde_json::Value::String(s))
                }
                other => other,
            };
            Some(ProposedToolCall {
                id: None,
                name,
                arguments,
            })
        })
        .collect()
}

/// Pull the most likely JSON payload out of a model message: the body of the
/// first fenced code block when present (dropping an optional language tag),
/// otherwise the span from the first `{`/`[` to the last `}`/`]`. Falls back
/// to the trimmed whole string.
fn extract_json_block(content: &str) -> String {
    let trimmed = content.trim();
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        // Skip an optional language tag (e.g. `json`) up to the first newline.
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        if let Some(end) = body.find("```") {
            return body[..end].trim().to_string();
        }
    }
    let first = trimmed.find(['{', '[']);
    let last = trimmed.rfind(['}', ']']);
    if let (Some(a), Some(b)) = (first, last) {
        if b > a {
            return trimmed[a..=b].to_string();
        }
    }
    trimmed.to_string()
}

/// Build the kernel's [`Usage`] from an Ollama response's eval counts.
fn usage_from(data: &OllamaResponse) -> Option<Usage> {
    let prompt = data.prompt_eval_count.unwrap_or(0);
    let completion = data.eval_count.unwrap_or(0);
    Some(Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> ToolDef {
        ToolDef {
            name: name.to_string(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn recovers_bare_json_object_call() {
        // The exact shape qwen2.5-coder:7b returns in `content`.
        let content = r#"{"name": "net.check", "arguments": {"host": "github.com"}}"#;
        let calls = tool_calls_from_content(content, &[tool("net.check")]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "net.check");
        assert_eq!(
            calls[0].arguments,
            serde_json::json!({"host": "github.com"})
        );
    }

    #[test]
    fn recovers_fenced_json_call() {
        let content =
            "Sure!\n```json\n{\"name\": \"net.check\", \"arguments\": {\"host\": \"x\"}}\n```";
        let calls = tool_calls_from_content(content, &[tool("net.check")]);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, serde_json::json!({"host": "x"}));
    }

    #[test]
    fn rejects_call_for_unoffered_tool() {
        // A model legitimately answering *with* JSON must not be hijacked into
        // a tool call: only names matching an offered tool are recovered.
        let content = r#"{"name": "totally.unknown", "arguments": {}}"#;
        assert!(tool_calls_from_content(content, &[tool("net.check")]).is_empty());
    }

    #[test]
    fn accepts_parameters_alias_and_missing_args() {
        let with_params = r#"{"name": "net.check", "parameters": {"host": "h"}}"#;
        let calls = tool_calls_from_content(with_params, &[tool("net.check")]);
        assert_eq!(calls[0].arguments, serde_json::json!({"host": "h"}));

        let no_args = r#"{"name": "status.now"}"#;
        let calls = tool_calls_from_content(no_args, &[tool("status.now")]);
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn recovers_double_encoded_arguments() {
        // Some models encode arguments as a JSON *string*.
        let content = r#"{"name": "net.check", "arguments": "{\"host\": \"h\"}"}"#;
        let calls = tool_calls_from_content(content, &[tool("net.check")]);
        assert_eq!(calls[0].arguments, serde_json::json!({"host": "h"}));
    }

    #[test]
    fn recovers_array_of_calls() {
        let content = r#"[{"name":"a.x","arguments":{}},{"name":"b.y","arguments":{}}]"#;
        let calls = tool_calls_from_content(content, &[tool("a.x"), tool("b.y")]);
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn plain_prose_is_not_a_call() {
        assert!(tool_calls_from_content("Hello! How can I help?", &[tool("net.check")]).is_empty());
    }
}
