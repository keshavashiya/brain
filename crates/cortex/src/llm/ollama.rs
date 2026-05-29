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
        let (content, tool_calls) = match data.message {
            Some(ref m) => (m.content.clone(), Self::extract_tool_calls(m)),
            None => (String::new(), Vec::new()),
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
