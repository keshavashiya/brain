use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use super::{LlmError, LlmProvider, Message, Response, ResponseChunk, Role, Usage};

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f64,
    max_tokens: Option<i32>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OpenAiMessage {
    role: String,
    content: String,
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
        let client = reqwest::Client::builder()
            .timeout(brain_core::timeouts::LLM_GENERATE)
            .build()
            .map_err(|e| {
                LlmError::ProviderUnavailable(format!("Failed to create HTTP client: {e}"))
            })?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(|s| s.to_string()),
            model: model.to_string(),
            temperature,
            max_tokens,
        })
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
        Self::new(
            "https://openrouter.ai/api/v1",
            Some(api_key),
            model,
            0.7,
            Some(4096),
        )
    }

    fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
        messages
            .iter()
            .map(|m| OpenAiMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
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
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let data: OpenAiResponse = resp.json().await?;
        let content = data
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        Ok(Response {
            content,
            usage: data.usage.map(|u| Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
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
        };

        let resp = self
            .build_request(self.client.post(&url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

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
        "openai"
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
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        let data: Models = resp.json().await?;
        Ok(data.data.into_iter().map(|m| m.id).collect())
    }
}
