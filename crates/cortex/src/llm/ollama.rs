use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

use super::{LlmError, LlmProvider, Message, Response, ResponseChunk, Role, Usage};

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
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
        let client = reqwest::Client::builder()
            .timeout(brain_core::timeouts::LLM_GENERATE)
            .build()
            .map_err(|e| {
                LlmError::ProviderUnavailable(format!("Failed to create HTTP client: {e}"))
            })?;

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
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
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
        };

        let resp = self.client.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let data: OllamaResponse = resp.json().await?;

        Ok(Response {
            content: data.message.map(|m| m.content).unwrap_or_default(),
            usage: Some(Usage {
                prompt_tokens: data.prompt_eval_count.unwrap_or(0),
                completion_tokens: data.eval_count.unwrap_or(0),
                total_tokens: data.prompt_eval_count.unwrap_or(0) + data.eval_count.unwrap_or(0),
            }),
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
        };

        let resp = self.client.post(&url).json(&request).send().await?;

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
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: body,
            });
        }
        let data: Tags = resp.json().await?;
        Ok(data.models.into_iter().map(|m| m.name).collect())
    }
}
