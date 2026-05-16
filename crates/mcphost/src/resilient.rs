//! Resilience-layered `MCPHost` decorator.
//!
//! Wraps any inner `MCPHost` with the Phase 4 Track A stack so a
//! single `call` traverses Timeout → RateLimit → CircuitBreaker →
//! LoopDetector → Retry → DLQ → real call. Each layer is optional —
//! a `ResilientMcpHost` with no layers wired is a pass-through.
//!
//! Stack order rationale (outermost → inner):
//! 1. **LoopDetector** runs first so a runaway agent loop fails fast
//!    without ever consuming a rate-limit token or breaker slot.
//! 2. **CircuitBreaker** check is the next short-circuit — a tool
//!    that's known-bad shouldn't burn tokens or retries.
//! 3. **RateLimit** acquires a token (may await — that's the back
//!    pressure point).
//! 4. **Timeout** wraps the retry cycle so the total wall-clock
//!    budget is bounded even when transient failures stack delays.
//! 5. **Retry** drives the actual call; uses the breaker as an abort
//!    signal so a retry burst can't drag a healthy breaker open.
//! 6. **DLQ** captures the final failure (retry-exhaustion, timeout,
//!    transport error, or `is_error: true`) so audit replay can
//!    surface it.
//!
//! Breaker accounting happens on the **inner** call: each attempt's
//! outcome (success, transport error, `is_error: true`) flows into
//! `record_success` / `record_failure` so the breaker reflects
//! ground truth, not the decorator's framing.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use intent::BreakerCheck;
use resilience::{
    retry, timeout, BreakerRegistry, DeadLetterQueue, DlqEntry, LoopDetector, LoopDetectorError,
    RateLimitRegistry, RetryConfig, RetryOutcome, TimeoutError,
};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::McpHostError;
use crate::types::{CallOutcome, ServerConfig, ServerStatus, ToolDescriptor};
use crate::MCPHost;

/// Resilience knobs that aren't already encoded in the layer
/// registries themselves.
#[derive(Debug, Clone, Default)]
pub struct ResilienceConfig {
    /// Wall-clock budget for one `call` (retries included). `None` =
    /// no enforced timeout (still bounded by transport timeouts).
    pub timeout: Option<Duration>,
    /// Retry policy. `None` = single-shot (no retry).
    pub retry: Option<RetryConfig>,
}

/// Decorator over `Arc<dyn MCPHost>` that layers the resilience stack
/// over `call`. All other trait methods delegate verbatim.
pub struct ResilientMcpHost {
    inner: Arc<dyn MCPHost>,
    principal: String,
    config: ResilienceConfig,
    breakers: Option<Arc<BreakerRegistry>>,
    rate_limits: Option<Arc<RateLimitRegistry>>,
    loop_detector: Option<Arc<LoopDetector>>,
    dlq: Option<Arc<dyn DeadLetterQueue>>,
}

impl ResilientMcpHost {
    /// Build a pass-through decorator. Wire layers via the builder
    /// methods. Without any layers, behavior is identical to the
    /// inner host (used to keep call sites symmetrical).
    pub fn new(inner: Arc<dyn MCPHost>) -> Self {
        Self {
            inner,
            principal: String::new(),
            config: ResilienceConfig::default(),
            breakers: None,
            rate_limits: None,
            loop_detector: None,
            dlq: None,
        }
    }

    /// Principal id used to scope the loop detector. Empty string
    /// (the default) means "anonymous" — still isolated from named
    /// agents because the loop detector keys on the literal string.
    pub fn with_principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = principal.into();
        self
    }

    pub fn with_config(mut self, config: ResilienceConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_breakers(mut self, breakers: Arc<BreakerRegistry>) -> Self {
        self.breakers = Some(breakers);
        self
    }

    pub fn with_rate_limits(mut self, rl: Arc<RateLimitRegistry>) -> Self {
        self.rate_limits = Some(rl);
        self
    }

    pub fn with_loop_detector(mut self, ld: Arc<LoopDetector>) -> Self {
        self.loop_detector = Some(ld);
        self
    }

    pub fn with_dlq(mut self, dlq: Arc<dyn DeadLetterQueue>) -> Self {
        self.dlq = Some(dlq);
        self
    }

    /// One inner attempt: invokes the underlying host and records
    /// breaker outcome. Kept separate so the retry closure can call
    /// it repeatedly without re-wiring breaker accounting.
    async fn one_attempt(
        &self,
        server: &str,
        tool: &str,
        tool_id: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let result = self.inner.call(server, tool, args).await;
        if let Some(breakers) = &self.breakers {
            match &result {
                Ok(outcome) if !outcome.is_error => breakers.record_success(tool_id).await,
                _ => breakers.record_failure(tool_id).await,
            }
        }
        result
    }

    async fn enqueue_dlq(
        &self,
        tool_id: &str,
        request_json: &str,
        error_message: String,
        attempts: u32,
    ) {
        let Some(dlq) = &self.dlq else { return };
        let entry = DlqEntry {
            id: Uuid::new_v4().to_string(),
            tool_id: tool_id.to_string(),
            request_json: request_json.to_string(),
            error_message,
            attempts,
            dlq_at: Utc::now(),
        };
        if let Err(e) = dlq.enqueue(entry).await {
            warn!(tool_id, error = %e, "failed to enqueue DLQ entry");
        }
    }
}

#[async_trait]
impl MCPHost for ResilientMcpHost {
    async fn mount(&self, name: String, cfg: ServerConfig) -> Result<(), McpHostError> {
        self.inner.mount(name, cfg).await
    }

    async fn unmount(&self, name: &str) -> Result<(), McpHostError> {
        self.inner.unmount(name).await
    }

    async fn list_servers(&self) -> Vec<ServerStatus> {
        self.inner.list_servers().await
    }

    async fn list_all_tools(&self) -> Vec<ToolDescriptor> {
        self.inner.list_all_tools().await
    }

    async fn call(
        &self,
        server: &str,
        tool: &str,
        args: serde_json::Value,
    ) -> Result<CallOutcome, McpHostError> {
        let tool_id = format!("mcp:{server}:{tool}");
        let request_json = serde_json::to_string(&args)
            .unwrap_or_else(|_| String::from(r#"{"error":"unserializable"}"#));

        // 1. LoopDetector — fail fast, never touches retry/DLQ.
        if let Some(ld) = &self.loop_detector {
            if let Err(LoopDetectorError::LoopDetected { count, window, .. }) =
                ld.check(&self.principal, &tool_id, &args).await
            {
                return Err(McpHostError::Transport(format!(
                    "loop detected: {tool_id} repeated {count} times in window {window}"
                )));
            }
        }

        // 2. CircuitBreaker — short-circuit when Open. The retry
        //    primitive also checks this between attempts, but a
        //    fast-fail on entry avoids burning a rate-limit token.
        if let Some(breakers) = &self.breakers {
            if breakers.is_open(&tool_id).await {
                debug!(tool_id, "breaker open — short-circuiting");
                return Err(McpHostError::Transport(format!(
                    "breaker open for {tool_id}"
                )));
            }
        }

        // 3. RateLimit — back-pressure.
        if let Some(rl) = &self.rate_limits {
            rl.acquire(&tool_id).await;
        }

        // 4 + 5. Timeout(Retry(real call)).
        let attempts_taken = std::sync::atomic::AtomicU32::new(0);
        let call_fut = async {
            if let Some(retry_cfg) = &self.config.retry {
                let breaker_check = self
                    .breakers
                    .as_ref()
                    .map(|b| (b.clone() as Arc<dyn intent::BreakerCheck>, tool_id.as_str()));
                retry(retry_cfg, breaker_check, || {
                    attempts_taken.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    self.one_attempt(server, tool, &tool_id, args.clone())
                })
                .await
            } else {
                attempts_taken.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.one_attempt(server, tool, &tool_id, args.clone())
                    .await
                    .map_err(RetryOutcome::Exhausted)
            }
        };

        let final_result: Result<CallOutcome, RetryOutcome<McpHostError>> =
            match self.config.timeout {
                Some(d) => match timeout(d, call_fut).await {
                    Ok(outcome) => Ok(outcome),
                    Err(TimeoutError::Elapsed) => {
                        // Treat the timeout as a failure for the breaker
                        // and DLQ surfaces.
                        if let Some(breakers) = &self.breakers {
                            breakers.record_failure(&tool_id).await;
                        }
                        let attempts = attempts_taken.load(std::sync::atomic::Ordering::SeqCst);
                        self.enqueue_dlq(
                            &tool_id,
                            &request_json,
                            format!("timeout after {}ms", d.as_millis()),
                            attempts,
                        )
                        .await;
                        return Err(McpHostError::Transport(format!(
                            "timeout after {}ms",
                            d.as_millis()
                        )));
                    }
                    Err(TimeoutError::Inner(retry_outcome)) => Err(retry_outcome),
                },
                None => call_fut.await,
            };

        let attempts = attempts_taken.load(std::sync::atomic::Ordering::SeqCst);
        match final_result {
            Ok(outcome) => {
                if outcome.is_error {
                    // is_error: true is a fail for DLQ purposes — the
                    // tool reported a logical failure, not a hiccup
                    // worth retrying again later.
                    self.enqueue_dlq(
                        &tool_id,
                        &request_json,
                        format!(
                            "tool returned is_error after {attempts} attempt(s): {}",
                            outcome.content
                        ),
                        attempts,
                    )
                    .await;
                }
                Ok(outcome)
            }
            Err(RetryOutcome::Exhausted(e)) => {
                self.enqueue_dlq(
                    &tool_id,
                    &request_json,
                    format!("exhausted after {attempts} attempt(s): {e}"),
                    attempts,
                )
                .await;
                Err(e)
            }
            Err(RetryOutcome::BreakerOpenAbort(abort)) => {
                self.enqueue_dlq(
                    &tool_id,
                    &request_json,
                    format!(
                        "breaker opened mid-retry after {attempts} attempt(s): {}",
                        abort.last_error
                    ),
                    attempts,
                )
                .await;
                Err(abort.last_error)
            }
        }
    }
}
