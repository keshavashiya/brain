//! Channel selection policy.
//!
//! Given a [`DeliveryIntent`] and a pool of registered channels, the router
//! returns an ordered list of candidates — highest priority first. Delivery
//! is attempted in order until one succeeds (or the list is exhausted).
//!
//! Selection factors, in order of weight:
//! 1. Caller's explicit `preferred_channel` (hard override).
//! 2. Urgency — `Critical`/`High` short-circuit to every `supports_response()`
//!    channel plus any alert channels, regardless of preference score.
//! 3. Learned preferences for `(namespace, category)` — pinned first, then
//!    weight-ranked with a `min_weight` threshold.
//! 4. Initiation channel (the channel the user came in on) as a fallback,
//!    so Brain always replies on the surface the user used.
//! 5. Any remaining healthy channels for the category — last resort.
//!
//! The router does not itself call transports — that is the caller's job
//! once it holds a [`RoutingDecision`]. This keeps routing pure/testable
//! and lets wire-time code decide how to actually push a message out.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::error::ChannelError;
use crate::preference::ChannelPreferenceStore;
#[cfg(test)]
use crate::types::ChannelKind;
use crate::types::{ChannelDescriptor, DeliveryIntent, UrgencyLevel};

/// Default minimum learned weight for a channel to be considered.
/// Pinned channels bypass this threshold.
pub const DEFAULT_MIN_WEIGHT: f32 = 0.2;

/// Routing context passed by the caller at decision time.
#[derive(Debug, Clone)]
pub struct RoutingContext {
    /// "Now" for tests / time-of-day awareness (injected for deterministic testing).
    pub now: DateTime<Utc>,
    /// Minimum weight for an unpinned learned preference to count.
    pub min_weight: f32,
    /// Hard cap on the number of candidates returned (attempts).
    pub max_candidates: usize,
}

impl Default for RoutingContext {
    fn default() -> Self {
        Self {
            now: Utc::now(),
            min_weight: DEFAULT_MIN_WEIGHT,
            max_candidates: 4,
        }
    }
}

/// A routing decision — ordered candidate channels + the reasoning trail.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Candidate channels in preferred order. Empty ⇒ no viable channel;
    /// the caller should surface an error via [`ChannelError::NoChannelAvailable`].
    pub candidates: Vec<ChannelDescriptor>,
    /// Human-readable reasons for each candidate (same length as `candidates`).
    /// Useful for CLI `--verbose` output and audit entries.
    pub reasons: Vec<String>,
}

impl RoutingDecision {
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// The routing trait. Implementations can be replaced to swap policies
/// (e.g., a future "LLM-assisted routing" experiment) without touching
/// callers.
#[async_trait]
pub trait ChannelRouter: Send + Sync {
    /// Pick an ordered candidate list for the given intent.
    async fn route(
        &self,
        intent: &DeliveryIntent,
        ctx: &RoutingContext,
    ) -> Result<RoutingDecision, ChannelError>;

    /// Register (or refresh) a channel descriptor. Caller is responsible
    /// for wiring the actual transport that delivers to it.
    async fn register(&self, descriptor: ChannelDescriptor) -> Result<(), ChannelError>;

    /// Mark a channel (un)healthy so the router can skip failing channels
    /// until they recover.
    async fn set_health(&self, channel_id: &str, healthy: bool) -> Result<(), ChannelError>;

    /// Return every currently registered channel.
    async fn list_channels(&self) -> Result<Vec<ChannelDescriptor>, ChannelError>;
}

/// Default policy — combines learned preferences with a sensible fallback order.
pub struct DefaultChannelRouter {
    preferences: std::sync::Arc<dyn ChannelPreferenceStore>,
    channels: tokio::sync::RwLock<HashMap<String, ChannelDescriptor>>,
}

impl DefaultChannelRouter {
    pub fn new(preferences: std::sync::Arc<dyn ChannelPreferenceStore>) -> Self {
        Self {
            preferences,
            channels: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Seed the router with a fixed channel set at construction time
    /// (convenience for tests and static configs).
    pub async fn with_channels(self, descriptors: Vec<ChannelDescriptor>) -> Self {
        {
            let mut map = self.channels.write().await;
            for d in descriptors {
                map.insert(d.id.clone(), d);
            }
        }
        self
    }

    /// Return channels that support user responses (Local/Http/WS/Relay).
    /// Used for urgency-escalated routing and as the fallback pool.
    async fn response_capable(&self) -> Vec<ChannelDescriptor> {
        let map = self.channels.read().await;
        map.values()
            .filter(|c| c.healthy && c.kind.supports_response())
            .cloned()
            .collect()
    }
}

#[async_trait]
impl ChannelRouter for DefaultChannelRouter {
    async fn route(
        &self,
        intent: &DeliveryIntent,
        ctx: &RoutingContext,
    ) -> Result<RoutingDecision, ChannelError> {
        let all_channels = {
            let map = self.channels.read().await;
            map.clone()
        };

        let mut ordered: Vec<ChannelDescriptor> = Vec::new();
        let mut reasons: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // 1. Explicit caller preference wins absolutely.
        if let Some(pref_id) = &intent.preferred_channel {
            if let Some(desc) = all_channels.get(pref_id) {
                if desc.healthy {
                    ordered.push(desc.clone());
                    reasons.push(format!("caller preferred '{pref_id}'"));
                    seen.insert(pref_id.clone());
                }
            }
        }

        // 2. Learned preferences for this (namespace, category).
        let prefs = self
            .preferences
            .get_preferences(&intent.namespace, intent.category, ctx.min_weight)
            .await?;

        for pref in &prefs {
            if seen.contains(&pref.channel_id) {
                continue;
            }
            let Some(desc) = all_channels.get(&pref.channel_id) else {
                continue;
            };
            if !desc.healthy {
                continue;
            }
            // Webhook channels can't carry a response — never route `Confirm` to them.
            if !desc.kind.supports_response()
                && intent.category == crate::types::DeliveryCategory::Confirm
            {
                continue;
            }
            ordered.push(desc.clone());
            reasons.push(if pref.pinned {
                format!("pinned preference (weight {:.2})", pref.weight)
            } else {
                format!("learned preference (weight {:.2})", pref.weight)
            });
            seen.insert(pref.channel_id.clone());
        }

        // 3. Urgency escalation — fan out to every response-capable channel.
        if matches!(intent.urgency, UrgencyLevel::High | UrgencyLevel::Critical) {
            for desc in self.response_capable().await {
                if !seen.contains(&desc.id) {
                    ordered.push(desc.clone());
                    reasons.push(format!("urgency={:?} fallout", intent.urgency));
                    seen.insert(desc.id.clone());
                }
            }
        }

        // 4. Initiation channel fallback.
        if let Some(init_id) = &intent.initiation_channel {
            if !seen.contains(init_id) {
                if let Some(desc) = all_channels.get(init_id) {
                    if desc.healthy {
                        ordered.push(desc.clone());
                        reasons.push(format!("initiation channel '{init_id}'"));
                        seen.insert(init_id.clone());
                    }
                }
            }
        }

        // 5. Last-resort: any healthy channel that can carry the category.
        for desc in all_channels.values() {
            if seen.contains(&desc.id) || !desc.healthy {
                continue;
            }
            // Never push an approval request to a webhook — it can't reply.
            if !desc.kind.supports_response()
                && intent.category == crate::types::DeliveryCategory::Confirm
            {
                continue;
            }
            ordered.push(desc.clone());
            reasons.push("fallback (any healthy channel)".to_string());
            seen.insert(desc.id.clone());
        }

        // Trim to the caller's attempt budget.
        ordered.truncate(ctx.max_candidates);
        reasons.truncate(ctx.max_candidates);

        if ordered.is_empty() {
            return Err(ChannelError::NoChannelAvailable(
                intent.category,
                intent.urgency,
            ));
        }

        Ok(RoutingDecision {
            candidates: ordered,
            reasons,
        })
    }

    async fn register(&self, descriptor: ChannelDescriptor) -> Result<(), ChannelError> {
        let mut map = self.channels.write().await;
        map.insert(descriptor.id.clone(), descriptor);
        Ok(())
    }

    async fn set_health(&self, channel_id: &str, healthy: bool) -> Result<(), ChannelError> {
        let mut map = self.channels.write().await;
        let desc = map
            .get_mut(channel_id)
            .ok_or_else(|| ChannelError::UnknownChannel(channel_id.to_string()))?;
        desc.healthy = healthy;
        Ok(())
    }

    async fn list_channels(&self) -> Result<Vec<ChannelDescriptor>, ChannelError> {
        let map = self.channels.read().await;
        let mut v: Vec<ChannelDescriptor> = map.values().cloned().collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preference::SqlitePreferenceStore;
    use crate::types::DeliveryCategory;
    use std::sync::Arc;

    async fn mk_router() -> (DefaultChannelRouter, Arc<SqlitePreferenceStore>) {
        let db = storage::SqlitePool::open_memory().unwrap();
        let store = Arc::new(SqlitePreferenceStore::new(db));
        store.ensure_tables().unwrap();
        let router = DefaultChannelRouter::new(store.clone() as Arc<dyn ChannelPreferenceStore>);
        (router, store)
    }

    fn local() -> ChannelDescriptor {
        ChannelDescriptor::new("cli", ChannelKind::Local, "CLI")
    }
    fn telegram() -> ChannelDescriptor {
        ChannelDescriptor::new("telegram", ChannelKind::Relay, "Telegram")
    }
    fn slack_webhook() -> ChannelDescriptor {
        ChannelDescriptor::new("slack-push", ChannelKind::Webhook, "Slack push")
    }

    #[tokio::test]
    async fn explicit_preferred_wins() {
        let (router, _) = mk_router().await;
        router.register(local()).await.unwrap();
        router.register(telegram()).await.unwrap();

        let intent = DeliveryIntent::new("hi", DeliveryCategory::Response, UrgencyLevel::Normal)
            .with_preferred("telegram");
        let decision = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap();
        assert_eq!(decision.candidates[0].id, "telegram");
        assert!(decision.reasons[0].contains("caller preferred"));
    }

    #[tokio::test]
    async fn learned_preference_wins_over_fallback() {
        let (router, store) = mk_router().await;
        router.register(local()).await.unwrap();
        router.register(telegram()).await.unwrap();

        store
            .upsert_preference(
                "personal",
                DeliveryCategory::Confirm,
                "telegram",
                0.8,
                false,
            )
            .await
            .unwrap();

        let intent =
            DeliveryIntent::new("deploy?", DeliveryCategory::Confirm, UrgencyLevel::Normal);
        let decision = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap();
        assert_eq!(decision.candidates[0].id, "telegram");
    }

    #[tokio::test]
    async fn confirm_never_routes_to_webhook() {
        let (router, _) = mk_router().await;
        router.register(slack_webhook()).await.unwrap();

        let intent =
            DeliveryIntent::new("approve", DeliveryCategory::Confirm, UrgencyLevel::Normal);
        let decision = router.route(&intent, &RoutingContext::default()).await;
        assert!(matches!(
            decision,
            Err(ChannelError::NoChannelAvailable(_, _))
        ));
    }

    #[tokio::test]
    async fn critical_urgency_fans_out() {
        let (router, store) = mk_router().await;
        router.register(local()).await.unwrap();
        router.register(telegram()).await.unwrap();

        // Give telegram a pin so it goes first, but critical urgency should still add local.
        store
            .upsert_preference("personal", DeliveryCategory::Alert, "telegram", 1.0, true)
            .await
            .unwrap();

        let intent = DeliveryIntent::new("fire!", DeliveryCategory::Alert, UrgencyLevel::Critical);
        let decision = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap();
        let ids: Vec<&str> = decision.candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"telegram"));
        assert!(ids.contains(&"cli"));
    }

    #[tokio::test]
    async fn unhealthy_channel_is_skipped() {
        let (router, _) = mk_router().await;
        router.register(local()).await.unwrap();
        router.register(telegram()).await.unwrap();
        router.set_health("telegram", false).await.unwrap();

        let intent = DeliveryIntent::new("ping", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        let decision = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap();
        for c in &decision.candidates {
            assert_ne!(c.id, "telegram");
        }
    }

    #[tokio::test]
    async fn initiation_channel_used_as_fallback() {
        let (router, _) = mk_router().await;
        router.register(local()).await.unwrap();
        router.register(telegram()).await.unwrap();

        let intent = DeliveryIntent::new("hi", DeliveryCategory::Response, UrgencyLevel::Normal)
            .with_initiation("cli");
        let decision = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap();
        // With no learned prefs, CLI (initiation) should appear in the candidate list.
        let ids: Vec<&str> = decision.candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"cli"));
    }

    #[tokio::test]
    async fn no_channels_is_error() {
        let (router, _) = mk_router().await;
        let intent = DeliveryIntent::new("hi", DeliveryCategory::Nudge, UrgencyLevel::Low);
        let err = router
            .route(&intent, &RoutingContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ChannelError::NoChannelAvailable(_, _)));
    }

    #[tokio::test]
    async fn max_candidates_respected() {
        let (router, _) = mk_router().await;
        // Register 5 channels.
        for i in 0..5 {
            router
                .register(ChannelDescriptor::new(
                    format!("ch-{i}"),
                    ChannelKind::Local,
                    format!("Channel {i}"),
                ))
                .await
                .unwrap();
        }
        let intent = DeliveryIntent::new("hi", DeliveryCategory::Nudge, UrgencyLevel::Normal);
        let ctx = RoutingContext {
            max_candidates: 2,
            ..RoutingContext::default()
        };
        let decision = router.route(&intent, &ctx).await.unwrap();
        assert_eq!(decision.candidates.len(), 2);
    }
}
