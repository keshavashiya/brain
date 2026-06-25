//! Answer-quality fitness (L1) — `SignalProcessor` orchestration that wires
//! the pure [`crate::answer_fitness`] classifiers, judge, and selector into the
//! single chat entry point.
//!
//! The hot-path cost is deliberately minimal: kind classification reuses the
//! query embedding the tool advertiser already needs, and the fast-tier
//! follow-up judge runs off the hot path (a spawned task), so a chat turn never
//! waits on a quality judgement before answering.

use cortex::llm::TaskTier;

use crate::answer_fitness::{self, PendingAnswer, TaskKind};
use crate::SignalProcessor;

/// Cap on the in-memory pending-judgement map, so a caller spinning up sessions
/// can't grow it without bound. On overflow we drop the whole map — the learned
/// signal is best-effort, and the next turn simply re-seeds it.
const MAX_PENDING_SESSIONS: usize = 1024;

impl SignalProcessor {
    /// Per-kind anchor embeddings for semantic kind classification, embedded
    /// once (lazily) from [`TaskKind::anchor_corpus`] through the capability
    /// embedder. Empty when no embedder is wired → callers fall back to keyword
    /// classification. Anchors are catalog text (not namespaced user data), so
    /// they embed unconditionally via the descriptor surface.
    pub(crate) async fn answer_anchors(&self) -> &[(TaskKind, Vec<f32>)] {
        self.answer_anchors
            .get_or_init(|| async {
                use intent::DescriptorEmbedder;
                let Some(embedder) = self.capability_embedder() else {
                    return Vec::new();
                };
                let mut out = Vec::new();
                for (kind, text) in TaskKind::anchor_corpus() {
                    if let Some(emb) = embedder.embed_descriptor(&text).await {
                        out.push((kind, emb));
                    }
                }
                out
            })
            .await
    }

    /// Which tier should serve `kind`, given the learned per-`(kind, model)`
    /// quality of each tier's current model. [`TaskTier::Deep`] (the static
    /// default) unless a cheaper tier with a different model has measurably
    /// better answers for this kind — see [`answer_fitness::select_tier`].
    pub(crate) fn answer_tier_for(&self, kind: TaskKind) -> TaskTier {
        let cfg = &self.config().learning.answer_fitness;
        if !cfg.enabled {
            return TaskTier::Deep;
        }
        let k = kind.as_str();
        let dm = answer_fitness::model_key(self.llm().name(), self.llm().model());
        let balanced = self.llm_tier(TaskTier::Balanced);
        let fast = self.llm_tier(TaskTier::Fast);
        let bm = answer_fitness::model_key(balanced.name(), balanced.model());
        let fm = answer_fitness::model_key(fast.name(), fast.model());

        let read = |model: &str| self.answer_fitness().quality(k, model).ok().flatten();
        let (dq, bq, fq) = (read(&dm), read(&bm), read(&fm));
        answer_fitness::select_tier(
            (&dm, dq.as_ref()),
            (&bm, bq.as_ref()),
            (&fm, fq.as_ref()),
            cfg.min_judged_turns,
            cfg.margin,
        )
    }

    /// Score the answer this session is awaiting (if any) against the user's
    /// `follow_up`, off the hot path. Takes the pending record, spawns the
    /// fast-tier judge, and records the outcome. No-op when answer-quality
    /// learning is off or there is nothing pending for this session.
    pub(crate) fn score_pending_answer(&self, key: &str, follow_up: &str) {
        if !self.answer_fitness().enabled() {
            return;
        }
        let pending = match self.answer_pending.lock() {
            Ok(mut map) => map.remove(key),
            Err(_) => None,
        };
        let Some(pending) = pending else {
            return;
        };
        // The fast tier judges; recording is a cheap SQLite write. Spawned so
        // the current turn never blocks on a quality judgement.
        let llm = self.llm_tier(TaskTier::Fast);
        let store = self.answer_fitness().clone();
        let follow_up = follow_up.to_string();
        tokio::spawn(async move {
            let outcome = answer_fitness::judge_outcome(
                &pending.prev_user,
                &pending.prev_answer,
                &follow_up,
                llm.as_ref(),
            )
            .await;
            if let Err(e) = store.record(pending.kind.as_str(), &pending.model, outcome) {
                tracing::debug!(error = %e, "answer-quality record failed");
            }
        });
    }

    /// Remember this turn's answer so the next turn in the same conversation
    /// can judge it. No-op when answer-quality learning is off.
    pub(crate) fn remember_pending_answer(&self, key: String, pending: PendingAnswer) {
        if !self.answer_fitness().enabled() {
            return;
        }
        if let Ok(mut map) = self.answer_pending.lock() {
            if map.len() >= MAX_PENDING_SESSIONS {
                map.clear();
            }
            map.insert(key, pending);
        }
    }

    /// A compact capability-digest line naming the task kinds whose routing the
    /// learned answer-quality has shifted away from the deep tier, or `None`
    /// when nothing has shifted (the common case) so the digest stays quiet.
    pub(crate) fn answer_quality_digest_line(&self) -> Option<String> {
        if !self.answer_fitness().enabled() {
            return None;
        }
        let shifted: Vec<String> = TaskKind::anchored()
            .into_iter()
            .filter_map(|kind| {
                let tier = self.answer_tier_for(kind);
                (tier != TaskTier::Deep).then(|| format!("{} → {}", kind.as_str(), tier.as_str()))
            })
            .collect();
        if shifted.is_empty() {
            None
        } else {
            Some(format!(
                "Answer quality (learned): routing {} to a better-fit local model for these.",
                shifted.join(", ")
            ))
        }
    }
}
