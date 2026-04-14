//! # Brain Audit Trail
//!
//! Immutable log of every autonomous action: who requested, what decided,
//! who approved, outcome, duration, stdout/stderr. Queryable via natural
//! language and any protocol.
//!
//! Append-only enforcement: rows are never updated or deleted. Pruning
//! requires explicit `brain audit prune` command with age threshold.

pub mod query;
pub mod rollback;
pub mod schema;

pub use query::AuditQuerySpec;
pub use rollback::RollbackPlan;
pub use schema::{ActionTier, AuditEntry, AuditError, AuditOutcome, AuditTrail, SqliteAuditTrail};
