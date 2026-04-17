//! Errors surfaced by channel components.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ChannelError {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::sqlite::SqliteError),

    #[error("Confirmation engine error: {0}")]
    Confirm(#[from] confirm::ConfirmError),

    #[error("No channel available for category {0:?} at urgency {1:?}")]
    NoChannelAvailable(crate::types::DeliveryCategory, crate::types::UrgencyLevel),

    #[error("Channel not registered: {0}")]
    UnknownChannel(String),

    #[error("Invalid preference weight: {0} (must be 0.0..=1.0)")]
    InvalidWeight(f32),

    #[error("Delivery failed on all candidates: {0}")]
    DeliveryFailed(String),

    #[error("Correlation parse error: {0}")]
    CorrelationParse(String),

    #[error("Relay error: {0}")]
    Relay(String),
}
