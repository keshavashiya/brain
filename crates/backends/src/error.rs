//! Typed errors for backend construction.
//!
//! Backends previously surfaced construction failures as `anyhow::Error`,
//! which erased the failure mode for callers. Every backend constructor
//! fails for exactly one reason — the underlying reqwest HTTP client can't
//! be built — so a single typed variant captures it while letting callers
//! match on it (and `anyhow`-using callers still get `?` via the
//! `std::error::Error` impl).

use thiserror::Error;

/// Failure constructing a backend's HTTP client.
#[derive(Debug, Error)]
pub enum BackendInitError {
    /// `reqwest::Client::builder().build()` failed. The first field labels
    /// which backend (e.g. `"message client"`); the source is the reqwest
    /// error.
    #[error("{0} init failed: {1}")]
    HttpClient(&'static str, #[source] reqwest::Error),
}
