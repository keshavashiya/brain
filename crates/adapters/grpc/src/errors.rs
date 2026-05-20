//! gRPC adapter error type and `SignalError` → `tonic::Status` mapping.

use signal::SignalError;
use tonic::Status;

#[derive(Debug, thiserror::Error)]
pub enum GrpcAdapterError {
    #[error("Server error: {0}")]
    Server(String),
}

/// Map a `SignalError` to a sanitized gRPC `Status` (Issue 44). Internal
/// detail (storage paths, SQL strings) stays in tracing logs only.
pub(crate) fn public_status(err: &SignalError) -> Status {
    let public = err.to_public();
    let code = match err {
        SignalError::Init(_) => tonic::Code::Unavailable,
        SignalError::Storage(_) => tonic::Code::Unavailable,
        SignalError::Llm(_) | SignalError::Processing(_) => tonic::Code::Internal,
    };
    Status::new(code, public.message)
}
