//! gRPC service-impl structs and their shared constructor/principal-resolution
//! glue. The actual `tonic::async_trait` impls live in `handlers::memory` and
//! `handlers::agent`.

use std::sync::Arc;

use tonic::Request;

use crate::auth::resolve_principal_from_metadata;

/// gRPC implementation of `MemoryService`.
pub struct MemoryServiceImpl {
    pub(crate) processor: Arc<signal::SignalProcessor>,
    pub(crate) api_keys: Arc<Vec<brain::ApiKeyConfig>>,
}

impl MemoryServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        let api_keys = Arc::new(processor.config().access.api_keys.clone());
        Self {
            processor,
            api_keys,
        }
    }

    /// Resolve the `Principal` from the request metadata. Returns `None`
    /// for keys without `agent_id` (back-compat).
    pub(crate) async fn resolve_principal<T>(
        &self,
        req: &Request<T>,
    ) -> Option<identity::Principal> {
        resolve_principal_from_metadata(req, &self.api_keys, self.processor.as_ref()).await
    }
}

/// gRPC implementation of `AgentService`.
pub struct AgentServiceImpl {
    pub(crate) processor: Arc<signal::SignalProcessor>,
    pub(crate) api_keys: Arc<Vec<brain::ApiKeyConfig>>,
}

impl AgentServiceImpl {
    pub fn new(processor: Arc<signal::SignalProcessor>) -> Self {
        let api_keys = Arc::new(processor.config().access.api_keys.clone());
        Self {
            processor,
            api_keys,
        }
    }

    pub(crate) async fn resolve_principal<T>(
        &self,
        req: &Request<T>,
    ) -> Option<identity::Principal> {
        resolve_principal_from_metadata(req, &self.api_keys, self.processor.as_ref()).await
    }
}
