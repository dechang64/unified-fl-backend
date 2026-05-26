//! Embedding gRPC Service Implementation

use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::proto::{
    embedding_service_server::EmbeddingService,
    EmbedRequest, EmbedResponse, EmbeddingResult,
};
use crate::embedding::EmbeddingProvider;

pub struct EmbeddingServiceImpl {
    provider: Arc<dyn EmbeddingProvider>,
}

impl EmbeddingServiceImpl {
    pub fn new(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self { provider }
    }
}

#[tonic::async_trait]
impl EmbeddingService for EmbeddingServiceImpl {
    async fn embed(
        &self,
        req: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = req.into_inner();
        if req.texts.is_empty() {
            return Err(Status::invalid_argument("texts cannot be empty"));
        }

        let texts: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
        let results = self.provider.embed_batch(&texts).await
            .map_err(|e| Status::internal(format!("Embedding failed: {}", e)))?;

        let dimension = results.first().map(|v| v.len()).unwrap_or(0);

        Ok(Response::new(EmbedResponse {
            results: texts.into_iter().zip(results).map(|(text, values)| {
                EmbeddingResult {
                    text: text.to_string(),
                    values,
                }
            }).collect(),
            provider: self.provider.name().to_string(),
            dimension: dimension as i32,
        }))
    }
}
