//! GraphRAG gRPC Service Implementation

use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::proto::{
    graph_rag_service_server::GraphRagService,
    GraphRagQuery, GraphRagResponse,
    ScoredNodeProto, SubgraphEdgeProto,
};
use crate::graphrag::GraphRAGEngine;

pub struct GraphRAGServiceImpl {
    engine: Arc<GraphRAGEngine>,
}

impl GraphRAGServiceImpl {
    pub fn new(engine: Arc<GraphRAGEngine>) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl GraphRagService for GraphRAGServiceImpl {
    async fn query(
        &self,
        req: Request<GraphRagQuery>,
    ) -> Result<Response<GraphRagResponse>, Status> {
        let req = req.into_inner();

        let config = crate::graphrag::GraphRAGConfig {
            top_k: if req.top_k > 0 { req.top_k as usize } else { 10 },
            max_hops: if req.max_hops > 0 { req.max_hops as usize } else { 2 },
            max_nodes: if req.max_nodes > 0 { req.max_nodes as usize } else { 30 },
            min_score: if req.min_score > 0.0 { req.min_score as f32 } else { 0.1 },
            vector_weight: if req.vector_weight > 0.0 { req.vector_weight as f32 } else { 0.7 },
            centrality_weight: if req.centrality_weight > 0.0 { req.centrality_weight as f32 } else { 0.3 },
        };

        let engine = crate::graphrag::GraphRAGEngine::new_with_config(
            self.engine.vector_store(),
            self.engine.graph_store(),
            self.engine.embedding(),
            config,
        );

        let start = std::time::Instant::now();
        let result = engine.query(&req.query).await
            .map_err(|e| Status::internal(format!("GraphRAG query failed: {}", e)))?;
        let elapsed = start.elapsed().as_millis() as f64;

        let nodes: Vec<ScoredNodeProto> = result.nodes.into_iter().map(|n| ScoredNodeProto {
            id: n.id,
            label: n.label,
            properties: n.properties,
            vector_score: n.vector_score,
            centrality_score: n.centrality_score,
            combined_score: n.combined_score,
        }).collect();
        let edges = result.edges.into_iter().map(|e| SubgraphEdgeProto {
            id: e.id,
            source: e.source,
            target: e.target,
            label: e.label,
        }).collect();
        let total_candidates = result.candidates_considered as i32;
        let total_expanded = nodes.len() as i32;

        Ok(Response::new(GraphRagResponse {
            query: req.query,
            nodes,
            edges,
            total_candidates,
            total_expanded,
            retrieval_time_ms: elapsed,
        }))
    }
}
