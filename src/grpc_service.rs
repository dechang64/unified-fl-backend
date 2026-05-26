//! gRPC Service Implementations

use crate::proto::{
    vector_database_server::{VectorDatabase, VectorDatabaseServer},
    audit_service_server::{AuditService, AuditServiceServer},
    *,
};
use crate::vector_db::SharedVectorDB;
use crate::audit::{SharedAuditChain, EventType};
use tonic::{Request, Response, Status};

pub struct VectorDBService {
    db: SharedVectorDB,
    audit: SharedAuditChain,
}

impl VectorDBService {
    pub fn new(db: SharedVectorDB, audit: SharedAuditChain) -> Self {
        Self { db, audit }
    }
}

#[tonic::async_trait]
impl VectorDatabase for VectorDBService {
    async fn insert(&self, req: Request<InsertRequest>) -> Result<Response<InsertResponse>, Status> {
        let vectors = req.into_inner().vectors;
        let count = vectors.len();

        // Validate: reject empty batch
        if count == 0 {
            return Ok(Response::new(InsertResponse { inserted: 0 }));
        }

        let mut db = self.db.write();
        let mut inserted = 0i32;

        for v in vectors {
            // Validate: reject empty vector values
            if v.values.is_empty() {
                tracing::warn!("Skipping insert for '{}': empty values", v.id);
                continue;
            }
            let entry = crate::vector_db::VectorEntry {
                id: v.id,
                values: v.values,
                metadata: v.metadata,
            };
            match db.insert(entry) {
                Ok(_) => inserted += 1,
                Err(e) => tracing::warn!("Insert failed: {}", e),
            }
        }

        drop(db);
        {
            let mut audit = self.audit.write();
            audit.append(
                EventType::VectorInsert { count: inserted as usize },
                "grpc",
                serde_json::json!({"source": "grpc", "batch_size": count}),
            );
        }

        Ok(Response::new(InsertResponse { inserted }))
    }

    async fn search(&self, req: Request<SearchRequest>) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();
        let k = if req.k > 0 { req.k as usize } else { 5 };
        let query = req.query;

        // Validate: reject empty query
        if query.is_empty() {
            return Ok(Response::new(SearchResponse { results: vec![] }));
        }

        let db = self.db.read();
        let hits = db.search(&query, k);
        drop(db);

        {
            let mut audit = self.audit.write();
            audit.append(
                EventType::VectorSearch { k, result_count: hits.len() },
                "grpc",
                serde_json::json!({}),
            );
        }

        let results = hits.into_iter().map(|h| SearchResult {
            id: h.id,
            values: vec![],
            distance: 1.0 - h.score,
            metadata: h.metadata,
        }).collect();

        Ok(Response::new(SearchResponse { results }))
    }

    async fn delete(&self, req: Request<DeleteRequest>) -> Result<Response<DeleteResponse>, Status> {
        let ids: Vec<String> = req.into_inner().ids;
        let count = ids.len();

        if count == 0 {
            return Ok(Response::new(DeleteResponse { deleted: 0 }));
        }

        let mut db = self.db.write();
        let deleted = db.delete(&ids) as i32;
        drop(db);

        {
            let mut audit = self.audit.write();
            audit.append(
                EventType::VectorDelete { count: deleted as usize },
                "grpc",
                serde_json::json!({"requested": count}),
            );
        }

        Ok(Response::new(DeleteResponse { deleted }))
    }

    async fn stats(&self, _req: Request<StatsRequest>) -> Result<Response<StatsResponse>, Status> {
        let db = self.db.read();
        let audit = self.audit.read();
        Ok(Response::new(StatsResponse {
            total_vectors: db.len() as i64,
            dimension: db.dimension() as i32,
            index_size: db.len() as i64,
            audit_chain_length: audit.len() as i64,
            audit_chain_valid: audit.verify().is_ok(),
        }))
    }
}

pub struct AuditServiceImpl {
    audit: SharedAuditChain,
    start_time: std::time::Instant,
}

impl AuditServiceImpl {
    pub fn new(audit: SharedAuditChain) -> Self {
        Self {
            audit,
            start_time: std::time::Instant::now(),
        }
    }
}

#[tonic::async_trait]
impl AuditService for AuditServiceImpl {
    async fn get_trail(&self, req: Request<AuditQuery>) -> Result<Response<AuditTrailResponse>, Status> {
        let req = req.into_inner();
        let limit = if req.limit > 0 { req.limit as usize } else { 50 };
        let audit = self.audit.read();
        let chain_valid = audit.verify().is_ok();
        let entries = audit.to_proto_entries(limit);

        Ok(Response::new(AuditTrailResponse { entries, chain_valid }))
    }

    async fn health(&self, _req: Request<HealthRequest>) -> Result<Response<HealthResponse>, Status> {
        let audit = self.audit.read();
        let chain_valid = audit.verify().is_ok();
        Ok(Response::new(HealthResponse {
            status: if chain_valid { "healthy".into() } else { "degraded".into() },
            version: env!("CARGO_PKG_VERSION").into(),
            uptime_seconds: self.start_time.elapsed().as_secs() as i64,
        }))
    }
}

pub fn create_grpc_server(
    db: SharedVectorDB,
    audit: SharedAuditChain,
) -> VectorDatabaseServer<VectorDBService> {
    VectorDatabaseServer::new(VectorDBService::new(db, audit))
}

pub fn create_audit_server(audit: SharedAuditChain) -> AuditServiceServer<AuditServiceImpl> {
    AuditServiceServer::new(AuditServiceImpl::new(audit))
}
