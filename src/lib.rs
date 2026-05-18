//! Unified Federated Intelligence Backend (FedCtx)
//!
//! Core modules: HNSW VectorDB + SHA-256 Audit Chain + REST API + gRPC
//! + Trait abstractions (VectorStore, GraphStore, AuditLog, ScenarioPlugin, MemoryStore)
//! + Runtime (scenario registry and server bootstrap)
//! + Memory Store (HNSW-backed semantic memory with Ebbinghaus forgetting)

pub mod vector_db;
pub mod audit;
pub mod storage;
pub mod grpc_service;
pub mod rest_api;
pub mod med_gps_service;

// FedCtx trait layer
pub mod traits;
pub mod hnsw_store;
pub mod json_graph_store;
pub mod chain_audit_log;

// FedCtx runtime
pub mod runtime;
pub mod med_gps_plugin;
pub mod mcp_server;
pub mod memory_store;

pub mod proto {
    tonic::include_proto!("federated_ai");
}

pub use vector_db::SharedVectorDB;
pub use audit::SharedAuditChain;

// Re-export trait types for convenience
pub use traits::{
    VectorStore, GraphStore, AuditLog, ScenarioPlugin, MemoryStore,
    VectorEntry, SearchHit, VectorStoreStats, VectorStoreError,
    GraphNode, GraphEdge, GraphPath, GraphStoreStats, GraphStoreError,
    AuditEntry, AuditLogError, CoreStores,
    MemoryEntry, MemoryRole, MemoryHit, MemoryStoreError, MemoryStoreStats,
};
pub use hnsw_store::HnswVectorStore;
pub use json_graph_store::JsonGraphStore;
pub use chain_audit_log::ChainAuditLog;
pub use runtime::FedCtxRuntime;
pub use med_gps_plugin::MedGpsPlugin;
pub use mcp_server::FedCtxMcpServer;
pub use memory_store::HnswMemoryStore;
