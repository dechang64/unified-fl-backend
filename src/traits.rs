//! FedCtx Core Traits
//!
//! Abstract interfaces for the federated semantic infrastructure.
//! All concrete implementations (HNSW, GraphLite, etc.) implement these traits.
//! Scenario plugins depend on traits, not concrete types.

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ════════════════════════════════════════════════════════════════
// Vector Store
// ════════════════════════════════════════════════════════════════

/// A vector entry with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

/// A search result from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

/// Error type for vector store operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty vector values for id '{id}'")]
    EmptyValues { id: String },
    #[error("internal error: {0}")]
    Internal(String),
}

/// Abstract vector store — HNSW, FAISS, Qdrant, etc.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert vectors. Returns count inserted.
    async fn insert(&self, entries: Vec<VectorEntry>) -> Result<usize, VectorStoreError>;

    /// Search for k nearest neighbors.
    async fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<SearchHit>, VectorStoreError>;

    /// Delete vectors by ID. Returns count deleted.
    async fn delete(&self, ids: &[String]) -> Result<usize, VectorStoreError>;

    /// Get store statistics.
    async fn stats(&self) -> VectorStoreStats;
}

/// Vector store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub index_type: String,
}

// ════════════════════════════════════════════════════════════════
// Graph Store
// ════════════════════════════════════════════════════════════════

/// A node in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub properties: HashMap<String, String>,
}

/// An edge in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: String,
    pub properties: HashMap<String, String>,
}

/// Result of a graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub total_weight: f32,
}

/// Error type for graph store operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphStoreError {
    #[error("node not found: {0}")]
    NodeNotFound(String),
    #[error("query error: {0}")]
    QueryError(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Abstract graph store — GraphLite, Neo4j, JSON-backed, etc.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Add a node.
    async fn add_node(&self, node: GraphNode) -> Result<(), GraphStoreError>;

    /// Add an edge.
    async fn add_edge(&self, edge: GraphEdge) -> Result<(), GraphStoreError>;

    /// Traverse from a node, following edges with the given label, up to `max_hops`.
    async fn traverse(
        &self,
        start_id: &str,
        edge_label: Option<&str>,
        max_hops: usize,
    ) -> Result<GraphPath, GraphStoreError>;

    /// Get a node by ID.
    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, GraphStoreError>;

    /// Get neighbors of a node.
    async fn neighbors(
        &self,
        node_id: &str,
        edge_label: Option<&str>,
    ) -> Result<Vec<GraphNode>, GraphStoreError>;

    /// Get graph statistics.
    async fn stats(&self) -> GraphStoreStats;
}

/// Graph store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStoreStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_labels: Vec<String>,
    pub edge_labels: Vec<String>,
}

// ════════════════════════════════════════════════════════════════
// Audit Log
// ════════════════════════════════════════════════════════════════

/// An audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub sequence: u64,
    pub timestamp: String,
    pub event_type: String,
    pub node_id: String,
    pub metadata: HashMap<String, String>,
    pub hash: String,
    pub prev_hash: String,
}

/// Error type for audit log operations.
#[derive(Debug, thiserror::Error)]
pub enum AuditLogError {
    #[error("chain verification failed: {0}")]
    ChainBroken(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Abstract audit log — SHA-256 chain, Merkle tree, external audit service, etc.
#[async_trait]
pub trait AuditLog: Send + Sync {
    /// Append an event to the audit log.
    async fn append(
        &self,
        event_type: &str,
        node_id: &str,
        metadata: HashMap<String, String>,
    ) -> Result<AuditEntry, AuditLogError>;

    /// Verify the integrity of the audit chain.
    async fn verify(&self) -> Result<(), AuditLogError>;

    /// Get recent entries.
    async fn recent(&self, n: usize) -> Vec<AuditEntry>;

    /// Get total entry count.
    async fn len(&self) -> usize;

    /// Whether the audit log is empty.
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// ════════════════════════════════════════════════════════════════
// Scenario Plugin
// ════════════════════════════════════════════════════════════════

/// A scenario plugin — implements domain-specific logic on top of FedCtx.
///
/// Each scenario (medical, financial, industrial) implements this trait
/// and registers with the FedCtx runtime.
///
/// gRPC service registration is done at compile time via the
/// `register_scenario!` macro (tonic requires concrete types).
/// REST routes can be added dynamically.
#[async_trait]
pub trait ScenarioPlugin: Send + Sync + 'static {
    /// Unique identifier for this scenario.
    fn id(&self) -> &str;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Description of what this scenario does.
    fn description(&self) -> &str;

    /// Initialize the plugin with access to core stores.
    /// Called once at startup.
    async fn initialize(
        &self,
        stores: CoreStores,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Register REST route(s) with the axum router.
    /// Returns None if the plugin doesn't need REST.
    fn rest_routes(&self) -> Option<axum::Router> {
        None
    }

    /// Shutdown hook.
    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════
// Memory Store
// ════════════════════════════════════════════════════════════════

/// A memory entry — stored as a vector in HNSW with rich metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub role: MemoryRole,
    pub session_id: Option<String>,
    pub timestamp: String,
    pub access_count: u64,
    pub importance: f32,
    pub metadata: HashMap<String, String>,
}

/// Role of a memory entry in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryRole {
    User,
    Assistant,
    System,
    Observation,
    Reflection,
}

/// Result of a memory recall operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub entry: MemoryEntry,
    pub score: f32,
    /// Retention score based on Ebbinghaus forgetting curve.
    pub retention: f32,
}

/// Error type for memory store operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryStoreError {
    #[error("memory not found: {0}")]
    NotFound(String),
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),
    #[error("persistence error: {0}")]
    PersistenceError(String),
    #[error("internal error: {0}")]
    Internal(String),
}

/// Abstract memory store — backed by HNSW for semantic recall.
///
/// Memory entries are stored as vectors in the same HNSW index as
/// business data, differentiated by metadata.type = "memory".
/// This enables cross-domain recall: "what did I discover about X?"
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a new memory entry. Returns the entry with generated ID.
    async fn remember(
        &self,
        content: &str,
        role: MemoryRole,
        session_id: Option<&str>,
        importance: f32,
        metadata: HashMap<String, String>,
    ) -> Result<MemoryEntry, MemoryStoreError>;

    /// Recall memories by semantic similarity.
    async fn recall(
        &self,
        query: &str,
        k: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryHit>, MemoryStoreError>;

    /// Get recent memories, ordered by timestamp (newest first).
    async fn recent(
        &self,
        n: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError>;

    /// Touch a memory entry (increment access count, boost retention).
    async fn touch(&self, id: &str) -> Result<(), MemoryStoreError>;

    /// Get a specific memory by ID.
    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, MemoryStoreError>;

    /// Forget (soft-delete) a memory entry.
    async fn forget(&self, id: &str) -> Result<bool, MemoryStoreError>;

    /// Get memory store statistics.
    async fn stats(&self) -> MemoryStoreStats;
}

/// Memory store statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStoreStats {
    pub total_memories: usize,
    pub active_memories: usize,
    pub cold_memories: usize,
    pub sessions: usize,
}

/// Core stores passed to scenario plugins at initialization.
///
/// All stores use trait objects for uniform abstraction — plugins depend
/// on traits, not concrete types. This allows swapping implementations
/// (e.g., HnswVectorStore → QdrantVectorStore) without changing plugin code.
#[derive(Clone)]
pub struct CoreStores {
    pub vector_store: std::sync::Arc<dyn VectorStore>,
    pub graph_store: std::sync::Arc<dyn GraphStore>,
    pub audit_log: std::sync::Arc<dyn AuditLog>,
    pub memory_store: std::sync::Arc<dyn MemoryStore>,
    /// Federated learning aggregation engine.
    pub fl_engine: Option<std::sync::Arc<crate::fl_engine::FlEngine>>,
    /// Text-to-vector embedding provider.
    pub embedding: Option<std::sync::Arc<dyn crate::embedding::EmbeddingProvider>>,
}
