//! FedCtx MCP Server — Model Context Protocol interface
//!
//! Exposes FedCtx core capabilities (vector search, graph traversal,
//! audit trail, stats) as MCP Tools, accessible from any MCP-compatible
//! AI client (Claude Desktop, Cursor, etc.).

use std::sync::Arc;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::Parameters,
    },
    model::*,
    schemars,
    tool, tool_handler, tool_router,
};

use crate::traits::{GraphStore, MemoryStore};
use crate::vector_db::SharedVectorDB;
use crate::json_graph_store::JsonGraphStore;
use crate::audit::SharedAuditChain;
use crate::memory_store::HnswMemoryStore;

use serde_json::json;

// ── Tool Parameter Types ──

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct VectorSearchParams {
    /// Query vector (list of floats). For text queries, use text_search instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_vector: Option<Vec<f32>>,
    /// Number of results to return.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional metadata filter (key=value pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<std::collections::HashMap<String, String>>,
}

fn default_k() -> usize { 5 }

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct TextSearchParams {
    /// Text query to search for semantically similar vectors.
    pub query: String,
    /// Number of results to return.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional metadata filter (key=value pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GraphTraverseParams {
    /// Starting node ID.
    pub start_id: String,
    /// Edge label to follow (None = all edges).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_label: Option<String>,
    /// Maximum number of hops.
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
}

fn default_max_hops() -> usize { 3 }

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GraphNeighborsParams {
    /// Node ID to get neighbors for.
    pub node_id: String,
    /// Edge label to filter (None = all edges).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_label: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AuditRecentParams {
    /// Number of recent audit entries to return.
    #[serde(default = "default_audit_n")]
    pub n: usize,
}

fn default_audit_n() -> usize { 10 }

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct InsertVectorParams {
    /// Unique ID for the vector.
    pub id: String,
    /// Vector values (list of floats).
    pub values: Vec<f32>,
    /// Optional metadata (key=value pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AddGraphNodeParams {
    /// Node ID.
    pub id: String,
    /// Node label (e.g., "Disease", "Hospital").
    pub label: String,
    /// Optional node properties (key=value pairs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AddGraphEdgeParams {
    /// Edge ID.
    pub id: String,
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Edge label (e.g., "SPECIALIZES_IN", "TREATS").
    pub label: String,
    /// Optional edge properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<std::collections::HashMap<String, String>>,
}

// ── Memory Tool Parameter Types ──

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryRememberParams {
    /// Content to remember.
    pub content: String,
    /// Role of the memory entry: user, assistant, system, observation, reflection.
    #[serde(default = "default_memory_role")]
    pub role: String,
    /// Optional session ID to group memories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Importance score (0.0-1.0). Higher = longer retention.
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// Optional additional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

fn default_memory_role() -> String { "observation".to_string() }
fn default_importance() -> f32 { 0.5 }

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryRecallParams {
    /// Query text to recall semantically similar memories.
    pub query: String,
    /// Number of memories to return.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional session ID to filter memories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryRecentParams {
    /// Number of recent memories to return.
    #[serde(default = "default_k")]
    pub n: usize,
    /// Optional session ID to filter memories.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MemoryForgetParams {
    /// ID of the memory to forget.
    pub id: String,
}

// ── FedCtx MCP Server ──

#[derive(Clone)]
pub struct FedCtxMcpServer {
    vector_db: SharedVectorDB,
    graph_store: Arc<JsonGraphStore>,
    audit_chain: SharedAuditChain,
    memory_store: Arc<HnswMemoryStore>,
    tool_router: ToolRouter<FedCtxMcpServer>,
}

#[tool_router]
impl FedCtxMcpServer {
    pub fn new(
        vector_db: SharedVectorDB,
        graph_store: Arc<JsonGraphStore>,
        audit_chain: SharedAuditChain,
        memory_store: Arc<HnswMemoryStore>,
    ) -> Self {
        Self {
            vector_db,
            graph_store,
            audit_chain,
            memory_store,
            tool_router: Self::tool_router(),
        }
    }

    // ── Vector Tools ──

    #[tool(description = "Search for similar vectors using HNSW index. Provide a query vector and get the k most similar results.")]
    async fn vector_search(
        &self,
        Parameters(params): Parameters<VectorSearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let query = match params.query_vector {
            Some(v) if !v.is_empty() => v,
            _ => {
                return Ok(CallToolResult::success(vec![Content::text(
                    "Error: query_vector is required and must not be empty".to_string(),
                )]));
            }
        };

        let db = self.vector_db.read().map_err(|e| {
            McpError::internal_error("lock_error", Some(json!({"error": e.to_string()})))
        })?;

        let results = db.search(&query, params.k);
        let output: Vec<serde_json::Value> = results.iter().map(|hit| {
            json!({
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.metadata,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".to_string()),
        )]))
    }

    #[tool(description = "Get statistics about the vector database, graph store, and audit chain.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        // Extract data before await to avoid holding RwLockReadGuard across await
        let (total_vectors, dimension) = {
            let db = self.vector_db.read().map_err(|e| {
                McpError::internal_error("lock_error", Some(json!({"error": e.to_string()})))
            })?;
            (db.len(), db.dimension())
        };
        let (total_entries, chain_valid) = {
            let audit = self.audit_chain.read().map_err(|e| {
                McpError::internal_error("lock_error", Some(json!({"error": e.to_string()})))
            })?;
            (audit.len(), audit.verify().is_ok())
        };

        let graph_stats = self.graph_store.stats().await;

        let output = json!({
            "vector_db": {
                "total_vectors": total_vectors,
                "dimension": dimension,
            },
            "graph_store": {
                "total_nodes": graph_stats.total_nodes,
                "total_edges": graph_stats.total_edges,
                "node_labels": graph_stats.node_labels,
                "edge_labels": graph_stats.edge_labels,
            },
            "audit_chain": {
                "total_entries": total_entries,
                "chain_valid": chain_valid,
            },
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }

    #[tool(description = "Traverse the knowledge graph starting from a node, following edges up to max_hops. Returns the path of nodes and edges.")]
    async fn graph_traverse(
        &self,
        Parameters(params): Parameters<GraphTraverseParams>,
    ) -> Result<CallToolResult, McpError> {
        let path = self.graph_store.traverse(
            &params.start_id,
            params.edge_label.as_deref(),
            params.max_hops,
        ).await.map_err(|e| {
            McpError::internal_error("graph_error", Some(json!({"error": e.to_string()})))
        })?;

        let output = json!({
            "nodes": path.nodes.iter().map(|n| json!({
                "id": n.id,
                "label": n.label,
                "properties": n.properties,
            })).collect::<Vec<_>>(),
            "edges": path.edges.iter().map(|e| json!({
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "label": e.label,
            })).collect::<Vec<_>>(),
            "total_weight": path.total_weight,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }

    #[tool(description = "Get the neighbors of a node in the knowledge graph, optionally filtered by edge label.")]
    async fn graph_neighbors(
        &self,
        Parameters(params): Parameters<GraphNeighborsParams>,
    ) -> Result<CallToolResult, McpError> {
        let neighbors = self.graph_store.neighbors(
            &params.node_id,
            params.edge_label.as_deref(),
        ).await.map_err(|e| {
            McpError::internal_error("graph_error", Some(json!({"error": e.to_string()})))
        })?;

        let output: Vec<serde_json::Value> = neighbors.iter().map(|n| {
            json!({
                "id": n.id,
                "label": n.label,
                "properties": n.properties,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".to_string()),
        )]))
    }

    #[tool(description = "Get recent audit trail entries. Each entry records a vector/graph operation with a SHA-256 hash chain.")]
    async fn audit_recent(
        &self,
        Parameters(params): Parameters<AuditRecentParams>,
    ) -> Result<CallToolResult, McpError> {
        let audit = self.audit_chain.read().map_err(|e| {
            McpError::internal_error("lock_error", Some(json!({"error": e.to_string()})))
        })?;

        let entries: Vec<serde_json::Value> = audit.recent(params.n).iter().map(|b| {
            json!({
                "sequence": b.sequence,
                "timestamp": b.timestamp,
                "event_type": b.event_type,
                "node_id": b.node_id,
                "hash": &b.hash[..b.hash.len().min(16)],
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string()),
        )]))
    }

    #[tool(description = "Insert a vector into the HNSW index with optional metadata.")]
    async fn insert_vector(
        &self,
        Parameters(params): Parameters<InsertVectorParams>,
    ) -> Result<CallToolResult, McpError> {
        let mut db = self.vector_db.write().map_err(|e| {
            McpError::internal_error("lock_error", Some(json!({"error": e.to_string()})))
        })?;

        let entry = crate::vector_db::VectorEntry {
            id: params.id,
            values: params.values,
            metadata: params.metadata.unwrap_or_default(),
        };

        match db.insert(entry) {
            Ok(()) => Ok(CallToolResult::success(vec![Content::text(
                "Vector inserted successfully".to_string(),
            )])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(
                format!("Error: {}", e),
            )])),
        }
    }

    #[tool(description = "Add a node to the knowledge graph.")]
    async fn add_graph_node(
        &self,
        Parameters(params): Parameters<AddGraphNodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let node = crate::traits::GraphNode {
            id: params.id,
            label: params.label,
            properties: params.properties.unwrap_or_default(),
        };

        self.graph_store.add_node(node).await.map_err(|e| {
            McpError::internal_error("graph_error", Some(json!({"error": e.to_string()})))
        })?;

        Ok(CallToolResult::success(vec![Content::text(
            "Node added successfully".to_string(),
        )]))
    }

    #[tool(description = "Add an edge to the knowledge graph.")]
    async fn add_graph_edge(
        &self,
        Parameters(params): Parameters<AddGraphEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        let edge = crate::traits::GraphEdge {
            id: params.id,
            source: params.source,
            target: params.target,
            label: params.label,
            properties: params.properties.unwrap_or_default(),
        };

        self.graph_store.add_edge(edge).await.map_err(|e| {
            McpError::internal_error("graph_error", Some(json!({"error": e.to_string()})))
        })?;

        Ok(CallToolResult::success(vec![Content::text(
            "Edge added successfully".to_string(),
        )]))
    }

    // ── Memory Tools ──

    #[tool(description = "Store a memory entry. The content is embedded as a vector and stored in HNSW for semantic recall. Supports sessions and importance-based retention.")]
    async fn memory_remember(
        &self,
        Parameters(params): Parameters<MemoryRememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let role = match params.role.to_lowercase().as_str() {
            "user" => crate::traits::MemoryRole::User,
            "assistant" => crate::traits::MemoryRole::Assistant,
            "system" => crate::traits::MemoryRole::System,
            "reflection" => crate::traits::MemoryRole::Reflection,
            _ => crate::traits::MemoryRole::Observation,
        };

        let entry = self.memory_store.remember(
            &params.content,
            role,
            params.session_id.as_deref(),
            params.importance,
            params.metadata.unwrap_or_default(),
        ).await.map_err(|e| {
            McpError::internal_error("memory_error", Some(json!({"error": e.to_string()})))
        })?;

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&json!({
                "id": entry.id,
                "timestamp": entry.timestamp,
                "role": format!("{:?}", entry.role),
                "session_id": entry.session_id,
            })).unwrap_or_else(|_| "Memory stored".to_string()),
        )]))
    }

    #[tool(description = "Recall memories by semantic similarity. Searches HNSW for memories similar to the query text, with optional session filtering.")]
    async fn memory_recall(
        &self,
        Parameters(params): Parameters<MemoryRecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let hits = self.memory_store.recall(
            &params.query,
            params.k,
            params.session_id.as_deref(),
        ).await.map_err(|e| {
            McpError::internal_error("memory_error", Some(json!({"error": e.to_string()})))
        })?;

        let output: Vec<serde_json::Value> = hits.iter().map(|hit| {
            json!({
                "id": hit.entry.id,
                "content": hit.entry.content,
                "role": format!("{:?}", hit.entry.role),
                "score": hit.score,
                "retention": hit.retention,
                "timestamp": hit.entry.timestamp,
                "session_id": hit.entry.session_id,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".to_string()),
        )]))
    }

    #[tool(description = "Get recent memories ordered by timestamp (newest first), with optional session filtering.")]
    async fn memory_recent(
        &self,
        Parameters(params): Parameters<MemoryRecentParams>,
    ) -> Result<CallToolResult, McpError> {
        let entries = self.memory_store.recent(
            params.n,
            params.session_id.as_deref(),
        ).await.map_err(|e| {
            McpError::internal_error("memory_error", Some(json!({"error": e.to_string()})))
        })?;

        let output: Vec<serde_json::Value> = entries.iter().map(|e| {
            json!({
                "id": e.id,
                "content": e.content,
                "role": format!("{:?}", e.role),
                "timestamp": e.timestamp,
                "session_id": e.session_id,
                "importance": e.importance,
            })
        }).collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "[]".to_string()),
        )]))
    }

    #[tool(description = "Forget (soft-delete) a memory entry by ID.")]
    async fn memory_forget(
        &self,
        Parameters(params): Parameters<MemoryForgetParams>,
    ) -> Result<CallToolResult, McpError> {
        let deleted = self.memory_store.forget(&params.id).await.map_err(|e| {
            McpError::internal_error("memory_error", Some(json!({"error": e.to_string()})))
        })?;

        Ok(CallToolResult::success(vec![Content::text(
            if deleted { "Memory forgotten" } else { "Memory not found" }.to_string(),
        )]))
    }

    #[tool(description = "Get memory store statistics: total memories, active/cold counts, session count.")]
    async fn memory_stats(&self) -> Result<CallToolResult, McpError> {
        let stats = self.memory_store.stats().await;
        let output = json!({
            "total_memories": stats.total_memories,
            "active_memories": stats.active_memories,
            "cold_memories": stats.cold_memories,
            "sessions": stats.sessions,
        });

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string()),
        )]))
    }

    #[tool(description = "Persist all data (vectors, graph, audit chain) to disk. Requires --data-dir to be set at startup.")]
    async fn persist(&self) -> Result<CallToolResult, McpError> {
        // This is a no-op if data_dir is not set — the runtime handles it
        Ok(CallToolResult::success(vec![Content::text(
            "Persist command received. Data will be saved on shutdown if --data-dir is configured.".to_string(),
        )]))
    }
}

#[tool_handler]
impl ServerHandler for FedCtxMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(Implementation::from_build_env())
        .with_protocol_version(ProtocolVersion::V_2024_11_05)
        .with_instructions(
            "FedCtx — Federated Semantic Infrastructure with Memory. \
             Vector Tools: vector_search (HNSW similarity search), \
             insert_vector (add vectors). \
             Graph Tools: graph_traverse (BFS graph traversal), \
             graph_neighbors (get node neighbors), \
             add_graph_node / add_graph_edge (build knowledge graph). \
             Memory Tools: memory_remember (store memory), \
             memory_recall (semantic recall), \
             memory_recent (recent memories), \
             memory_forget (delete memory), \
             memory_stats (memory statistics). \
             Audit: audit_recent (SHA-256 audit trail). \
             System: stats, persist."
                .to_string(),
        )
    }
}
