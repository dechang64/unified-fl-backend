//! REST API (Axum)
//!
//! JSON/HTTP endpoints for health, stats, search, audit, graph, and memory.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;

use crate::vector_db::SharedVectorDB;
use crate::audit::SharedAuditChain;
use crate::json_graph_store::JsonGraphStore;
use crate::memory_store::HnswMemoryStore;
use crate::embedding::EmbeddingProvider;
use crate::traits::{GraphStore, MemoryStore};

use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub db: SharedVectorDB,
    pub audit: SharedAuditChain,
    pub graph_store: Option<Arc<JsonGraphStore>>,
    pub memory_store: Option<Arc<HnswMemoryStore>>,
    pub embedding: Option<Arc<dyn EmbeddingProvider>>,
    pub start_time: std::time::Instant,
}

// ── Response Types ──

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub audit_chain_valid: bool,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_vectors: usize,
    pub dimension: usize,
    pub audit_chain_length: usize,
    pub audit_chain_valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_stats: Option<GraphStatsResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_stats: Option<MemoryStatsResponse>,
}

#[derive(Serialize)]
pub struct GraphStatsResponse {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_labels: Vec<String>,
    pub edge_labels: Vec<String>,
}

#[derive(Serialize)]
pub struct MemoryStatsResponse {
    pub total_memories: usize,
    pub active_memories: usize,
    pub cold_memories: usize,
    pub sessions: usize,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<crate::vector_db::SearchHit>,
    pub query_time_ms: u128,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
}

fn default_k() -> usize { 5 }
const MAX_K: usize = 100;

#[derive(Deserialize)]
pub struct TextSearchRequest {
    pub text: String,
    #[serde(default = "default_k")]
    pub k: usize,
}

#[derive(Serialize)]
pub struct InsertResponse {
    pub success: bool,
    pub id: String,
}

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct AuditQueryParams {
    /// Number of recent entries to return (default 50, max 500).
    #[serde(default = "default_audit_limit")]
    pub limit: usize,
}

fn default_audit_limit() -> usize { 50 }
const MAX_AUDIT_LIMIT: usize = 500;

#[derive(Serialize)]
pub struct AuditResponse {
    pub entries: Vec<serde_json::Value>,
    pub chain_valid: bool,
    pub total: usize,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[allow(dead_code)]
fn internal_error(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: msg.into() }))
}

fn bad_request(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg.into() }))
}

fn not_found(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::NOT_FOUND, Json(ErrorResponse { error: msg.into() }))
}

// ── Handlers ──

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let audit_len = state.audit.read().len();
    let chain_likely_valid = audit_len > 0;
    Json(HealthResponse {
        status: if chain_likely_valid { "healthy".into() } else { "ok".into() },
        version: env!("CARGO_PKG_VERSION").into(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        audit_chain_valid: chain_likely_valid,
    })
}

async fn stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read();
    let audit = state.audit.read();
    let chain_valid = audit.verify().is_ok();

    let graph_stats = if let Some(ref gs) = state.graph_store {
        let s = gs.stats_blocking();
        Some(GraphStatsResponse {
            total_nodes: s.total_nodes,
            total_edges: s.total_edges,
            node_labels: s.node_labels,
            edge_labels: s.edge_labels,
        })
    } else {
        None
    };

    let memory_stats = if let Some(ref ms) = state.memory_store {
        let s = ms.stats_blocking();
        Some(MemoryStatsResponse {
            total_memories: s.total_memories,
            active_memories: s.active_memories,
            cold_memories: s.cold_memories,
            sessions: s.sessions,
        })
    } else {
        None
    };

    Ok(Json(StatsResponse {
        total_vectors: db.len(),
        dimension: db.dimension(),
        audit_chain_length: audit.len(),
        audit_chain_valid: chain_valid,
        graph_stats,
        memory_stats,
    }))
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.query.is_empty() {
        return Err(bad_request("query values must not be empty"));
    }
    if req.k == 0 {
        return Err(bad_request("k must be > 0"));
    }
    if req.k > MAX_K {
        return Err(bad_request(&format!("k must be <= {}", MAX_K)));
    }

    let t0 = std::time::Instant::now();
    let db = state.db.read();
    let results = db.search(&req.query, req.k);
    let elapsed = t0.elapsed().as_millis();

    {
        let mut audit = state.audit.write();
        audit.append(
            crate::audit::EventType::VectorSearch { k: req.k, result_count: results.len() },
            "rest",
            serde_json::json!({"source": "rest"}),
        );
    }

    Ok(Json(SearchResponse { results, query_time_ms: elapsed }))
}

async fn text_search(
    State(state): State<AppState>,
    Json(req): Json<TextSearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    let embedding = state.embedding.as_ref()
        .ok_or_else(|| bad_request("Embedding provider not configured. Set --embedding-model-dir for text search."))?;

    if req.text.is_empty() {
        return Err(bad_request("text must not be empty"));
    }
    if req.k == 0 {
        return Err(bad_request("k must be > 0"));
    }
    if req.k > MAX_K {
        return Err(bad_request(&format!("k must be <= {}", MAX_K)));
    }

    let query_vec = embedding.embed(&req.text).await
        .map_err(|e| bad_request(&format!("Embedding failed: {}", e)))?;

    let t0 = std::time::Instant::now();
    let db = state.db.read();
    let results = db.search(&query_vec, req.k);
    let elapsed = t0.elapsed().as_millis();

    {
        let mut audit = state.audit.write();
        audit.append(
            crate::audit::EventType::VectorSearch { k: req.k, result_count: results.len() },
            "rest",
            serde_json::json!({"source": "rest_text_search", "query_len": req.text.len()}),
        );
    }

    Ok(Json(SearchResponse { results, query_time_ms: elapsed }))
}

async fn insert_vector(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, (StatusCode, Json<ErrorResponse>)> {
    let entry = crate::vector_db::VectorEntry {
        id: req.id.clone(),
        values: req.values,
        metadata: req.metadata,
    };
    match state.db.write().insert(entry) {
        Ok(_) => {
            {
                let mut audit = state.audit.write();
                audit.append(
                    crate::audit::EventType::VectorInsert { count: 1 },
                    "rest",
                    serde_json::json!({"source": "rest", "id": req.id}),
                );
            }
            Ok(Json(InsertResponse { success: true, id: req.id }))
        }
        Err(e) => {
            tracing::error!("Insert failed: {}", e);
            Err(bad_request(&e.to_string()))
        }
    }
}

async fn get_audit(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<AuditQueryParams>,
) -> Result<Json<AuditResponse>, (StatusCode, Json<ErrorResponse>)> {
    let limit = params.limit.clamp(1, MAX_AUDIT_LIMIT);
    let audit = state.audit.read();
    let chain_valid = audit.verify().is_ok();
    let entries: Vec<serde_json::Value> = audit.recent(limit).iter().map(|b| {
        let hash_display = if b.hash.len() >= 16 { &b.hash[..16] } else { &b.hash };
        serde_json::json!({
            "sequence": b.sequence,
            "timestamp": b.timestamp,
            "event_type": b.event_type,
            "node_id": b.node_id,
            "metadata": b.metadata,
            "hash": hash_display,
        })
    }).collect();
    let total = audit.len();
    Ok(Json(AuditResponse { entries, chain_valid, total }))
}

// ── Graph Handlers ──

#[derive(Deserialize)]
pub struct AddNodeRequest {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub properties: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct AddEdgeRequest {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: String,
    #[serde(default)]
    pub properties: std::collections::HashMap<String, String>,
}

#[derive(Deserialize)]
pub struct TraverseQueryParams {
    pub start_id: String,
    pub edge_label: Option<String>,
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
}

fn default_max_hops() -> usize { 3 }

#[derive(Deserialize)]
pub struct NeighborsQueryParams {
    pub node_id: String,
    pub edge_label: Option<String>,
}

async fn graph_add_node(
    State(state): State<AppState>,
    Json(req): Json<AddNodeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let gs = state.graph_store.as_ref()
        .ok_or_else(|| not_found("Graph store not available"))?;
    let node = crate::traits::GraphNode {
        id: req.id,
        label: req.label,
        properties: req.properties,
    };
    gs.add_node(node).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(serde_json::json!({"success": true})))
}

async fn graph_add_edge(
    State(state): State<AppState>,
    Json(req): Json<AddEdgeRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let gs = state.graph_store.as_ref()
        .ok_or_else(|| not_found("Graph store not available"))?;
    let edge = crate::traits::GraphEdge {
        id: req.id,
        source: req.source,
        target: req.target,
        label: req.label,
        properties: req.properties,
    };
    gs.add_edge(edge).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(serde_json::json!({"success": true})))
}

async fn graph_traverse(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<TraverseQueryParams>,
) -> Result<Json<crate::traits::GraphPath>, (StatusCode, Json<ErrorResponse>)> {
    let gs = state.graph_store.as_ref()
        .ok_or_else(|| not_found("Graph store not available"))?;
    let path = gs.traverse(&params.start_id, params.edge_label.as_deref(), params.max_hops).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(path))
}

async fn graph_neighbors(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<NeighborsQueryParams>,
) -> Result<Json<Vec<crate::traits::GraphNode>>, (StatusCode, Json<ErrorResponse>)> {
    let gs = state.graph_store.as_ref()
        .ok_or_else(|| not_found("Graph store not available"))?;
    let neighbors = gs.neighbors(&params.node_id, params.edge_label.as_deref()).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(neighbors))
}

// ── Memory Handlers ──

#[derive(Deserialize)]
pub struct RememberRequest {
    pub content: String,
    pub role: Option<String>,
    pub session_id: Option<String>,
    #[serde(default = "default_importance")]
    pub importance: f32,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

fn default_importance() -> f32 { 0.5 }

#[derive(Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct RecentMemoriesQueryParams {
    #[serde(default = "default_audit_limit")]
    pub limit: usize,
    pub session_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ForgetRequest {
    pub id: String,
}

async fn memory_remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<crate::traits::MemoryEntry>, (StatusCode, Json<ErrorResponse>)> {
    let ms = state.memory_store.as_ref()
        .ok_or_else(|| not_found("Memory store not available"))?;
    let role = match req.role.as_deref() {
        Some("user") => crate::traits::MemoryRole::User,
        Some("assistant") => crate::traits::MemoryRole::Assistant,
        Some("system") => crate::traits::MemoryRole::System,
        Some("observation") => crate::traits::MemoryRole::Observation,
        Some("reflection") => crate::traits::MemoryRole::Reflection,
        _ => crate::traits::MemoryRole::Observation,
    };
    let entry = ms.remember(&req.content, role, req.session_id.as_deref(), req.importance, req.metadata).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(entry))
}

async fn memory_recall(
    State(state): State<AppState>,
    Json(req): Json<RecallRequest>,
) -> Result<Json<Vec<crate::traits::MemoryHit>>, (StatusCode, Json<ErrorResponse>)> {
    let ms = state.memory_store.as_ref()
        .ok_or_else(|| not_found("Memory store not available"))?;
    if req.k > MAX_K {
        return Err(bad_request(&format!("k must be <= {}", MAX_K)));
    }
    let hits = ms.recall(&req.query, req.k, req.session_id.as_deref()).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(hits))
}

async fn memory_recent(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<RecentMemoriesQueryParams>,
) -> Result<Json<Vec<crate::traits::MemoryEntry>>, (StatusCode, Json<ErrorResponse>)> {
    let ms = state.memory_store.as_ref()
        .ok_or_else(|| not_found("Memory store not available"))?;
    let limit = params.limit.clamp(1, MAX_AUDIT_LIMIT);
    let entries = ms.recent(limit, params.session_id.as_deref()).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(entries))
}

async fn memory_forget(
    State(state): State<AppState>,
    Json(req): Json<ForgetRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let ms = state.memory_store.as_ref()
        .ok_or_else(|| not_found("Memory store not available"))?;
    let deleted = ms.forget(&req.id).await
        .map_err(|e| bad_request(&e.to_string()))?;
    Ok(Json(serde_json::json!({"success": deleted})))
}

async fn memory_stats(
    State(state): State<AppState>,
) -> Result<Json<crate::traits::MemoryStoreStats>, (StatusCode, Json<ErrorResponse>)> {
    let ms = state.memory_store.as_ref()
        .ok_or_else(|| not_found("Memory store not available"))?;
    let stats = ms.stats().await;
    Ok(Json(stats))
}

// ── Router ──

pub fn create_router(
    db: SharedVectorDB,
    audit: SharedAuditChain,
    graph_store: Option<Arc<JsonGraphStore>>,
    memory_store: Option<Arc<HnswMemoryStore>>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
) -> Router {
    let state = AppState {
        db,
        audit,
        graph_store,
        memory_store,
        embedding,
        start_time: std::time::Instant::now(),
    };

    Router::new()
        .route("/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/search", post(search))
        .route("/api/text_search", post(text_search))
        .route("/api/vectors", post(insert_vector))
        .route("/api/audit", get(get_audit))
        // Graph endpoints
        .route("/api/graph/nodes", post(graph_add_node))
        .route("/api/graph/edges", post(graph_add_edge))
        .route("/api/graph/traverse", get(graph_traverse))
        .route("/api/graph/neighbors", get(graph_neighbors))
        // Memory endpoints
        .route("/api/memory/remember", post(memory_remember))
        .route("/api/memory/recall", post(memory_recall))
        .route("/api/memory/recent", get(memory_recent))
        .route("/api/memory/forget", post(memory_forget))
        .route("/api/memory/stats", get(memory_stats))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
