//! REST API (Axum)
//!
//! JSON/HTTP endpoints for health, stats, search, and audit.

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

#[derive(Clone)]
pub struct AppState {
    pub db: SharedVectorDB,
    pub audit: SharedAuditChain,
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

#[derive(Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub values: Vec<f32>,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Serialize)]
pub struct InsertResponse {
    pub success: bool,
    pub id: String,
}

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

fn internal_error(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: msg.into() }))
}

fn bad_request(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg.into() }))
}

// ── Handlers ──

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let audit = state.audit.read().unwrap();
    let chain_valid = audit.verify().is_ok();
    Json(HealthResponse {
        status: if chain_valid { "healthy".into() } else { "degraded".into() },
        version: env!("CARGO_PKG_VERSION").into(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        audit_chain_valid: chain_valid,
    })
}

async fn stats(State(state): State<AppState>) -> Result<Json<StatsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let db = state.db.read().map_err(|e| internal_error(&format!("db lock poisoned: {e}")))?;
    let audit = state.audit.read().map_err(|e| internal_error(&format!("audit lock poisoned: {e}")))?;
    let chain_valid = audit.verify().is_ok();
    Ok(Json(StatsResponse {
        total_vectors: db.len(),
        dimension: db.dimension(),
        audit_chain_length: audit.len(),
        audit_chain_valid: chain_valid,
    }))
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate
    if req.query.is_empty() {
        return Err(bad_request("query values must not be empty"));
    }
    if req.k == 0 {
        return Err(bad_request("k must be > 0"));
    }

    let t0 = std::time::Instant::now();
    let db = state.db.read().map_err(|e| internal_error(&format!("db lock poisoned: {e}")))?;
    let results = db.search(&req.query, req.k);
    let elapsed = t0.elapsed().as_millis();

    // Audit (best-effort, don't fail the request)
    if let Ok(mut audit) = state.audit.write() {
        audit.append(
            crate::audit::EventType::VectorSearch { k: req.k, result_count: results.len() },
            "rest",
            serde_json::json!({"source": "rest"}),
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
    match state.db.write().map_err(|e| internal_error(&format!("db lock poisoned: {e}")))?.insert(entry) {
        Ok(_) => {
            // Audit (best-effort)
            if let Ok(mut audit) = state.audit.write() {
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

async fn get_audit(State(state): State<AppState>) -> Result<Json<AuditResponse>, (StatusCode, Json<ErrorResponse>)> {
    let audit = state.audit.read().map_err(|e| internal_error(&format!("audit lock poisoned: {e}")))?;
    let chain_valid = audit.verify().is_ok();
    let entries: Vec<serde_json::Value> = audit.recent(50).iter().map(|b| {
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

// ── Router ──

pub fn create_router(db: SharedVectorDB, audit: SharedAuditChain) -> Router {
    let state = AppState {
        db,
        audit,
        start_time: std::time::Instant::now(),
    };

    Router::new()
        .route("/health", get(health))
        .route("/api/stats", get(stats))
        .route("/api/search", post(search))
        .route("/api/vectors", post(insert_vector))
        .route("/api/audit", get(get_audit))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
