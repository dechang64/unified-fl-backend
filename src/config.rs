//! Configuration System
//!
//! Supports layered configuration: TOML file → environment variables → CLI args.
//! Priority: CLI > env vars > config file > defaults.

use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::info;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FedCtxConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub vector: VectorConfig,
    #[serde(default)]
    pub graph: GraphConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub fl: FlConfig,
    #[serde(default)]
    pub graphrag: GraphRAGConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,
    #[serde(default = "default_http_port")]
    pub http_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    #[serde(default = "default_dimension")]
    pub dimension: usize,
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: usize,
    #[serde(default = "default_hnsw_ef_search")]
    pub hnsw_ef_search: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    #[serde(default = "default_graph_backend")]
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default = "default_stability_days")]
    pub stability_days: f64,
    #[serde(default = "default_importance_boost")]
    pub importance_boost: f32,
    #[serde(default = "default_access_boost_days")]
    pub access_boost_days: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    #[serde(default)]
    pub data_dir: Option<String>,
    #[serde(default = "default_auto_persist_secs")]
    pub auto_persist_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlConfig {
    #[serde(default = "default_fl_strategy")]
    pub strategy: String,
    #[serde(default = "default_min_clients")]
    pub min_clients: usize,
    #[serde(default = "default_fedprox_mu")]
    pub fedprox_mu: f64,
    #[serde(default = "default_ewa_alpha")]
    pub ewa_alpha: f64,
    #[serde(default)]
    pub dp_epsilon: Option<f64>,
    #[serde(default)]
    pub dp_delta: Option<f64>,
    #[serde(default = "default_dp_clip_norm")]
    pub dp_clip_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    #[serde(default = "default_grag_top_k")]
    pub top_k: usize,
    #[serde(default = "default_grag_max_hops")]
    pub max_hops: usize,
    #[serde(default = "default_grag_max_nodes")]
    pub max_nodes: usize,
    #[serde(default = "default_grag_vector_weight")]
    pub vector_weight: f32,
    #[serde(default = "default_grag_centrality_weight")]
    pub centrality_weight: f32,
    #[serde(default = "default_grag_min_score")]
    pub min_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_embedding_provider")]
    pub provider: String,
    #[serde(default)]
    pub model_dir: Option<String>,
}

// ── Default functions ──

fn default_grpc_port() -> u16 { 50051 }
fn default_http_port() -> u16 { 8080 }
fn default_dimension() -> usize { 384 }
fn default_hnsw_m() -> usize { 16 }
fn default_hnsw_ef_search() -> usize { 50 }
fn default_graph_backend() -> String { "json".to_string() }
fn default_stability_days() -> f64 { 7.0 }
fn default_importance_boost() -> f32 { 0.3 }
fn default_access_boost_days() -> f64 { 2.0 }
fn default_auto_persist_secs() -> u64 { 300 }
fn default_fl_strategy() -> String { "fedavg".to_string() }
fn default_min_clients() -> usize { 2 }
fn default_fedprox_mu() -> f64 { 0.01 }
fn default_ewa_alpha() -> f64 { 2.0 }
fn default_dp_clip_norm() -> f64 { 1.0 }
fn default_grag_top_k() -> usize { 10 }
fn default_grag_max_hops() -> usize { 2 }
fn default_grag_max_nodes() -> usize { 30 }
fn default_grag_vector_weight() -> f32 { 0.6 }
fn default_grag_centrality_weight() -> f32 { 0.4 }
fn default_grag_min_score() -> f32 { 0.1 }
fn default_embedding_provider() -> String { "hash".to_string() }

impl Default for ServerConfig {
    fn default() -> Self {
        Self { grpc_port: default_grpc_port(), http_port: default_http_port() }
    }
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self { dimension: default_dimension(), hnsw_m: default_hnsw_m(), hnsw_ef_search: default_hnsw_ef_search() }
    }
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { backend: default_graph_backend() }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            stability_days: default_stability_days(),
            importance_boost: default_importance_boost(),
            access_boost_days: default_access_boost_days(),
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self { data_dir: None, auto_persist_secs: default_auto_persist_secs() }
    }
}

impl Default for FlConfig {
    fn default() -> Self {
        Self {
            strategy: default_fl_strategy(),
            min_clients: default_min_clients(),
            fedprox_mu: default_fedprox_mu(),
            ewa_alpha: default_ewa_alpha(),
            dp_epsilon: None,
            dp_delta: None,
            dp_clip_norm: default_dp_clip_norm(),
        }
    }
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            top_k: default_grag_top_k(),
            max_hops: default_grag_max_hops(),
            max_nodes: default_grag_max_nodes(),
            vector_weight: default_grag_vector_weight(),
            centrality_weight: default_grag_centrality_weight(),
            min_score: default_grag_min_score(),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self { provider: default_embedding_provider(), model_dir: None }
    }
}

impl FedCtxConfig {
    /// Load configuration from a TOML file.
    pub fn load_from_file(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse config file: {}", e))?;
        info!("Loaded config from {:?}", path);
        Ok(config)
    }

    /// Load configuration from a TOML string.
    pub fn load_from_str(toml_str: &str) -> Result<Self, String> {
        toml::from_str(toml_str)
            .map_err(|e| format!("Failed to parse config: {}", e))
    }

    /// Generate a default config file as a TOML string.
    pub fn to_toml_string(&self) -> Result<String, String> {
        toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))
    }

    /// Generate a template config file.
    pub fn template() -> String {
        let config = Self::default();
        config.to_toml_string().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FedCtxConfig::default();
        assert_eq!(config.server.grpc_port, 50051);
        assert_eq!(config.vector.dimension, 384);
        assert_eq!(config.fl.strategy, "fedavg");
        assert_eq!(config.graphrag.top_k, 10);
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
[server]
grpc_port = 50052
http_port = 8081

[vector]
dimension = 768

[fl]
strategy = "ewa"
min_clients = 3

[graphrag]
top_k = 20
max_hops = 3
"#;
        let config = FedCtxConfig::load_from_str(toml).unwrap();
        assert_eq!(config.server.grpc_port, 50052);
        assert_eq!(config.vector.dimension, 768);
        assert_eq!(config.fl.strategy, "ewa");
        assert_eq!(config.graphrag.top_k, 20);
    }

    #[test]
    fn test_roundtrip() {
        let config = FedCtxConfig::default();
        let toml = config.to_toml_string().unwrap();
        let parsed = FedCtxConfig::load_from_str(&toml).unwrap();
        assert_eq!(parsed.server.grpc_port, config.server.grpc_port);
        assert_eq!(parsed.vector.dimension, config.vector.dimension);
        assert_eq!(parsed.fl.strategy, config.fl.strategy);
    }

    #[test]
    fn test_template() {
        let template = FedCtxConfig::template();
        assert!(template.contains("grpc_port"));
        assert!(template.contains("dimension"));
        assert!(template.contains("strategy"));
    }
}
