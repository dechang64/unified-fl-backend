//! Embedding Provider — pluggable text-to-vector encoding
//!
//! Provides the `EmbeddingProvider` trait with two implementations:
//! - `HashEmbedder`: deterministic hash-based embedding (zero dependency, no semantics)
//! - `OnnxEmbedder`: ONNX Runtime + sentence-transformers (semantic, requires model file)
//!
//! All stores that need text embedding should depend on this trait,
//! not on a specific implementation.

use std::sync::Arc;
use async_trait::async_trait;
use sha2::{Sha256, Digest};

/// Error type for embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("ONNX Runtime error: {0}")]
    OnnxRuntime(String),
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty input text")]
    EmptyInput,
    #[error("internal error: {0}")]
    Internal(String),
}

/// Pluggable embedding provider trait.
///
/// Implementations convert text → fixed-dimension float vectors.
/// The trait is `Send + Sync` so it can be shared across async tasks.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a single text string into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed multiple texts in batch (default: sequential, override for GPU batching).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// The dimension of the output vectors.
    fn dimension(&self) -> usize;

    /// Human-readable name of this provider.
    fn name(&self) -> &str;
}

// ════════════════════════════════════════════════════════════════
// Hash Embedder — deterministic, no semantics, zero dependency
// ════════════════════════════════════════════════════════════════

/// Hash-based embedding provider.
///
/// Uses multi-round SHA-256 hashing to produce a fixed-dimension vector.
/// **Warning**: This produces semantically meaningless vectors —
/// "antibody affinity" and "antibody binding" will NOT be close.
/// Use `OnnxEmbedder` for production semantic search.
pub struct HashEmbedder {
    dimension: usize,
}

impl HashEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for HashEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            return Err(EmbedError::EmptyInput);
        }
        Ok(hash_embed(text, self.dimension))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "hash"
    }
}

/// Deterministic hash-based embedding.
/// Produces a normalized vector from text via multi-round SHA-256.
fn hash_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(dim);
    let mut hasher = Sha256::new();

    // First round: hash the text itself
    hasher.update(text.as_bytes());
    let mut hash = hasher.finalize();

    for i in 0..dim {
        // Each dimension gets its own hash round
        let mut round_hasher = Sha256::new();
        round_hasher.update(hash);
        round_hasher.update((i as u32).to_le_bytes());
        hash = round_hasher.finalize();

        // Take first 4 bytes as f32 bits, normalize to [-1, 1]
        let bytes: [u8; 4] = hash[..4].try_into().unwrap_or([0; 4]);
        let raw = u32::from_be_bytes(bytes);
        let normalized = (raw as f64 / u32::MAX as f64) * 2.0 - 1.0;
        result.push(normalized as f32);
    }

    // L2 normalize
    let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut result {
            *v /= norm;
        }
    }

    result
}

// ════════════════════════════════════════════════════════════════
// ONNX Embedder — semantic, requires model file
// ════════════════════════════════════════════════════════════════

/// ONNX Runtime-based embedding provider.
///
/// Uses sentence-transformers models (e.g., `all-MiniLM-L6-v2`)
/// exported to ONNX format for high-performance CPU inference.
///
/// **Setup**:
/// 1. Download model: `huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/minilm`
/// 2. Export to ONNX: `optimum-cli export onnx --model ./models/minilm ./models/minilm_onnx`
/// 3. Point `model_dir` to the ONNX directory
///
/// If ONNX Runtime is not available at compile time, this struct
/// will still compile but `new()` will return an error at runtime.
pub struct OnnxEmbedder {
    dimension: usize,
    // NOTE: ONNX Runtime types are not Send+Sync in all configurations.
    // We wrap the actual session in a Mutex for thread safety.
    // When `ort` crate is added as a dependency, uncomment the field below.
    // session: Mutex<ort::Session>,
    _placeholder: (),
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder from a model directory.
    ///
    /// The directory should contain `model.onnx` and `tokenizer.json`.
    /// Returns an error if ONNX Runtime is not available or model is not found.
    pub fn new(_model_dir: &str, _dimension: usize) -> Result<Self, EmbedError> {
        // TODO: Uncomment when `ort` crate is added as dependency
        // let session = ort::Session::builder()
        //     .map_err(|e| EmbedError::OnnxRuntime(e.to_string()))?
        //     .commit_from_file(format!("{model_dir}/model.onnx"))
        //     .map_err(|e| EmbedError::ModelNotFound(e.to_string()))?;
        // Ok(Self { dimension, session: Mutex::new(session) })

        // Placeholder: ONNX Runtime not yet linked
        Err(EmbedError::OnnxRuntime(
            "ONNX Runtime not linked. Add `ort = { version = \"2\", features = [\"load-dynamic\"] }` to Cargo.toml and uncomment OnnxEmbedder implementation.".to_string()
        ))
    }
}

#[async_trait]
impl EmbeddingProvider for OnnxEmbedder {
    async fn embed(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        // TODO: Implement with ort crate
        // 1. Tokenize text using tokenizer.json
        // 2. Run ONNX inference
        // 3. Mean-pool token embeddings → single vector
        // 4. L2 normalize
        Err(EmbedError::OnnxRuntime("OnnxEmbedder not yet implemented".to_string()))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "onnx"
    }
}

/// Create the default embedding provider based on available dependencies.
///
/// Returns `OnnxEmbedder` if a model path is provided and ONNX RT is available,
/// otherwise falls back to `HashEmbedder`.
pub fn default_embedding_provider(dimension: usize, model_dir: Option<&str>) -> Arc<dyn EmbeddingProvider> {
    match model_dir {
        Some(dir) => {
            match OnnxEmbedder::new(dir, dimension) {
                Ok(embedder) => Arc::new(embedder),
                Err(e) => {
                    tracing::warn!("Failed to initialize ONNX embedder ({}), falling back to hash embedder. Semantic search will be degraded.", e);
                    Arc::new(HashEmbedder::new(dimension))
                }
            }
        }
        None => {
            tracing::info!("No embedding model specified, using hash embedder. Set --embedding-model for semantic search.");
            Arc::new(HashEmbedder::new(dimension))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hash_embedder_basic() {
        let embedder = HashEmbedder::new(384);
        let vec = embedder.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 384);

        // L2 norm should be ~1.0
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_hash_embedder_deterministic() {
        let embedder = HashEmbedder::new(128);
        let v1 = embedder.embed("test input").await.unwrap();
        let v2 = embedder.embed("test input").await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn test_hash_embedder_different_inputs() {
        let embedder = HashEmbedder::new(128);
        let v1 = embedder.embed("input A").await.unwrap();
        let v2 = embedder.embed("input B").await.unwrap();
        // Different inputs should produce different vectors
        assert_ne!(v1, v2);
    }

    #[tokio::test]
    async fn test_hash_embedder_empty_input() {
        let embedder = HashEmbedder::new(128);
        assert!(embedder.embed("").await.is_err());
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let embedder = HashEmbedder::new(64);
        let texts = vec!["hello", "world", "test"];
        let results = embedder.embed_batch(&texts).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 64);
    }

    #[test]
    fn test_default_provider_no_model() {
        let provider = default_embedding_provider(384, None);
        assert_eq!(provider.name(), "hash");
    }
}
