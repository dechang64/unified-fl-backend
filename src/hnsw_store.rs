//! HNSW-backed VectorStore implementation
//!
//! Adapts the existing VectorDB to the VectorStore trait.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;

use crate::traits::*;
use crate::vector_db::{VectorDB, VectorEntry as DbEntry, SearchHit as DbHit, VectorDBError};

/// HNSW-backed VectorStore implementation.
pub struct HnswVectorStore {
    db: Arc<RwLock<VectorDB>>,
}

impl HnswVectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            db: Arc::new(RwLock::new(VectorDB::new(dimension))),
        }
    }

    pub fn from_shared(db: Arc<RwLock<VectorDB>>) -> Self {
        Self { db }
    }

    /// Get a reference to the underlying shared VectorDB.
    /// Useful for backward compatibility with existing gRPC services.
    pub fn shared(&self) -> Arc<RwLock<VectorDB>> {
        self.db.clone()
    }
}

fn convert_entry(entry: VectorEntry) -> DbEntry {
    DbEntry {
        id: entry.id,
        values: entry.values,
        metadata: entry.metadata,
    }
}

fn convert_hit(hit: DbHit) -> SearchHit {
    SearchHit {
        id: hit.id,
        score: hit.score,
        metadata: hit.metadata,
    }
}

#[async_trait]
impl VectorStore for HnswVectorStore {
    async fn insert(&self, entries: Vec<VectorEntry>) -> Result<usize, VectorStoreError> {
        let mut db = self.db.write().map_err(|e| {
            VectorStoreError::Internal(format!("db lock poisoned: {e}"))
        })?;
        let mut count = 0;
        for entry in entries {
            match db.insert(convert_entry(entry)) {
                Ok(()) => count += 1,
                Err(VectorDBError::DimensionMismatch { expected, actual }) => {
                    return Err(VectorStoreError::DimensionMismatch { expected, actual });
                }
                Err(VectorDBError::EmptyValues { id }) => {
                    return Err(VectorStoreError::EmptyValues { id });
                }
            }
        }
        Ok(count)
    }

    async fn search(
        &self,
        query: &[f32],
        k: usize,
        _filter: Option<HashMap<String, String>>,
    ) -> Result<Vec<SearchHit>, VectorStoreError> {
        let db = self.db.read().map_err(|e| {
            VectorStoreError::Internal(format!("db lock poisoned: {e}"))
        })?;
        let results = db.search(query, k);
        // Apply metadata filter if provided
        let filtered = match _filter {
            Some(filter) => results
                .into_iter()
                .filter(|hit| {
                    filter.iter().all(|(k, v)| {
                        hit.metadata.get(k).map_or(false, |hv| hv == v)
                    })
                })
                .collect(),
            None => results,
        };
        Ok(filtered.into_iter().map(convert_hit).collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<usize, VectorStoreError> {
        let mut db = self.db.write().map_err(|e| {
            VectorStoreError::Internal(format!("db lock poisoned: {e}"))
        })?;
        Ok(db.delete(ids))
    }

    async fn stats(&self) -> VectorStoreStats {
        let db = self.db.read().unwrap();
        VectorStoreStats {
            total_vectors: db.len(),
            dimension: db.dimension(),
            index_type: "hnsw".to_string(),
        }
    }
}
