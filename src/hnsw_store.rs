//! HNSW-backed VectorStore implementation
//!
//! Adapts the existing VectorDB to the VectorStore trait.
//! Uses parking_lot::RwLock via SharedVectorDB type alias.

use std::collections::HashMap;
use async_trait::async_trait;

use crate::traits::*;
use crate::vector_db::{SharedVectorDB, VectorDB, VectorEntry as DbEntry, SearchHit as DbHit, VectorDBError};

/// HNSW-backed VectorStore implementation.
pub struct HnswVectorStore {
    db: SharedVectorDB,
}

impl HnswVectorStore {
    pub fn new(dimension: usize) -> Self {
        Self {
            db: std::sync::Arc::new(parking_lot::RwLock::new(VectorDB::new(dimension))),
        }
    }

    pub fn from_shared(db: SharedVectorDB) -> Self {
        Self { db }
    }

    /// Get a reference to the underlying shared VectorDB.
    /// Useful for backward compatibility with existing gRPC services.
    pub fn shared(&self) -> SharedVectorDB {
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
        let mut db = self.db.write();
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
        let db = self.db.read();
        let results = db.search(query, k);
        let filtered = match _filter {
            Some(filter) => results
                .into_iter()
                .filter(|hit| {
                    filter.iter().all(|(k, v)| {
                        hit.metadata.get(k) == Some(v)
                    })
                })
                .collect(),
            None => results,
        };
        Ok(filtered.into_iter().map(convert_hit).collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<usize, VectorStoreError> {
        let mut db = self.db.write();
        Ok(db.delete(ids))
    }

    async fn stats(&self) -> VectorStoreStats {
        let db = self.db.read();
        VectorStoreStats {
            total_vectors: db.len(),
            dimension: db.dimension(),
            index_type: "hnsw".to_string(),
        }
    }
}
