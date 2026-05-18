//! Self-implemented HNSW Vector Database
//!
//! Wraps the `hnsw` crate with metadata, thread-safe access.
//! Based on organoid-fl's verified HNSW implementation.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ── Distance Metric ──

#[derive(Clone, Debug, Default)]
pub struct Cosine;

impl space::Metric<Vec<f32>> for Cosine {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        let similarity = if denom > 1e-10 { dot / denom } else { 0.0 };
        let dist = 1.0 - similarity;
        (dist * 1_000_000.0).clamp(0.0, u32::MAX as f32) as u32
    }
}

// ── Data Types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Error)]
pub enum VectorDBError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("empty vector values for id '{id}'")]
    EmptyValues { id: String },
}

// ── HNSW Index (verified from organoid-fl) ──

pub struct HnswIndex<const M: usize = 16, const M0: usize = 32> {
    #[allow(dead_code)]
    dimension: usize,
    id_to_string: HashMap<usize, String>,
    string_to_id: HashMap<String, usize>,
    graph: hnsw::Hnsw<Cosine, Vec<f32>, rand::rngs::StdRng, M, M0>,
    searcher: hnsw::Searcher<u32>,
    next_internal_id: usize,
}

impl<const M: usize, const M0: usize> HnswIndex<M, M0> {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            id_to_string: HashMap::new(),
            string_to_id: HashMap::new(),
            graph: hnsw::Hnsw::new(Cosine),
            searcher: hnsw::Searcher::new(),
            next_internal_id: 0,
        }
    }

    /// Insert a vector. Returns the internal ID.
    pub fn insert(&mut self, id: &str, values: &[f32]) -> usize {
        let internal_id = self.next_internal_id;
        self.next_internal_id += 1;
        self.id_to_string.insert(internal_id, id.to_string());
        self.string_to_id.insert(id.to_string(), internal_id);
        let _ = self.graph.insert(values.to_vec(), &mut self.searcher);
        internal_id
    }

    /// Search for k nearest neighbors. Returns (id, distance) sorted by distance.
    pub fn search(&mut self, query: &[f32], k: usize, ef: usize) -> Vec<(String, f32)> {
        if self.id_to_string.is_empty() {
            return Vec::new();
        }
        let n = self.id_to_string.len();
        let actual_k = k.min(n);
        let buf_size = ef.min(n);

        let mut neighbors = vec![space::Neighbor { index: 0, distance: u32::MAX }; buf_size];

        let results = self.graph.nearest(
            &query.to_vec(),
            buf_size,
            &mut self.searcher,
            &mut neighbors,
        );

        results[..actual_k.min(results.len())]
            .iter()
            .filter_map(|neighbor| {
                self.id_to_string.get(&neighbor.index).map(|id| {
                    let distance = f32::from_bits(neighbor.distance);
                    (id.clone(), distance)
                })
            })
            .collect()
    }

    /// Soft remove: removes from ID mapping only (HNSW graph keeps the node).
    pub fn remove(&mut self, id: &str) {
        if let Some(internal_id) = self.string_to_id.remove(id) {
            self.id_to_string.remove(&internal_id);
        }
    }

    pub fn len(&self) -> usize {
        self.string_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.string_to_id.is_empty()
    }

    pub fn layers(&self) -> usize {
        self.graph.layers()
    }
}

// ── VectorDB ──

pub struct VectorDB {
    dimension: usize,
    entries: HashMap<String, VectorEntry>,
    hnsw: Mutex<HnswIndex<16, 32>>,
}

impl VectorDB {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: HashMap::new(),
            hnsw: Mutex::new(HnswIndex::new(dimension)),
        }
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, entry: VectorEntry) -> Result<(), VectorDBError> {
        if entry.values.is_empty() {
            return Err(VectorDBError::EmptyValues { id: entry.id.clone() });
        }
        if entry.values.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: entry.values.len(),
            });
        }

        let id = entry.id.clone();
        let is_update = self.entries.contains_key(&id);

        // If updating, soft-remove old entry from HNSW first
        if is_update {
            self.hnsw.lock().unwrap().remove(&id);
        }

        self.hnsw.lock().unwrap().insert(&id, &entry.values);
        self.entries.insert(id, entry);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchHit> {
        if query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }
        let ef = k.max(50);
        let raw = self.hnsw.lock().unwrap().search(query, k, ef);
        raw.into_iter()
            .filter_map(|(id, distance)| {
                // Convert distance back to similarity score
                let score = 1.0 - (distance / 1_000_000.0);
                self.entries.get(&id).map(|e| SearchHit {
                    id: id.clone(),
                    score,
                    metadata: e.metadata.clone(),
                })
            })
            .collect()
    }

    /// Search with metadata filter (key=value pairs).
    /// Only returns hits where all filter keys match the entry's metadata.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&HashMap<String, String>>,
    ) -> Vec<SearchHit> {
        let mut results = self.search(query, k * 3); // Over-fetch for filtering
        if let Some(filter) = filter {
            results.retain(|hit| {
                filter.iter().all(|(k, v)| {
                    hit.metadata.get(k).map_or(false, |hv| hv == v)
                })
            });
            results.truncate(k);
        }
        results
    }

    pub fn delete(&mut self, ids: &[String]) -> usize {
        let mut count = 0;
        for id in ids {
            if self.entries.remove(id).is_some() {
                self.hnsw.lock().unwrap().remove(id);
                count += 1;
            }
        }
        count
    }

    pub fn get(&self, id: &str) -> Option<&VectorEntry> {
        self.entries.get(id)
    }

    pub fn all_entries(&self) -> Vec<&VectorEntry> {
        self.entries.values().collect()
    }

    pub fn hnsw_layers(&self) -> usize {
        self.hnsw.lock().unwrap().layers()
    }

    // ── Persistence ──

    /// Persist all entries to a binary file.
    ///
    /// Format: [u32 dimension] [u64 count] [entries...]
    /// Each entry: [u16 id_len] [id bytes] [f32 * dimension values] [u32 meta_json_len] [meta json bytes]
    pub fn persist(&self, path: &Path) -> Result<(), std::io::Error> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());

        for entry in self.entries.values() {
            // id
            let id_bytes = entry.id.as_bytes();
            buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(id_bytes);
            // values
            for v in &entry.values {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            // metadata as JSON
            let meta_json = serde_json::to_string(&entry.metadata).unwrap_or_default();
            let meta_bytes = meta_json.as_bytes();
            buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(meta_bytes);
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, buf)
    }

    /// Restore entries from a binary file, rebuilding the HNSW index.
    ///
    /// Returns the number of entries restored.
    pub fn restore(&mut self, path: &Path) -> Result<usize, std::io::Error> {
        if !path.exists() {
            return Ok(0);
        }

        let data = std::fs::read(path)?;
        if data.len() < 12 {
            return Ok(0); // Too small to contain header
        }

        let mut cursor = 0usize;

        // Read dimension
        let dim = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if dim != self.dimension {
            tracing::warn!(
                "Persisted dimension {} differs from current {}, skipping restore",
                dim, self.dimension
            );
            return Ok(0);
        }

        // Read count
        let count = u64::from_le_bytes(data[cursor..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;

        let mut restored = 0;
        for _ in 0..count {
            if cursor + 2 > data.len() { break; }
            // id
            let id_len = u16::from_le_bytes(data[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2;
            if cursor + id_len > data.len() { break; }
            let id = String::from_utf8_lossy(&data[cursor..cursor + id_len]).to_string();
            cursor += id_len;

            // values
            let values_len = dim * 4; // f32 = 4 bytes
            if cursor + values_len > data.len() { break; }
            let values: Vec<f32> = (0..dim)
                .map(|i| {
                    let start = cursor + i * 4;
                    f32::from_le_bytes(data[start..start + 4].try_into().unwrap())
                })
                .collect();
            cursor += values_len;

            // metadata
            if cursor + 4 > data.len() { break; }
            let meta_len = u32::from_le_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let metadata: HashMap<String, String> = if meta_len > 0 && cursor + meta_len <= data.len() {
                let meta_str = String::from_utf8_lossy(&data[cursor..cursor + meta_len]);
                cursor += meta_len;
                serde_json::from_str(&meta_str).unwrap_or_default()
            } else {
                cursor += meta_len;
                HashMap::new()
            };

            // Insert into entries + HNSW
            let entry = VectorEntry { id: id.clone(), values, metadata };
            self.hnsw.lock().unwrap().insert(&id, &entry.values);
            self.entries.insert(id, entry);
            restored += 1;
        }

        tracing::info!("Restored {} vectors from {:?}", restored, path);
        Ok(restored)
    }
}

pub type SharedVectorDB = Arc<RwLock<VectorDB>>;

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "v1".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        db.insert(VectorEntry {
            id: "v2".into(),
            values: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        let results = db.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].score > 0.9);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut db = VectorDB::new(3);
        let result = db.insert(VectorEntry {
            id: "bad".into(),
            values: vec![1.0, 2.0],
            metadata: HashMap::new(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_values_rejected() {
        let mut db = VectorDB::new(3);
        let result = db.insert(VectorEntry {
            id: "empty".into(),
            values: vec![],
            metadata: HashMap::new(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "v1".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        assert_eq!(db.len(), 1);
        let deleted = db.delete(&["v1".into()]);
        assert_eq!(deleted, 1);
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut db = VectorDB::new(3);
        let deleted = db.delete(&["nonexistent".into()]);
        assert_eq!(deleted, 0);
    }

    #[test]
    fn test_upsert_same_id() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "v1".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        // Insert again with same ID — should replace
        db.insert(VectorEntry {
            id: "v1".into(),
            values: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        assert_eq!(db.len(), 1);
        let results = db.search(&[0.0, 1.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_search_empty_db() {
        let db = VectorDB::new(3);
        let results = db.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_query() {
        let mut db = VectorDB::new(3);
        db.insert(VectorEntry {
            id: "v1".into(),
            values: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        }).unwrap();
        let results = db.search(&[], 5);
        assert!(results.is_empty());
    }
}
