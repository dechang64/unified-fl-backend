//! HNSW-backed Memory Store
//!
//! Memory entries are stored as vectors in the same HNSW index as
//! business data, differentiated by metadata.type = "memory".
//! This enables cross-domain recall: "what did I discover about X?"

use std::collections::HashMap;
use parking_lot::RwLock;
use async_trait::async_trait;
use chrono::Utc;

use crate::traits::*;
use crate::vector_db::{SharedVectorDB, VectorEntry as DbEntry};

/// Ebbinghaus forgetting curve parameters.
/// R = e^(-t/S) where t = time since last access, S = stability.
const DEFAULT_STABILITY: f64 = 86400.0 * 7.0; // 7 days in seconds
const IMPORTANCE_BOOST: f32 = 0.3; // Each importance point adds 30% stability
const ACCESS_BOOST: f64 = 86400.0 * 2.0; // Each access adds 2 days to stability

/// In-memory index for session-based timeline queries.
/// Maps session_id → ordered list of memory IDs.
struct SessionTimeline {
    sessions: HashMap<String, Vec<String>>,
    /// All memory entries indexed by ID (mirrors HNSW metadata).
    entries: HashMap<String, MemoryEntry>,
}

/// HNSW-backed memory store implementation.
pub struct HnswMemoryStore {
    vector_db: SharedVectorDB,
    timeline: RwLock<SessionTimeline>,
    dimension: usize,
}

impl HnswMemoryStore {
    pub fn new(vector_db: SharedVectorDB, dimension: usize) -> Self {
        Self {
            vector_db,
            timeline: RwLock::new(SessionTimeline {
                sessions: HashMap::new(),
                entries: HashMap::new(),
            }),
            dimension,
        }
    }

    /// Synchronous stats access (for non-async contexts like load_from_disk).
    pub fn stats_blocking(&self) -> MemoryStoreStats {
        let timeline = self.timeline.read();
        let total = timeline.entries.len();
        let active = timeline.entries.values()
            .filter(|e| Self::retention_score(e) > 0.5)
            .count();
        let cold = total - active;
        let sessions = timeline.sessions.len();

        MemoryStoreStats {
            total_memories: total,
            active_memories: active,
            cold_memories: cold,
            sessions,
        }
    }

    /// Rebuild the session timeline from HNSW vector metadata.
    /// Called after restoring vectors from disk.
    pub fn rebuild_timeline(&self) {
        let db = self.vector_db.read();
        let mut timeline = self.timeline.write();
        timeline.sessions.clear();
        timeline.entries.clear();

        for entry in db.all_entries() {
            if entry.metadata.get("type").is_some_and(|t| t == "memory") {
                if let Some(mem_entry) = Self::entry_from_metadata(entry) {
                    let id = mem_entry.id.clone();
                    if let Some(ref sid) = mem_entry.session_id {
                        timeline.sessions.entry(sid.clone()).or_default().push(id.clone());
                    }
                    timeline.entries.insert(id, mem_entry);
                }
            }
        }
    }

    /// Generate a deterministic hash-based embedding for text.
    /// This is the zero-dependency fallback (same as claw-med-gps).
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut result = vec![0.0f32; self.dimension];
        // Use multiple hash rounds to fill the dimension
        for i in 0..self.dimension {
            let input = format!("{}:round{}", text, i / 8);
            let hash = {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(input.as_bytes());
                let hash_bytes = hasher.finalize();
                // Take 4 bytes as f32 bits
                let bits = u32::from_le_bytes(hash_bytes[i % 8 * 4..i % 8 * 4 + 4].try_into().unwrap_or([0; 4]));
                f32::from_bits(bits)
            };
            result[i] = hash;
        }
        // L2 normalize
        let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in result.iter_mut() {
                *x /= norm;
            }
        }
        result
    }

    /// Compute retention score using Ebbinghaus forgetting curve.
    /// R = e^(-t/S) where t = seconds since last access, S = stability.
    fn retention_score(entry: &MemoryEntry) -> f32 {
        let last_accessed = chrono::DateTime::parse_from_rfc3339(&entry.timestamp)
            .map(|dt| dt.timestamp() as f64)
            .unwrap_or_else(|_| Utc::now().timestamp() as f64);

        let now = Utc::now().timestamp() as f64;
        let elapsed = (now - last_accessed).max(0.0);

        // Stability increases with importance and access count
        let stability = DEFAULT_STABILITY
            * (1.0 + IMPORTANCE_BOOST as f64 * entry.importance as f64)
            * (1.0 + ACCESS_BOOST * entry.access_count as f64 / 10.0);

        let retention = (-elapsed / stability).exp();
        retention as f32
    }

    /// Convert a MemoryEntry to a VectorEntry for HNSW insertion.
    fn memory_to_vector(entry: &MemoryEntry) -> DbEntry {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "memory".to_string());
        metadata.insert("content".to_string(), entry.content.clone());
        metadata.insert("role".to_string(), serde_json::to_string(&entry.role).unwrap_or_default());
        metadata.insert("timestamp".to_string(), entry.timestamp.clone());
        metadata.insert("access_count".to_string(), entry.access_count.to_string());
        metadata.insert("importance".to_string(), entry.importance.to_string());
        if let Some(ref sid) = entry.session_id {
            metadata.insert("session_id".to_string(), sid.clone());
        }
        for (k, v) in &entry.metadata {
            metadata.insert(format!("meta_{}", k), v.clone());
        }

        DbEntry {
            id: entry.id.clone(),
            values: vec![0.0; 384], // placeholder, will be replaced by hash_embed
            metadata,
        }
    }

    /// Convert a VectorEntry (from HNSW) back to a MemoryEntry.
    fn vector_to_memory(entry: &DbEntry) -> Option<MemoryEntry> {
        if entry.metadata.get("type")? != "memory" {
            return None;
        }
        let content = entry.metadata.get("content")?.clone();
        let role: MemoryRole = entry.metadata.get("role")
            .and_then(|r| serde_json::from_str(r).ok())
            .unwrap_or(MemoryRole::Observation);
        let session_id = entry.metadata.get("session_id").cloned();
        let timestamp = entry.metadata.get("timestamp").cloned().unwrap_or_default();
        let access_count = entry.metadata.get("access_count")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let importance = entry.metadata.get("importance")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);

        // Extract custom metadata (meta_* keys)
        let mut metadata = HashMap::new();
        for (k, v) in &entry.metadata {
            if let Some(key) = k.strip_prefix("meta_") {
                metadata.insert(key.to_string(), v.clone());
            }
        }

        Some(MemoryEntry {
            id: entry.id.clone(),
            content,
            role,
            session_id,
            timestamp,
            access_count,
            importance,
            metadata,
        })
    }

    /// Alias for vector_to_memory, used by rebuild_timeline.
    fn entry_from_metadata(entry: &DbEntry) -> Option<MemoryEntry> {
        Self::vector_to_memory(entry)
    }
}

#[async_trait]
impl MemoryStore for HnswMemoryStore {
    async fn remember(
        &self,
        content: &str,
        role: MemoryRole,
        session_id: Option<&str>,
        importance: f32,
        metadata: HashMap<String, String>,
    ) -> Result<MemoryEntry, MemoryStoreError> {
        let uuid_str = uuid::Uuid::new_v4().to_string().replace("-", "");
        let id = format!("mem_{}", &uuid_str[..12]);
        let entry = MemoryEntry {
            id: id.clone(),
            content: content.to_string(),
            role,
            session_id: session_id.map(|s| s.to_string()),
            timestamp: Utc::now().to_rfc3339(),
            access_count: 0,
            importance,
            metadata,
        };

        // Generate embedding and insert into HNSW
        let values = self.hash_embed(content);
        let mut vector_entry = Self::memory_to_vector(&entry);
        vector_entry.values = values;

        {
            let mut db = self.vector_db.write();
            db.insert(vector_entry).map_err(|e| {
                MemoryStoreError::Internal(format!("insert failed: {e}"))
            })?;
        }

        // Update timeline index
        {
            let mut timeline = self.timeline.write();
            if let Some(ref sid) = entry.session_id {
                timeline.sessions.entry(sid.clone()).or_default().push(id.clone());
            }
            timeline.entries.insert(id, entry.clone());
        }

        Ok(entry)
    }

    async fn recall(
        &self,
        query: &str,
        k: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryHit>, MemoryStoreError> {
        let query_vec = self.hash_embed(query);

        let results = {
            let db = self.vector_db.read();

            let mut filter = HashMap::new();
            filter.insert("type".to_string(), "memory".to_string());
            if let Some(sid) = session_id {
                filter.insert("session_id".to_string(), sid.to_string());
            }

            db.search_with_filter(&query_vec, k, Some(&filter))
        };

        let hits: Vec<MemoryHit> = results.into_iter()
            .filter_map(|hit| {
                let entry = Self::vector_to_memory(&DbEntry {
                    id: hit.id.clone(),
                    values: vec![],
                    metadata: hit.metadata.clone(),
                })?;
                let retention = Self::retention_score(&entry);
                Some(MemoryHit {
                    entry,
                    score: hit.score,
                    retention,
                })
            })
            .collect();

        Ok(hits)
    }

    async fn recent(
        &self,
        n: usize,
        session_id: Option<&str>,
    ) -> Result<Vec<MemoryEntry>, MemoryStoreError> {
        let timeline = self.timeline.read();

        let entries: Vec<MemoryEntry> = if let Some(sid) = session_id {
            // Return entries for specific session
            timeline.sessions.get(sid)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| timeline.entries.get(id).cloned())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            // Return all entries, sorted by timestamp (newest first)
            let mut all: Vec<MemoryEntry> = timeline.entries.values().cloned().collect();
            all.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            all
        };

        Ok(entries.into_iter().take(n).collect())
    }

    async fn touch(&self, id: &str) -> Result<(), MemoryStoreError> {
        let mut timeline = self.timeline.write();
        if let Some(entry) = timeline.entries.get_mut(id) {
            entry.access_count += 1;
            entry.timestamp = Utc::now().to_rfc3339();

            // Also update the vector entry's metadata in HNSW
            let values = self.hash_embed(&entry.content);
            let mut vector_entry = Self::memory_to_vector(entry);
            vector_entry.values = values;

            {
                let mut db = self.vector_db.write();
                db.insert(vector_entry).map_err(|e| {
                    MemoryStoreError::Internal(format!("update failed: {e}"))
                })?;
            }
            Ok(())
        } else {
            Err(MemoryStoreError::NotFound(id.to_string()))
        }
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, MemoryStoreError> {
        let timeline = self.timeline.read();
        Ok(timeline.entries.get(id).cloned())
    }

    async fn forget(&self, id: &str) -> Result<bool, MemoryStoreError> {
        // Remove from HNSW
        let deleted = {
            let mut db = self.vector_db.write();
            db.delete(&[id.to_string()])
        };

        // Remove from timeline
        {
            let mut timeline = self.timeline.write();
            if let Some(entry) = timeline.entries.remove(id) {
                if let Some(ref sid) = entry.session_id {
                    if let Some(ids) = timeline.sessions.get_mut(sid) {
                        ids.retain(|i| i != id);
                    }
                }
            }
        }

        Ok(deleted > 0)
    }

    async fn stats(&self) -> MemoryStoreStats {
        let timeline = self.timeline.read();
        let total = timeline.entries.len();
        let active = timeline.entries.values()
            .filter(|e| Self::retention_score(e) > 0.5)
            .count();
        let cold = total - active;
        let sessions = timeline.sessions.len();

        MemoryStoreStats {
            total_memories: total,
            active_memories: active,
            cold_memories: cold,
            sessions,
        }
    }
}
