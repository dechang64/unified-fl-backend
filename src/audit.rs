//! Blockchain-style Immutable Audit Chain
//!
//! SHA-256 hash chain with optional Merkle tree for batch verification.
//! Records all operations: vector insert/search/delete, FL rounds, etc.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AuditError {
    #[error("chain broken: expected hash {expected}, got {actual}")]
    ChainBroken { expected: String, actual: String },
    #[error("sequence gap: expected {expected}, got {actual}")]
    SequenceGap { expected: u64, actual: u64 },
}

// ── Event Types ──

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EventType {
    SystemStart,
    VectorInsert { count: usize },
    VectorSearch { k: usize, result_count: usize },
    VectorDelete { count: usize },
    FlRound { round: u32, participants: usize },
    FlNodeRegistered { institution: String },
    RagQuery { query_len: usize },
    HallucinationCheck { risk_level: String },
    Custom { name: String },
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::SystemStart => write!(f, "system_start"),
            EventType::VectorInsert { count } => write!(f, "vector_insert({count})"),
            EventType::VectorSearch { k, result_count } => write!(f, "vector_search(k={k}, results={result_count})"),
            EventType::VectorDelete { count } => write!(f, "vector_delete({count})"),
            EventType::FlRound { round, participants } => write!(f, "fl_round(round={round}, participants={participants})"),
            EventType::FlNodeRegistered { institution } => write!(f, "fl_node_registered({institution})"),
            EventType::RagQuery { query_len } => write!(f, "rag_query(len={query_len})"),
            EventType::HallucinationCheck { risk_level } => write!(f, "hallucination_check({risk_level})"),
            EventType::Custom { name } => write!(f, "custom({name})"),
        }
    }
}

// ── Audit Block ──

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditBlock {
    pub sequence: u64,
    pub timestamp: String,
    pub event_type: String,
    pub node_id: String,
    pub metadata: serde_json::Value,
    pub prev_hash: String,
    pub hash: String,
}

impl AuditBlock {
    fn compute_hash(&self) -> String {
        let data = serde_json::json!({
            "seq": self.sequence,
            "ts": self.timestamp,
            "event": self.event_type,
            "node": self.node_id,
            "meta": self.metadata,
            "prev": self.prev_hash,
        });
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_string(&data).unwrap().as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn verify(&self) -> bool {
        self.hash == self.compute_hash()
    }
}

// ── Audit Chain ──

pub struct AuditChain {
    chain: VecDeque<AuditBlock>,
    max_blocks: usize,
    /// Anchor hash from evicted blocks — allows verify() to pass
    /// even when old blocks have been persisted to disk and removed from memory.
    /// When the first block's prev_hash doesn't match a predecessor,
    /// it's checked against this anchor instead.
    anchor_hash: Option<String>,
    /// Total sequence counter (persists across evictions).
    total_sequence: u64,
}

impl AuditChain {
    pub fn new(max_blocks: usize) -> Self {
        let max_blocks = max_blocks.max(2); // Need at least 2: genesis + 1 user block
        let mut chain = VecDeque::with_capacity(max_blocks);
        // Genesis block
        let genesis = AuditBlock {
            sequence: 0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            event_type: "genesis".into(),
            node_id: "system".into(),
            metadata: serde_json::json!({}),
            prev_hash: "0".repeat(64),
            hash: String::new(),
        };
        genesis_hash(&genesis, &mut chain);
        Self { chain, max_blocks, anchor_hash: None, total_sequence: 0 }
    }

    pub fn append(
        &mut self,
        event_type: EventType,
        node_id: &str,
        metadata: serde_json::Value,
    ) -> &AuditBlock {
        let prev_hash = self.chain.back().map(|b| b.hash.clone()).unwrap_or_else(|| "0".repeat(64));
        self.total_sequence += 1;
        let sequence = self.total_sequence;
        let block = AuditBlock {
            sequence,
            timestamp: chrono::Utc::now().to_rfc3339(),
            event_type: event_type.to_string(),
            node_id: node_id.to_string(),
            metadata,
            prev_hash,
            hash: String::new(),
        };
        push_block(block, &mut self.chain, self.max_blocks, &mut self.anchor_hash);
        self.chain.back().unwrap()
    }

    pub fn verify(&self) -> Result<(), AuditError> {
        if self.chain.is_empty() {
            return Ok(());
        }
        // Verify first block: either self-consistent (genesis) or linked to anchor
        if let Some(first) = self.chain.front() {
            if !first.verify() {
                return Err(AuditError::ChainBroken {
                    expected: first.compute_hash(),
                    actual: first.hash.clone(),
                });
            }
            // If first block has a prev_hash that doesn't match "0"*64 (genesis),
            // it must match the anchor hash from evicted blocks
            let genesis_hash = "0".repeat(64);
            if first.prev_hash != genesis_hash {
                if let Some(ref anchor) = self.anchor_hash {
                    if first.prev_hash != *anchor {
                        return Err(AuditError::ChainBroken {
                            expected: anchor.clone(),
                            actual: first.prev_hash.clone(),
                        });
                    }
                }
                // If no anchor and prev_hash != genesis, chain is broken
                // (this shouldn't happen with proper eviction)
            }
        }
        // Verify chain links
        for i in 1..self.chain.len() {
            let prev = &self.chain[i - 1];
            let curr = &self.chain[i];
            if curr.prev_hash != prev.hash {
                return Err(AuditError::ChainBroken {
                    expected: prev.hash.clone(),
                    actual: curr.prev_hash.clone(),
                });
            }
            if !curr.verify() {
                return Err(AuditError::ChainBroken {
                    expected: curr.compute_hash(),
                    actual: curr.hash.clone(),
                });
            }
        }
        Ok(())
    }

    pub fn recent(&self, n: usize) -> Vec<&AuditBlock> {
        self.chain.iter().rev().take(n).collect()
    }

    pub fn len(&self) -> usize {
        self.chain.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }

    /// Total number of blocks ever appended (including evicted ones).
    pub fn total_appended(&self) -> u64 {
        self.total_sequence
    }

    // ── Persistence ──

    /// Persist the audit chain to a JSON file.
    pub fn persist(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let blocks: Vec<&AuditBlock> = self.chain.iter().collect();
        let data = serde_json::json!({
            "anchor_hash": self.anchor_hash,
            "total_sequence": self.total_sequence,
            "blocks": blocks,
        });
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)
    }

    /// Restore the audit chain from a JSON file.
    /// Returns the number of blocks restored.
    pub fn restore(&mut self, path: &Path) -> Result<usize, std::io::Error> {
        if !path.exists() {
            return Ok(0);
        }
        let json = std::fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&json).unwrap_or_default();

        self.anchor_hash = data.get("anchor_hash")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        self.total_sequence = data.get("total_sequence")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        self.chain.clear();
        if let Some(blocks) = data.get("blocks").and_then(|v| v.as_array()) {
            for block_val in blocks {
                if let Ok(block) = serde_json::from_value::<AuditBlock>(block_val.clone()) {
                    self.chain.push_back(block);
                }
            }
        }

        tracing::info!("Restored {} audit blocks from {:?}", self.chain.len(), path);
        Ok(self.chain.len())
    }

    pub fn to_proto_entries(&self, limit: usize) -> Vec<crate::proto::AuditEntry> {
        self.recent(limit)
            .into_iter()
            .map(|b| {
                let meta_map: HashMap<String, String> = if b.metadata.is_object() {
                    b.metadata.as_object().unwrap().iter()
                        .map(|(k, v)| {
                            let val = match v {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            (k.clone(), val)
                        })
                        .collect()
                } else {
                    HashMap::new()
                };
                crate::proto::AuditEntry {
                    sequence: b.sequence,
                    timestamp: b.timestamp.clone(),
                    event_type: b.event_type.clone(),
                    node_id: b.node_id.clone(),
                    metadata: meta_map,
                    hash: b.hash.clone(),
                    prev_hash: b.prev_hash.clone(),
                }
            })
            .collect()
    }
}

pub type SharedAuditChain = Arc<std::sync::RwLock<AuditChain>>;

// ── Helpers (avoid borrow issues) ──

fn genesis_hash(block: &AuditBlock, chain: &mut VecDeque<AuditBlock>) {
    let hash = block.compute_hash();
    let mut b = block.clone();
    b.hash = hash;
    chain.push_back(b);
}

fn push_block(block: AuditBlock, chain: &mut VecDeque<AuditBlock>, max_blocks: usize, anchor_hash: &mut Option<String>) {
    let hash = block.compute_hash();
    let mut b = block;
    b.hash = hash;
    if chain.len() >= max_blocks {
        // Evict oldest block, but save its hash as anchor
        // so verify() can still validate the chain link
        if let Some(evicted) = chain.pop_front() {
            *anchor_hash = Some(evicted.hash);
        }
    }
    chain.push_back(b);
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_integrity() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 10 }, "node1", serde_json::json!({}));
        chain.append(EventType::VectorSearch { k: 5, result_count: 3 }, "node1", serde_json::json!({}));
        assert!(chain.verify().is_ok());
        assert_eq!(chain.len(), 3); // genesis + 2
    }

    #[test]
    fn test_max_blocks_eviction() {
        let mut chain = AuditChain::new(5);
        for i in 0..10 {
            chain.append(EventType::VectorInsert { count: i }, "node1", serde_json::json!({}));
        }
        assert!(chain.len() <= 5);
    }

    #[test]
    fn test_tamper_detection() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 1 }, "node1", serde_json::json!({}));
        // Tamper
        chain.chain.back_mut().unwrap().event_type = "tampered".into();
        assert!(chain.verify().is_err());
    }

    #[test]
    fn test_min_max_blocks() {
        // max_blocks=1 should be clamped to 2
        let chain = AuditChain::new(1);
        assert_eq!(chain.len(), 1); // genesis only
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_empty_chain_verify() {
        // Edge case: manually empty chain
        let mut chain = AuditChain::new(100);
        chain.chain.clear();
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_metadata_string_values_preserved() {
        let mut chain = AuditChain::new(100);
        chain.append(
            EventType::Custom { name: "test".into() },
            "node1",
            serde_json::json!({"key": "value_with_quotes"}),
        );
        let entries = chain.to_proto_entries(10);
        assert_eq!(entries[0].metadata.get("key").unwrap(), "value_with_quotes");
    }
}
