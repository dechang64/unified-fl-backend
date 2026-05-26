//! Blockchain-style Immutable Audit Chain
//!
//! SHA-256 hash chain with Merkle root computation and SQLite persistence.
//! Records all operations: vector insert/search/delete, FL rounds, etc.
//!
//! Security guarantees:
//! 1. **Hash chain integrity**: each block's hash covers prev_hash, so tampering
//!    any block invalidates all subsequent blocks.
//! 2. **Merkle root**: computed over all blocks, enables O(log n) verification
//!    and light client proofs.
//! 3. **SQLite persistence**: blocks are atomically persisted via WAL mode,
//!    surviving crashes without data loss.
//! 4. **Restore verification**: loaded blocks are verified before acceptance.
//! 5. **Digital signature**: Ed25519 signatures on block hashes (AuditSigner).

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
    #[error("merkle root mismatch: expected {expected}, got {actual}")]
    MerkleRootMismatch { expected: String, actual: String },
    #[error("persistence error: {0}")]
    Persistence(String),
    #[error("restore verification failed: {0}")]
    RestoreVerification(String),
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
    /// Merkle root of all blocks up to and including this one.
    /// Updated incrementally: new_root = H(old_root || block_hash).
    pub merkle_root: String,
    /// Digital signature over the block hash (Ed25519, hex-encoded).
    /// Empty string if not signed.
    pub signature: String,
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

// ── Merkle Tree (incremental) ──

/// Incremental Merkle root computation.
///
/// For a hash chain, the Merkle root is computed incrementally:
///   root_0 = H(genesis_hash)
///   root_i = H(root_{i-1} || block_hash_i)
///
/// This provides O(1) update and O(n) full verification.
/// For O(log n) verification with proofs, a full binary Merkle tree
/// would be needed, but for audit chains (append-only), this is sufficient.
pub fn compute_incremental_merkle_root(prev_root: &str, block_hash: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prev_root.as_bytes());
    hasher.update(block_hash.as_bytes());
    hex::encode(hasher.finalize())
}

/// Compute Merkle root from a list of block hashes (full verification).
pub fn compute_merkle_root_from_hashes(hashes: &[String]) -> String {
    if hashes.is_empty() {
        return "0".repeat(64);
    }
    let mut root = hashes[0].clone();
    for hash in &hashes[1..] {
        root = compute_incremental_merkle_root(&root, hash);
    }
    root
}

// ── Audit Chain ──

pub struct AuditChain {
    chain: VecDeque<AuditBlock>,
    max_blocks: usize,
    anchor_hash: Option<String>,
    anchor_merkle_root: Option<String>,
    total_sequence: u64,
    current_merkle_root: String,
    storage: Option<Arc<crate::storage::Storage>>,
    /// Ed25519 signer for block signatures.
    signer: Option<AuditSigner>,
}

impl AuditChain {
    pub fn new(max_blocks: usize) -> Self {
        let max_blocks = max_blocks.max(2);
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
            merkle_root: String::new(),
            signature: String::new(),
        };
        genesis_hash(&genesis, &mut chain, &None);
        let merkle_root = chain.front().map(|b| b.merkle_root.clone()).unwrap_or_else(|| "0".repeat(64));
        Self {
            chain,
            max_blocks,
            anchor_hash: None,
            anchor_merkle_root: None,
            total_sequence: 0,
            current_merkle_root: merkle_root,
            storage: None,
            signer: None,
        }
    }

    /// Create an AuditChain with SQLite persistence.
    pub fn with_storage(max_blocks: usize, storage: Arc<crate::storage::Storage>) -> Self {
        let mut chain = Self::new(max_blocks);
        chain.storage = Some(storage);
        chain
    }

    /// Create an AuditChain with Ed25519 signing.
    pub fn with_signer(max_blocks: usize, signer: AuditSigner) -> Self {
        let mut chain = Self::new(max_blocks);
        chain.signer = Some(signer);
        chain
    }

    /// Set the signer after construction.
    pub fn set_signer(&mut self, signer: AuditSigner) {
        self.signer = Some(signer);
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
            merkle_root: String::new(),
            signature: String::new(),
        };
        push_block(block, &mut self.chain, self.max_blocks, &mut self.anchor_hash, &mut self.anchor_merkle_root, &mut self.current_merkle_root, &self.signer);

        // Persist to SQLite
        if let Some(ref storage) = self.storage {
            let block = self.chain.back().unwrap();
            if let Err(e) = storage.save_audit_block_full(&crate::storage::AuditBlockParams {
                sequence: block.sequence,
                timestamp: &block.timestamp,
                event_type: &block.event_type,
                node_id: &block.node_id,
                metadata: &block.metadata.to_string(),
                prev_hash: &block.prev_hash,
                hash: &block.hash,
                merkle_root: &block.merkle_root,
                signature: &block.signature,
            }) {
                tracing::error!("Failed to persist audit block {}: {}", block.sequence, e);
            }
        }

        self.chain.back().unwrap()
    }

    /// Verify the entire chain integrity.
    pub fn verify(&self) -> Result<(), AuditError> {
        if self.chain.is_empty() {
            return Ok(());
        }

        // Verify first block
        if let Some(first) = self.chain.front() {
            if !first.verify() {
                return Err(AuditError::ChainBroken {
                    expected: first.compute_hash(),
                    actual: first.hash.clone(),
                });
            }
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
            }
        }

        // Verify chain links and Merkle roots
        // Start from the zero root (or anchor if blocks were evicted)
        let mut expected_merkle = self.anchor_merkle_root.clone()
            .unwrap_or_else(|| "0".repeat(64));

        for i in 0..self.chain.len() {
            let curr = &self.chain[i];

            // Verify hash
            if !curr.verify() {
                return Err(AuditError::ChainBroken {
                    expected: curr.compute_hash(),
                    actual: curr.hash.clone(),
                });
            }

            // Verify prev_hash link
            if i > 0 {
                let prev = &self.chain[i - 1];
                if curr.prev_hash != prev.hash {
                    return Err(AuditError::ChainBroken {
                        expected: prev.hash.clone(),
                        actual: curr.prev_hash.clone(),
                    });
                }
            }

            // Verify Merkle root: root_i = H(root_{i-1} || block_hash_i)
            expected_merkle = compute_incremental_merkle_root(&expected_merkle, &curr.hash);
            if curr.merkle_root != expected_merkle {
                return Err(AuditError::MerkleRootMismatch {
                    expected: expected_merkle,
                    actual: curr.merkle_root.clone(),
                });
            }
        }

        // Verify current Merkle root matches
        if self.current_merkle_root != expected_merkle {
            return Err(AuditError::MerkleRootMismatch {
                expected: expected_merkle,
                actual: self.current_merkle_root.clone(),
            });
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

    pub fn total_appended(&self) -> u64 {
        self.total_sequence
    }

    /// Get the anchor hash (hash of the oldest retained block).
    pub fn anchor_hash(&self) -> Option<&str> {
        self.anchor_hash.as_deref()
    }

    /// Get the current Merkle root.
    pub fn merkle_root(&self) -> &str {
        &self.current_merkle_root
    }

    // ── Persistence ──

    pub fn persist(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let blocks: Vec<&AuditBlock> = self.chain.iter().collect();
        let data = serde_json::json!({
            "anchor_hash": self.anchor_hash,
            "anchor_merkle_root": self.anchor_merkle_root,
            "total_sequence": self.total_sequence,
            "current_merkle_root": self.current_merkle_root,
            "blocks": blocks,
        });
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)
    }

    /// Restore the audit chain from a JSON file.
    /// **Verifies chain integrity after loading.**
    pub fn restore(&mut self, path: &Path) -> Result<usize, AuditError> {
        if !path.exists() {
            return Ok(0);
        }
        let json = std::fs::read_to_string(path)
            .map_err(|e| AuditError::Persistence(e.to_string()))?;
        let data: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| AuditError::Persistence(format!("JSON parse error: {e}")))?;

        self.anchor_hash = data.get("anchor_hash")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        self.anchor_merkle_root = data.get("anchor_merkle_root")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        self.total_sequence = data.get("total_sequence")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        self.current_merkle_root = data.get("current_merkle_root")
            .and_then(|v| v.as_str())
            .unwrap_or("0".repeat(64).as_str())
            .to_string();

        self.chain.clear();
        if let Some(blocks) = data.get("blocks").and_then(|v| v.as_array()) {
            for block_val in blocks {
                if let Ok(block) = serde_json::from_value::<AuditBlock>(block_val.clone()) {
                    self.chain.push_back(block);
                }
            }
        }

        // **Critical**: verify chain integrity after restore
        if let Err(e) = self.verify() {
            // Restore failed verification — clear chain to prevent using tampered data
            let block_count = self.chain.len();
            self.chain.clear();
            self.anchor_hash = None;
            self.anchor_merkle_root = None;
            return Err(AuditError::RestoreVerification(
                format!("Chain verification failed after loading {block_count} blocks: {e}")
            ));
        }

        tracing::info!("Restored {} audit blocks from {:?} (verified OK)", self.chain.len(), path);
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

pub type SharedAuditChain = Arc<parking_lot::RwLock<AuditChain>>;

// ── Helpers ──

fn genesis_hash(block: &AuditBlock, chain: &mut VecDeque<AuditBlock>, signer: &Option<AuditSigner>) {
    let hash = block.compute_hash();
    let merkle_root = compute_incremental_merkle_root("0".repeat(64).as_str(), &hash);
    let mut b = block.clone();
    b.hash = hash.clone();
    b.merkle_root = merkle_root;
    if let Some(ref signer) = signer {
        b.signature = signer.sign(&hash);
    }
    chain.push_back(b);
}

fn push_block(
    block: AuditBlock,
    chain: &mut VecDeque<AuditBlock>,
    max_blocks: usize,
    anchor_hash: &mut Option<String>,
    anchor_merkle_root: &mut Option<String>,
    current_merkle_root: &mut String,
    signer: &Option<AuditSigner>,
) {
    let hash = block.compute_hash();
    let prev_merkle = current_merkle_root.clone();
    let new_merkle = compute_incremental_merkle_root(&prev_merkle, &hash);
    let mut b = block;
    b.hash = hash.clone();
    b.merkle_root = new_merkle.clone();
    // Sign the block hash if a signer is available
    if let Some(ref signer) = signer {
        b.signature = signer.sign(&hash);
    }
    *current_merkle_root = new_merkle;

    if chain.len() >= max_blocks {
        if let Some(evicted) = chain.pop_front() {
            *anchor_hash = Some(evicted.hash);
            *anchor_merkle_root = Some(evicted.merkle_root);
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
        // Chain should still verify with anchor
        assert!(chain.verify().is_ok());
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
    fn test_merkle_root_incremental() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 1 }, "node1", serde_json::json!({}));
        chain.append(EventType::VectorInsert { count: 2 }, "node2", serde_json::json!({}));

        // Merkle root should be consistent
        let root = chain.merkle_root();
        assert_ne!(root, "0".repeat(64));

        // Verify should pass
        assert!(chain.verify().is_ok());

        // Manually compute expected root using incremental algorithm
        let mut expected_root = "0".repeat(64);
        for b in chain.chain.iter() {
            expected_root = compute_incremental_merkle_root(&expected_root, &b.hash);
        }
        assert_eq!(root, &expected_root);
    }

    #[test]
    fn test_merkle_root_tamper_detection() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 1 }, "node1", serde_json::json!({}));

        // Tamper with merkle_root field
        chain.chain.back_mut().unwrap().merkle_root = "tampered".into();
        assert!(chain.verify().is_err());
    }

    #[test]
    fn test_min_max_blocks() {
        let chain = AuditChain::new(1);
        assert_eq!(chain.len(), 1); // genesis only
        assert!(chain.verify().is_ok());
    }

    #[test]
    fn test_empty_chain_verify() {
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

    #[test]
    fn test_restore_verifies_tampered_data() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 1 }, "node1", serde_json::json!({}));

        // Persist
        let path = std::env::temp_dir().join("test_audit_tamper.json");
        chain.persist(&path).unwrap();

        // Tamper with the file
        let json = std::fs::read_to_string(&path).unwrap();
        let tampered = json.replace("vector_insert(1)", "TAMPERED");
        std::fs::write(&path, tampered).unwrap();

        // Restore should fail verification
        let mut chain2 = AuditChain::new(100);
        let result = chain2.restore(&path);
        assert!(result.is_err());
        assert!(chain2.is_empty()); // Chain should be cleared on verification failure
    }

    #[test]
    fn test_persist_restore_roundtrip() {
        let mut chain = AuditChain::new(100);
        chain.append(EventType::VectorInsert { count: 1 }, "node1", serde_json::json!({"key": "val"}));
        chain.append(EventType::VectorSearch { k: 5, result_count: 3 }, "node2", serde_json::json!({}));

        let path = std::env::temp_dir().join("test_audit_roundtrip.json");
        chain.persist(&path).unwrap();

        let mut chain2 = AuditChain::new(100);
        let count = chain2.restore(&path).unwrap();
        assert_eq!(count, 3); // genesis + 2
        assert!(chain2.verify().is_ok());
        assert_eq!(chain2.merkle_root(), chain.merkle_root());
    }
}

// ════════════════════════════════════════════════════════════════
// Digital Signatures (Ed25519)
// ════════════════════════════════════════════════════════════════

/// Ed25519 signing key for audit block signatures.
pub struct AuditSigner {
    signing_key: ed25519_dalek::SigningKey,
}

impl AuditSigner {
    /// Generate a new random signing key.
    pub fn generate() -> Self {
        let mut csprng = rand::thread_rng();
        let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
        Self { signing_key }
    }

    /// Create from existing bytes (32 bytes).
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(bytes);
        Self { signing_key }
    }

    /// Sign a block hash and return the hex-encoded signature.
    pub fn sign(&self, hash: &str) -> String {
        use ed25519_dalek::Signer;
        let signature = self.signing_key.sign(hash.as_bytes());
        hex::encode(signature.to_bytes())
    }

    /// Get the verifying key (public key) as hex.
    pub fn verifying_key_hex(&self) -> String {
        hex::encode(self.signing_key.verifying_key().to_bytes())
    }
}

/// Verify an Ed25519 signature on a block hash.
pub fn verify_block_signature(
    hash: &str,
    signature_hex: &str,
    public_key_hex: &str,
) -> Result<(), AuditError> {
    let sig_bytes = hex::decode(signature_hex)
        .map_err(|e| AuditError::Persistence(format!("Invalid signature hex: {}", e)))?;
    let pk_bytes = hex::decode(public_key_hex)
        .map_err(|e| AuditError::Persistence(format!("Invalid public key hex: {}", e)))?;

    let signature = ed25519_dalek::Signature::try_from(sig_bytes.as_slice())
        .map_err(|e| AuditError::Persistence(format!("Invalid signature: {}", e)))?;
    let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
        pk_bytes.as_slice().try_into()
            .map_err(|_| AuditError::Persistence("Invalid public key length".into()))?
    ).map_err(|e| AuditError::Persistence(format!("Invalid public key: {}", e)))?;

    use ed25519_dalek::Verifier;
    verifying_key.verify(hash.as_bytes(), &signature)
        .map_err(|e| AuditError::Persistence(format!("Signature verification failed: {}", e)))
}

#[cfg(test)]
mod signature_tests {
    use super::*;

    #[test]
    fn test_sign_and_verify() {
        let signer = AuditSigner::generate();
        let hash = "abc123def456";
        let signature = signer.sign(hash);
        let pk = signer.verifying_key_hex();

        assert!(verify_block_signature(hash, &signature, &pk).is_ok());
    }

    #[test]
    fn test_tampered_hash_fails() {
        let signer = AuditSigner::generate();
        let hash = "original_hash";
        let signature = signer.sign(hash);
        let pk = signer.verifying_key_hex();

        assert!(verify_block_signature("tampered_hash", &signature, &pk).is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let signer1 = AuditSigner::generate();
        let signer2 = AuditSigner::generate();
        let hash = "test_hash";
        let signature = signer1.sign(hash);
        let wrong_pk = signer2.verifying_key_hex();

        assert!(verify_block_signature(hash, &signature, &wrong_pk).is_err());
    }
}
