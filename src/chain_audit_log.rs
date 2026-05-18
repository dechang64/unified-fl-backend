//! AuditLog trait implementation backed by the existing AuditChain

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;

use crate::traits::*;
use crate::audit::{AuditChain, EventType, SharedAuditChain};

/// AuditLog implementation backed by the existing SHA-256 AuditChain.
pub struct ChainAuditLog {
    chain: SharedAuditChain,
}

impl ChainAuditLog {
    pub fn new(max_blocks: usize) -> Self {
        Self {
            chain: Arc::new(std::sync::RwLock::new(AuditChain::new(max_blocks))),
        }
    }

    pub fn from_shared(chain: SharedAuditChain) -> Self {
        Self { chain }
    }

    /// Get a reference to the underlying shared AuditChain.
    /// Useful for backward compatibility with existing gRPC services.
    pub fn shared(&self) -> SharedAuditChain {
        self.chain.clone()
    }
}

fn metadata_to_json(meta: &HashMap<String, String>) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = meta
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    serde_json::Value::Object(map)
}

fn json_to_metadata(val: &serde_json::Value) -> HashMap<String, String> {
    if val.is_object() {
        val.as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), s)
            })
            .collect()
    } else {
        HashMap::new()
    }
}

#[async_trait]
impl AuditLog for ChainAuditLog {
    async fn append(
        &self,
        event_type: &str,
        node_id: &str,
        metadata: HashMap<String, String>,
    ) -> Result<AuditEntry, AuditLogError> {
        let mut chain = self.chain.write().map_err(|e| {
            AuditLogError::Internal(format!("audit lock poisoned: {e}"))
        })?;

        let event = EventType::Custom {
            name: event_type.to_string(),
        };
        let meta_json = metadata_to_json(&metadata);
        let block = chain.append(event, node_id, meta_json);

        Ok(AuditEntry {
            sequence: block.sequence,
            timestamp: block.timestamp.clone(),
            event_type: block.event_type.clone(),
            node_id: block.node_id.clone(),
            metadata: json_to_metadata(&block.metadata),
            hash: block.hash.clone(),
            prev_hash: block.prev_hash.clone(),
        })
    }

    async fn verify(&self) -> Result<(), AuditLogError> {
        let chain = self.chain.read().map_err(|e| {
            AuditLogError::Internal(format!("audit lock poisoned: {e}"))
        })?;
        chain.verify().map_err(|e| {
            AuditLogError::ChainBroken(e.to_string())
        })
    }

    async fn recent(&self, n: usize) -> Vec<AuditEntry> {
        let chain = self.chain.read().unwrap();
        chain
            .recent(n)
            .iter()
            .map(|b| AuditEntry {
                sequence: b.sequence,
                timestamp: b.timestamp.clone(),
                event_type: b.event_type.clone(),
                node_id: b.node_id.clone(),
                metadata: json_to_metadata(&b.metadata),
                hash: b.hash.clone(),
                prev_hash: b.prev_hash.clone(),
            })
            .collect()
    }

    async fn len(&self) -> usize {
        self.chain.read().unwrap().len()
    }
}
