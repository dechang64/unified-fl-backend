//! SQLite Persistence Layer
//!
//! Persists vector entries and audit blocks to disk for durability.
//! Uses WAL mode for crash-safe atomic writes.

use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;
use parking_lot::Mutex;

/// A loaded vector entry: (id, dimension, values, metadata_json).
pub type VectorRecord = (String, usize, Vec<f32>, String);

pub struct Storage {
    conn: Mutex<Connection>,
}

/// Parameters for saving an audit block.
pub struct AuditBlockParams<'a> {
    pub sequence: u64,
    pub timestamp: &'a str,
    pub event_type: &'a str,
    pub node_id: &'a str,
    pub metadata: &'a str,
    pub prev_hash: &'a str,
    pub hash: &'a str,
    pub merkle_root: &'a str,
    pub signature: &'a str,
}

impl<'a> AuditBlockParams<'a> {
    /// Create params with empty merkle_root and signature.
    pub fn new(
        sequence: u64,
        timestamp: &'a str,
        event_type: &'a str,
        node_id: &'a str,
        metadata: &'a str,
        prev_hash: &'a str,
        hash: &'a str,
    ) -> Self {
        Self {
            sequence, timestamp, event_type, node_id, metadata, prev_hash, hash,
            merkle_root: "",
            signature: "",
        }
    }
}

impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA foreign_keys=ON;"
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                values BLOB NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at);

            CREATE TABLE IF NOT EXISTS audit_blocks (
                sequence INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                node_id TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL,
                merkle_root TEXT NOT NULL DEFAULT '',
                signature TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_audit_seq ON audit_blocks(sequence);

            CREATE TABLE IF NOT EXISTS chain_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );"
        )?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn save_vector(&self, id: &str, dimension: usize, values: &[f32], metadata: &str) -> Result<()> {
        let conn = self.conn.lock();
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        conn.execute(
            "INSERT OR REPLACE INTO vectors (id, dimension, values, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![id, dimension as i64, bytes, metadata],
        )?;
        Ok(())
    }

    pub fn load_vectors(&self) -> Result<Vec<VectorRecord>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT id, dimension, values, metadata FROM vectors")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let dim: i64 = row.get(1)?;
            let bytes: Vec<u8> = row.get(2)?;
            let meta: String = row.get(3)?;
            let mut values = Vec::with_capacity(dim as usize);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok((id, dim as usize, values, meta))
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    pub fn delete_vector(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock();
        let affected = conn.execute("DELETE FROM vectors WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    pub fn save_audit_block(&self, params: &AuditBlockParams<'_>) -> Result<()> {
        self.save_audit_block_full(params)
    }

    pub fn save_audit_block_full(&self, params: &AuditBlockParams<'_>) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO audit_blocks (sequence, timestamp, event_type, node_id, metadata, prev_hash, hash, merkle_root, signature) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![params.sequence as i64, params.timestamp, params.event_type, params.node_id, params.metadata, params.prev_hash, params.hash, params.merkle_root, params.signature],
        )?;
        Ok(())
    }

    /// Save chain metadata (anchor_hash, merkle_root, etc.)
    pub fn save_chain_metadata(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO chain_metadata (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    /// Load chain metadata.
    pub fn load_chain_metadata(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT value FROM chain_metadata WHERE key = ?1")?;
        let result = stmt.query_row(params![key], |row| row.get::<_, String>(0)).ok();
        Ok(result)
    }

    pub fn vector_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM vectors", [], |r| r.get(0))?;
        Ok(count)
    }

    pub fn audit_block_count(&self) -> Result<i64> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM audit_blocks", [], |r| r.get(0))?;
        Ok(count)
    }
}
