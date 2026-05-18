//! SQLite Persistence Layer
//!
//! Persists vector entries and audit chain to disk for durability.

use anyhow::Result;
use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::Mutex;

pub struct Storage {
    conn: Mutex<Connection>,
}

impl Storage {
    pub fn open(path: &Path) -> Result<Self> {
        // Ensure parent directory exists
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
                hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_seq ON audit_blocks(sequence);"
        )?;

        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn save_vector(&self, id: &str, dimension: usize, values: &[f32], metadata: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
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

    pub fn load_vectors(&self) -> Result<Vec<(String, usize, Vec<f32>, String)>> {
        let conn = self.conn.lock().unwrap();
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
        let conn = self.conn.lock().unwrap();
        let affected = conn.execute("DELETE FROM vectors WHERE id = ?1", params![id])?;
        Ok(affected > 0)
    }

    pub fn save_audit_block(
        &self,
        sequence: u64,
        timestamp: &str,
        event_type: &str,
        node_id: &str,
        metadata: &str,
        prev_hash: &str,
        hash: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO audit_blocks (sequence, timestamp, event_type, node_id, metadata, prev_hash, hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![sequence as i64, timestamp, event_type, node_id, metadata, prev_hash, hash],
        )?;
        Ok(())
    }

    pub fn vector_count(&self) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM vectors", [], |r| r.get(0))?;
        Ok(count)
    }
}
