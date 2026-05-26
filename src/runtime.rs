//! FedCtx Runtime — Scenario Registry and Server Bootstrap
//!
//! Manages scenario plugins and starts gRPC + REST servers.
//! Scenarios are registered at compile time via `register_scenarios!`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

use crate::traits::{ScenarioPlugin, CoreStores};
use crate::vector_db::SharedVectorDB;
use crate::audit::SharedAuditChain;
use crate::hnsw_store::HnswVectorStore;
use crate::json_graph_store::JsonGraphStore;
use crate::chain_audit_log::ChainAuditLog;
use crate::memory_store::HnswMemoryStore;
use crate::embedding::default_embedding_provider;

/// The FedCtx runtime — holds all registered plugins and core stores.
pub struct FedCtxRuntime {
    plugins: Vec<Box<dyn ScenarioPlugin>>,
    stores: CoreStores,
    // Concrete types for backward compat with gRPC services
    vector_db: SharedVectorDB,
    audit_chain: SharedAuditChain,
    graph_store_concrete: Arc<JsonGraphStore>,
    memory_store_concrete: Arc<HnswMemoryStore>,
    fl_engine: Arc<crate::fl_engine::FlEngine>,
    data_dir: Option<PathBuf>,
    dimension: usize,
    auto_persist_interval: Option<Duration>,
}

impl FedCtxRuntime {
    /// Create a new runtime with the given core store parameters.
    pub fn new(
        dimension: usize,
        max_audit_blocks: usize,
        data_dir: Option<PathBuf>,
    ) -> Self {
        Self::with_options(dimension, max_audit_blocks, data_dir, None, None)
    }

    /// Create a new runtime with all options.
    pub fn with_options(
        dimension: usize,
        max_audit_blocks: usize,
        data_dir: Option<PathBuf>,
        embedding_model_dir: Option<PathBuf>,
        auto_persist_secs: Option<u64>,
    ) -> Self {
        // Create concrete implementations
        let vector_db = Arc::new(parking_lot::RwLock::new(
            crate::vector_db::VectorDB::new(dimension)
        ));
        let graph_store = Arc::new(JsonGraphStore::new());
        let audit_chain = Arc::new(parking_lot::RwLock::new(
            crate::audit::AuditChain::new(max_audit_blocks)
        ));
        let memory_store = Arc::new(HnswMemoryStore::new(vector_db.clone(), dimension));

        // Create embedding provider
        let embedding: Arc<dyn crate::embedding::EmbeddingProvider> =
            default_embedding_provider(dimension, embedding_model_dir.as_ref().map(|p| p.to_str().unwrap_or("")));

        // Create FL engine with FedAvg as default strategy + audit logging
        let chain_audit_log: Arc<dyn crate::traits::AuditLog> = Arc::new(
            crate::chain_audit_log::ChainAuditLog::from_shared(audit_chain.clone())
        );
        let fl_engine = Arc::new(crate::fl_engine::FlEngine::new(
            Arc::new(crate::fl_engine::FedAvg),
            1, // min_clients
        ).with_audit(chain_audit_log));

        // Load from disk if data_dir exists
        if let Some(ref dir) = data_dir {
            if dir.exists() {
                Self::load_from_disk(&vector_db, &graph_store, &audit_chain, &memory_store, dir);
            }
        }

        // Build trait-object CoreStores
        let hnsw_vector_store = Arc::new(HnswVectorStore::from_shared(vector_db.clone()));
        let chain_audit_log = Arc::new(ChainAuditLog::from_shared(audit_chain.clone()));

        let stores = CoreStores {
            vector_store: hnsw_vector_store,
            graph_store: graph_store.clone() as Arc<dyn crate::traits::GraphStore>,
            audit_log: chain_audit_log,
            memory_store: memory_store.clone() as Arc<dyn crate::traits::MemoryStore>,
            fl_engine: Some(fl_engine.clone()),
            embedding: Some(embedding),
        };

        let auto_persist_interval = auto_persist_secs
            .filter(|&s| s > 0)
            .map(Duration::from_secs);

        Self {
            plugins: Vec::new(),
            stores,
            vector_db,
            audit_chain,
            graph_store_concrete: graph_store,
            memory_store_concrete: memory_store,
            fl_engine,
            data_dir,
            dimension,
            auto_persist_interval,
        }
    }

    /// Load persisted data from disk.
    fn load_from_disk(
        vector_db: &SharedVectorDB,
        graph_store: &Arc<JsonGraphStore>,
        audit_chain: &SharedAuditChain,
        memory_store: &Arc<HnswMemoryStore>,
        dir: &Path,
    ) {
        // Load vectors using existing persist/restore
        let vectors_path = dir.join("vectors.bin");
        if vectors_path.exists() {
            let mut db = vector_db.write();
            if let Ok(count) = db.restore(&vectors_path) {
                info!("Loaded {} vectors from disk", count);
            }
        }

        // Load graph using existing persist/restore
        let graph_path = dir.join("graph.json");
        if graph_path.exists() {
            if let Ok(()) = graph_store.restore(&graph_path) {
                info!("Loaded graph from disk");
            }
        }

        // Load audit chain using proper restore (preserves hashes & verifies integrity)
        let audit_path = dir.join("audit.json");
        if audit_path.exists() {
            let mut chain = audit_chain.write();
            match chain.restore(&audit_path) {
                Ok(count) => info!("Loaded {} audit blocks from disk (verified)", count),
                Err(e) => tracing::error!("Audit chain restore failed: {}", e),
            }
        }

        // Rebuild memory timeline from HNSW vector metadata
        // (memory entries are persisted as vectors with type=memory metadata)
        memory_store.rebuild_timeline();
        let mem_stats = memory_store.stats_blocking();
        if mem_stats.total_memories > 0 {
            info!("Rebuilt memory timeline: {} memories across {} sessions", mem_stats.total_memories, mem_stats.sessions);
        }
    }

    /// Register a scenario plugin.
    pub fn register(&mut self, plugin: Box<dyn ScenarioPlugin>) {
        let plugin_id = plugin.id().to_string();
        let plugin_name = plugin.name().to_string();
        self.plugins.push(plugin);
        info!("Registered plugin: {} ({})", plugin_name, plugin_id);
    }

    /// Initialize all plugins (call after register, in async context).
    pub async fn initialize_plugins(&self) {
        for plugin in &self.plugins {
            if let Err(e) = plugin.initialize(self.stores.clone()).await {
                tracing::error!("Plugin {} init error: {}", plugin.id(), e);
            }
        }
    }

    /// Start auto-persist background task.
    /// Returns the JoinHandle so the caller can abort it on shutdown.
    pub fn start_auto_persist(&self) -> Option<tokio::task::JoinHandle<()>> {
        let interval = self.auto_persist_interval?;
        let db = self.vector_db.clone();
        let audit = self.audit_chain.clone();
        let graph = self.graph_store_concrete.clone();
        let dir = self.data_dir.clone()?;

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                Self::do_persist(&db, &audit, &graph, &dir);
            }
        });
        info!("Auto-persist enabled: every {}s", interval.as_secs());
        Some(handle)
    }

    /// Persist all data to disk.
    pub fn persist_to_disk(&self) {
        if let Some(ref dir) = self.data_dir {
            Self::do_persist(&self.vector_db, &self.audit_chain, &self.graph_store_concrete, dir);
        }
    }

    fn do_persist(
        db: &SharedVectorDB,
        audit: &SharedAuditChain,
        graph: &Arc<JsonGraphStore>,
        dir: &Path,
    ) {
        std::fs::create_dir_all(dir).ok();

        // Persist vectors using existing persist()
        {
            let db_guard = db.read();
            let path = dir.join("vectors.bin");
            if let Err(e) = db_guard.persist(&path) {
                tracing::error!("Persist vectors: {}", e);
            }
        }

        // Persist graph using existing persist()
        {
            let path = dir.join("graph.json");
            if let Err(e) = graph.persist(&path) {
                tracing::error!("Persist graph: {}", e);
            }
        }

        // Persist audit
        {
            let chain = audit.read();
            let path = dir.join("audit.json");
            let entries = chain.to_proto_entries(chain.len());
            let json_entries: Vec<serde_json::Value> = entries.iter().map(|e| {
                serde_json::json!({
                    "sequence": e.sequence,
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "node_id": e.node_id,
                    "metadata": e.metadata,
                    "hash": e.hash,
                    "prev_hash": e.prev_hash,
                })
            }).collect();
            // Also persist chain-level metadata for verification on restore
            let audit_data = serde_json::json!({
                "anchor_hash": chain.anchor_hash(),
                "merkle_root": chain.merkle_root(),
                "total_appended": chain.total_appended(),
                "entries": json_entries,
            });
            if let Ok(json) = serde_json::to_string_pretty(&audit_data) {
                if let Err(e) = std::fs::write(&path, &json) {
                    tracing::error!("Persist audit: {}", e);
                }
            }
        }

        tracing::debug!("Persist completed to {:?}", dir);
    }

    // ── Accessors ──

    pub fn vector_db(&self) -> SharedVectorDB { self.vector_db.clone() }
    pub fn audit_chain(&self) -> SharedAuditChain { self.audit_chain.clone() }
    pub fn graph_store(&self) -> Arc<JsonGraphStore> { self.graph_store_concrete.clone() }
    pub fn memory_store(&self) -> Arc<HnswMemoryStore> { self.memory_store_concrete.clone() }
    pub fn fl_engine(&self) -> Arc<crate::fl_engine::FlEngine> { self.fl_engine.clone() }
    pub fn stores(&self) -> &CoreStores { &self.stores }
    pub fn dimension(&self) -> usize { self.dimension }

    /// List registered plugins.
    pub fn list_plugins(&self) -> Vec<(&str, &str, &str)> {
        self.plugins.iter()
            .map(|p| (p.id(), p.name(), p.description()))
            .collect()
    }

    /// Collect REST routes from all plugins and merge into a single router.
    pub fn collect_rest_routes(&self) -> axum::Router {
        let mut router = axum::Router::new();
        for plugin in &self.plugins {
            if let Some(routes) = plugin.rest_routes() {
                router = router.merge(routes);
            }
        }
        router
    }

    /// Shutdown all plugins and persist data.
    pub async fn shutdown(&self) {
        for plugin in &self.plugins {
            if let Err(e) = plugin.shutdown().await {
                tracing::error!("Plugin {} shutdown error: {}", plugin.id(), e);
            }
        }
        self.persist_to_disk();
        info!("All plugins shut down, data persisted");
    }
}
