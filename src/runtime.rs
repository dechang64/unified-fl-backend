//! FedCtx Runtime — Scenario Registry and Server Bootstrap
//!
//! Manages scenario plugins and starts gRPC + REST servers.
//! Scenarios are registered at compile time via `register_scenarios!`.

use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use crate::traits::{ScenarioPlugin, CoreStores};
use crate::vector_db::SharedVectorDB;
use crate::audit::SharedAuditChain;
use crate::json_graph_store::JsonGraphStore;
use crate::memory_store::HnswMemoryStore;

/// The FedCtx runtime — holds all registered plugins and core stores.
pub struct FedCtxRuntime {
    plugins: Vec<Box<dyn ScenarioPlugin>>,
    vector_db: SharedVectorDB,
    graph_store: Arc<JsonGraphStore>,
    audit_chain: SharedAuditChain,
    memory_store: Arc<HnswMemoryStore>,
    data_dir: Option<PathBuf>,
    dimension: usize,
}

impl FedCtxRuntime {
    /// Create a new runtime with the given core store parameters.
    pub fn new(
        dimension: usize,
        max_audit_blocks: usize,
        data_dir: Option<PathBuf>,
    ) -> Self {
        let vector_db = Arc::new(std::sync::RwLock::new(
            crate::vector_db::VectorDB::new(dimension)
        ));
        let graph_store = Arc::new(JsonGraphStore::new());
        let audit_chain = Arc::new(std::sync::RwLock::new(
            crate::audit::AuditChain::new(max_audit_blocks)
        ));
        let memory_store = Arc::new(HnswMemoryStore::new(vector_db.clone(), dimension));

        let rt = Self {
            plugins: Vec::new(),
            vector_db,
            graph_store,
            audit_chain,
            memory_store,
            data_dir,
            dimension,
        };

        // Restore from data_dir if specified
        if let Some(ref dir) = rt.data_dir {
            rt.restore_from_disk(dir);
        }

        rt
    }

    /// Restore all stores from disk.
    fn restore_from_disk(&self, dir: &PathBuf) {
        std::fs::create_dir_all(dir).ok();

        // Restore vectors
        let vectors_path = dir.join("vectors.bin");
        if vectors_path.exists() {
            let mut db = self.vector_db.write().unwrap();
            if let Ok(count) = db.restore(&vectors_path) {
                info!("Restored {} vectors from {:?}", count, vectors_path);
            }
        }

        // Restore graph
        let graph_path = dir.join("graph.json");
        if graph_path.exists() {
            if let Ok(()) = self.graph_store.restore(&graph_path) {
                info!("Restored graph from {:?}", graph_path);
            }
        }

        // Restore audit chain
        let audit_path = dir.join("audit.json");
        if audit_path.exists() {
            let mut chain = self.audit_chain.write().unwrap();
            if let Ok(count) = chain.restore(&audit_path) {
                info!("Restored {} audit blocks from {:?}", count, audit_path);
            }
        }

        // Restore memory timeline from vector metadata
        // (memories are stored as vectors with type=memory, so they're
        // already restored with the vectors.bin file)
        self.memory_store.rebuild_timeline();
    }

    /// Persist all stores to disk.
    pub fn persist_to_disk(&self) {
        if let Some(ref dir) = self.data_dir {
            std::fs::create_dir_all(dir).ok();

            // Persist vectors
            let vectors_path = dir.join("vectors.bin");
            if let Ok(()) = self.vector_db.read().unwrap().persist(&vectors_path) {
                info!("Persisted vectors to {:?}", vectors_path);
            }

            // Persist graph
            let graph_path = dir.join("graph.json");
            if let Ok(()) = self.graph_store.persist(&graph_path) {
                info!("Persisted graph to {:?}", graph_path);
            }

            // Persist audit chain
            let audit_path = dir.join("audit.json");
            if let Ok(()) = self.audit_chain.read().unwrap().persist(&audit_path) {
                info!("Persisted audit chain to {:?}", audit_path);
            }
        }
    }

    /// Register a scenario plugin.
    pub fn register(&mut self, plugin: Box<dyn ScenarioPlugin>) {
        info!("Registering scenario plugin: {} ({})", plugin.name(), plugin.id());
        self.plugins.push(plugin);
    }

    /// Initialize all registered plugins.
    pub async fn initialize_plugins(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let stores = CoreStores {
            vector_db: self.vector_db.clone(),
            graph_store: self.graph_store.clone(),
            audit_chain: self.audit_chain.clone(),
            memory_store: self.memory_store.clone(),
        };

        for plugin in &self.plugins {
            info!("Initializing plugin: {} ({})", plugin.name(), plugin.id());
            plugin.initialize(stores.clone()).await?;
        }
        Ok(())
    }

    /// Get shared references to core stores.
    pub fn vector_db(&self) -> SharedVectorDB {
        self.vector_db.clone()
    }

    pub fn graph_store(&self) -> Arc<JsonGraphStore> {
        self.graph_store.clone()
    }

    pub fn audit_chain(&self) -> SharedAuditChain {
        self.audit_chain.clone()
    }

    pub fn memory_store(&self) -> Arc<HnswMemoryStore> {
        self.memory_store.clone()
    }

    /// List registered plugins.
    pub fn list_plugins(&self) -> Vec<(&str, &str, &str)> {
        self.plugins
            .iter()
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

    /// Shutdown all plugins.
    pub async fn shutdown(&self) {
        for plugin in &self.plugins {
            if let Err(e) = plugin.shutdown().await {
                tracing::error!("Plugin {} shutdown error: {}", plugin.id(), e);
            }
        }
    }
}
