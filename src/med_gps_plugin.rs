//! Med GPS Scenario Plugin — 罕见病就医导航
//!
//! Implements ScenarioPlugin trait, decoupled from the core runtime.
//! gRPC service registration is done in main.rs at compile time.

use crate::traits::{ScenarioPlugin, CoreStores};
use crate::vector_db::SharedVectorDB;
use crate::audit::SharedAuditChain;
use crate::med_gps_service::MedGpsServiceImpl;

/// Medical GPS scenario plugin.
pub struct MedGpsPlugin {
    db: std::sync::OnceLock<SharedVectorDB>,
    audit: std::sync::OnceLock<SharedAuditChain>,
}

impl Default for MedGpsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl MedGpsPlugin {
    pub fn new() -> Self {
        Self {
            db: std::sync::OnceLock::new(),
            audit: std::sync::OnceLock::new(),
        }
    }

    /// Create the gRPC service implementation.
    /// Called by main.rs to register with tonic.
    pub fn create_grpc_service(&self) -> Option<crate::proto::med_gps_service_server::MedGpsServiceServer<MedGpsServiceImpl>> {
        let db = self.db.get()?.clone();
        let audit = self.audit.get()?.clone();
        Some(crate::med_gps_service::create_med_gps_server(db, audit))
    }

    /// Set the concrete stores directly (called by runtime during initialization).
    /// This avoids the need for downcast on trait objects.
    pub fn set_concrete_stores(
        &self,
        db: SharedVectorDB,
        audit: SharedAuditChain,
    ) -> Result<(), &'static str> {
        self.db.set(db).map_err(|_| "vector_db already initialized")?;
        self.audit.set(audit).map_err(|_| "audit_chain already initialized")?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl ScenarioPlugin for MedGpsPlugin {
    fn id(&self) -> &str {
        "med-gps"
    }

    fn name(&self) -> &str {
        "Medical GPS"
    }

    fn description(&self) -> &str {
        "Rare disease medical navigation — patient profile → hospital matching → route generation"
    }

    async fn initialize(
        &self,
        _stores: CoreStores,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // MedGpsPlugin uses concrete types (SharedVectorDB, SharedAuditChain)
        // which are set via set_concrete_stores() by the runtime.
        // The trait-object CoreStores are available for plugins that don't
        // need concrete types.
        Ok(())
    }

    fn rest_routes(&self) -> Option<axum::Router> {
        None
    }
}
