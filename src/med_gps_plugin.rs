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
        stores: CoreStores,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.db.set(stores.vector_db).map_err(|_| "vector_db already initialized")?;
        self.audit.set(stores.audit_chain).map_err(|_| "audit_chain already initialized")?;
        // memory_store is available in stores but MedGpsPlugin doesn't use it yet
        Ok(())
    }

    fn rest_routes(&self) -> Option<axum::Router> {
        // MedGps currently only uses gRPC; REST routes can be added here
        None
    }
}
