//! Federated Learning Aggregation Engine
//!
//! Core FL capabilities: model update submission, aggregation strategies,
//! and round management. This makes FedCtx a true "federated" infrastructure.
//!
//! Supported strategies:
//! - FedAvg (McMahan et al., 2017): weighted average of model updates
//! - FedProx (Li et al., 2020): FedAvg + proximal term for Non-IID data
//! - EWA (Entropy-Weighted Aggregation): adaptive weighting based on
//!   client entropy to protect minority knowledge

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use rand::Rng;
use chrono::Utc;
use tracing::info;

use crate::traits::AuditLog;

// ════════════════════════════════════════════════════════════════
// Data Types
// ════════════════════════════════════════════════════════════════

/// A model update from a single client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Client identifier (e.g., institution name).
    pub client_id: String,
    /// Round number this update belongs to.
    pub round: u32,
    /// Model parameters as a flat vector of floats.
    pub parameters: Vec<f32>,
    /// Number of training samples used.
    pub num_samples: usize,
    /// Training loss reported by client.
    pub train_loss: Option<f32>,
    /// Softmax entropy of client's output distribution (for EWA).
    pub entropy: Option<f64>,
    /// Timestamp of submission.
    pub submitted_at: String,
}

/// Result of an aggregation round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Round number.
    pub round: u32,
    /// Aggregated model parameters.
    pub parameters: Vec<f32>,
    /// Number of clients that contributed.
    pub num_clients: usize,
    /// Total samples across all clients.
    pub total_samples: usize,
    /// Per-client weights used in aggregation.
    pub client_weights: HashMap<String, f64>,
    /// Aggregation strategy used.
    pub strategy: String,
    /// Timestamp of aggregation.
    pub aggregated_at: String,
}

/// Status of a federated round.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RoundStatus {
    /// Waiting for client updates.
    Collecting,
    /// Minimum clients reached, ready to aggregate.
    Ready,
    /// Aggregation completed.
    Completed,
}

/// State of a single federated round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundState {
    pub round: u32,
    pub status: RoundStatus,
    pub min_clients: usize,
    pub max_rounds: u32,
    pub updates: Vec<ModelUpdate>,
    pub result: Option<AggregationResult>,
    pub started_at: String,
}

/// Error type for FL operations.
#[derive(Debug, thiserror::Error)]
pub enum FlError {
    #[error("round {round} not found")]
    RoundNotFound { round: u32 },
    #[error("round {round} is not collecting (status: {status:?})")]
    RoundNotCollecting { round: u32, status: RoundStatus },
    #[error("client {client_id} already submitted for round {round}")]
    DuplicateSubmission { client_id: String, round: u32 },
    #[error("not enough clients: {actual} < {required}")]
    InsufficientClients { actual: usize, required: usize },
    #[error("parameter dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("no global model available")]
    NoGlobalModel,
    #[error("aggregation failed: {reason}")]
    AggregationFailed { reason: String },
    #[error("internal error: {0}")]
    Internal(String),
}

// ════════════════════════════════════════════════════════════════
// Aggregation Strategies
// ════════════════════════════════════════════════════════════════

/// Trait for aggregation strategies.
#[async_trait]
pub trait AggregationStrategy: Send + Sync {
    /// Name of the strategy.
    fn name(&self) -> &str;

    /// Aggregate model updates into a single model.
    async fn aggregate(
        &self,
        updates: &[ModelUpdate],
        global_params: Option<&[f32]>,
    ) -> Result<AggregationResult, FlError>;
}

/// FedAvg: weighted average by sample count.
pub struct FedAvg;

#[async_trait]
impl AggregationStrategy for FedAvg {
    fn name(&self) -> &str { "fedavg" }

    async fn aggregate(
        &self,
        updates: &[ModelUpdate],
        _global_params: Option<&[f32]>,
    ) -> Result<AggregationResult, FlError> {
        if updates.is_empty() {
            return Err(FlError::InsufficientClients { actual: 0, required: 1 });
        }

        let dim = updates[0].parameters.len();
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum::<usize>().max(1);
        let total_weight: f64 = updates.iter().map(|u| u.num_samples as f64).sum();

        let mut aggregated = vec![0.0f32; dim];
        let mut client_weights = HashMap::new();

        for update in updates {
            if update.parameters.len() != dim {
                return Err(FlError::DimensionMismatch {
                    expected: dim,
                    actual: update.parameters.len(),
                });
            }
            let weight = update.num_samples as f64 / total_weight;
            client_weights.insert(update.client_id.clone(), weight);
            for (i, p) in update.parameters.iter().enumerate() {
                aggregated[i] += *p * weight as f32;
            }
        }

        Ok(AggregationResult {
            round: updates[0].round,
            parameters: aggregated,
            num_clients: updates.len(),
            total_samples,
            client_weights,
            strategy: "fedavg".to_string(),
            aggregated_at: Utc::now().to_rfc3339(),
        })
    }
}

/// FedProx: FedAvg + proximal term (mu * ||w - w_global||^2).
/// The proximal term is applied during client training, not during aggregation.
/// At the server level, FedProx aggregation is identical to FedAvg.
/// The `mu` parameter is sent to clients so they can apply the proximal term.
pub struct FedProx {
    pub mu: f64,
}

impl FedProx {
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }
}

#[async_trait]
impl AggregationStrategy for FedProx {
    fn name(&self) -> &str { "fedprox" }

    async fn aggregate(
        &self,
        updates: &[ModelUpdate],
        global_params: Option<&[f32]>,
    ) -> Result<AggregationResult, FlError> {
        // FedProx server-side aggregation is FedAvg
        let fedavg = FedAvg;
        let mut result = fedavg.aggregate(updates, global_params).await?;
        result.strategy = format!("fedprox(mu={})", self.mu);
        Ok(result)
    }
}

/// EWA-Fed: Entropy-Weighted Aggregation.
///
/// Clients with lower entropy (more specialized/confident) get higher weight.
/// This protects minority expert knowledge from being washed out by
/// majority clients with high entropy (uncertain/uniform predictions).
///
/// Weight formula: w_i = exp(-alpha * entropy_i) / sum(exp(-alpha * entropy_j))
pub struct EWA {
    /// Entropy scaling factor. Higher alpha = more protection for low-entropy clients.
    pub alpha: f64,
}

impl EWA {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

#[async_trait]
impl AggregationStrategy for EWA {
    fn name(&self) -> &str { "ewa" }

    async fn aggregate(
        &self,
        updates: &[ModelUpdate],
        _global_params: Option<&[f32]>,
    ) -> Result<AggregationResult, FlError> {
        if updates.is_empty() {
            return Err(FlError::InsufficientClients { actual: 0, required: 1 });
        }

        let dim = updates[0].parameters.len();

        // Compute entropy-based weights
        let entropies: Vec<f64> = updates.iter()
            .map(|u| u.entropy.unwrap_or(1.0)) // Default entropy = 1.0 (uniform)
            .collect();

        // exp(-alpha * entropy) — low entropy → high weight
        let exp_weights: Vec<f64> = entropies.iter()
            .map(|e| (-self.alpha * e).exp())
            .collect();
        let total_exp_weight: f64 = exp_weights.iter().sum();

        let mut aggregated = vec![0.0f32; dim];
        let mut client_weights = HashMap::new();
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum::<usize>().max(1);

        for (i, update) in updates.iter().enumerate() {
            if update.parameters.len() != dim {
                return Err(FlError::DimensionMismatch {
                    expected: dim,
                    actual: update.parameters.len(),
                });
            }
            let weight = exp_weights[i] / total_exp_weight;
            client_weights.insert(update.client_id.clone(), weight);
            for (j, p) in update.parameters.iter().enumerate() {
                aggregated[j] += *p * weight as f32;
            }
        }

        Ok(AggregationResult {
            round: updates[0].round,
            parameters: aggregated,
            num_clients: updates.len(),
            total_samples,
            client_weights,
            strategy: format!("ewa(alpha={})", self.alpha),
            aggregated_at: Utc::now().to_rfc3339(),
        })
    }
}

// ════════════════════════════════════════════════════════════════
// FL Engine
// ════════════════════════════════════════════════════════════════

/// The federated learning engine — manages rounds, collects updates,
/// and aggregates them using a configurable strategy.
pub struct FlEngine {
    strategy: parking_lot::RwLock<Arc<dyn AggregationStrategy>>,
    rounds: Arc<parking_lot::RwLock<HashMap<u32, RoundState>>>,
    global_model: Arc<parking_lot::RwLock<Option<Vec<f32>>>>,
    current_round: Arc<parking_lot::RwLock<u32>>,
    min_clients: usize,
    audit_log: Option<Arc<dyn AuditLog>>,
}

impl FlEngine {
    pub fn new(
        strategy: Arc<dyn AggregationStrategy>,
        min_clients: usize,
    ) -> Self {
        Self {
            strategy: parking_lot::RwLock::new(strategy),
            rounds: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            global_model: Arc::new(parking_lot::RwLock::new(None)),
            current_round: Arc::new(parking_lot::RwLock::new(0)),
            min_clients,
            audit_log: None,
        }
    }

    pub fn with_audit(mut self, audit_log: Arc<dyn AuditLog>) -> Self {
        self.audit_log = Some(audit_log);
        self
    }

    /// Start a new federated round.
    pub fn start_round(&self) -> u32 {
        let mut current = self.current_round.write();
        *current += 1;
        let round = *current;

        let state = RoundState {
            round,
            status: RoundStatus::Collecting,
            min_clients: self.min_clients,
            max_rounds: 0,
            updates: Vec::new(),
            result: None,
            started_at: Utc::now().to_rfc3339(),
        };

        self.rounds.write().insert(round, state);
        info!("FL round {} started, waiting for {} clients", round, self.min_clients);
        round
    }

    /// Submit a model update from a client.
    pub fn submit_update(&self, update: ModelUpdate) -> Result<(), FlError> {
        let mut rounds = self.rounds.write();
        let state = rounds.get_mut(&update.round)
            .ok_or(FlError::RoundNotFound { round: update.round })?;

        if state.status != RoundStatus::Collecting {
            return Err(FlError::RoundNotCollecting {
                round: update.round,
                status: state.status.clone(),
            });
        }

        // Check for duplicate submission
        if state.updates.iter().any(|u| u.client_id == update.client_id) {
            return Err(FlError::DuplicateSubmission {
                client_id: update.client_id,
                round: update.round,
            });
        }

        state.updates.push(update);

        // Check if we have enough clients
        if state.updates.len() >= self.min_clients {
            state.status = RoundStatus::Ready;
        }

        Ok(())
    }

    /// Aggregate updates for a round.
    pub async fn aggregate_round(&self, round: u32) -> Result<AggregationResult, FlError> {
        // Get updates and check status
        let (updates, strategy_name) = {
            let rounds = self.rounds.read();
            let state = rounds.get(&round)
                .ok_or(FlError::RoundNotFound { round })?;
            if state.status == RoundStatus::Completed {
                return Err(FlError::RoundNotCollecting {
                    round,
                    status: state.status.clone(),
                });
            }
            (state.updates.clone(), self.strategy.read().name().to_string())
        };

        if updates.len() < self.min_clients {
            return Err(FlError::InsufficientClients {
                actual: updates.len(),
                required: self.min_clients,
            });
        }

        // Get global model for proximal strategies
        let global_params = self.global_model.read().clone();

        // Aggregate — clone the strategy Arc to avoid holding the lock across await
        let strategy = self.strategy.read().clone();
        let result = strategy.aggregate(
            &updates,
            global_params.as_deref(),
        ).await?;

        // Update state
        {
            let mut rounds = self.rounds.write();
            if let Some(state) = rounds.get_mut(&round) {
                state.status = RoundStatus::Completed;
                state.result = Some(result.clone());
            }
        }

        // Update global model
        {
            let mut model = self.global_model.write();
            *model = Some(result.parameters.clone());
        }

        // Audit log
        if let Some(ref audit) = self.audit_log {
            let mut m = HashMap::new();
            m.insert("round".to_string(), round.to_string());
            m.insert("num_clients".to_string(), result.num_clients.to_string());
            m.insert("total_samples".to_string(), result.total_samples.to_string());
            m.insert("strategy".to_string(), strategy_name.to_string());
            let _ = audit.append(
                &format!("fl_aggregate_{}", strategy_name),
                "fl-engine",
                m,
            ).await;
        }

        info!(
            "FL round {} aggregated: {} clients, {} samples, strategy={}",
            round, result.num_clients, result.total_samples, result.strategy
        );

        Ok(result)
    }

    /// Get the current global model parameters.
    pub fn global_model(&self) -> Option<Vec<f32>> {
        self.global_model.read().clone()
    }

    /// Get the state of a specific round.
    pub fn round_state(&self, round: u32) -> Option<RoundState> {
        self.rounds.read().get(&round).cloned()
    }

    /// Get the current round number.
    pub fn current_round(&self) -> u32 {
        *self.current_round.read()
    }

    /// Get all round states.
    pub fn all_rounds(&self) -> Vec<RoundState> {
        self.rounds.read().values().cloned().collect()
    }

    /// Get the strategy name (owned String to avoid lifetime issues with RwLockReadGuard).
    pub fn strategy_name(&self) -> String {
        self.strategy.read().name().to_string()
    }

    /// Get the number of updates for a specific round.
    pub fn round_update_count(&self, round: u32) -> usize {
        self.rounds.read().get(&round).map(|s| s.updates.len()).unwrap_or(0)
    }

    /// Set the aggregation strategy at runtime.
    pub fn set_strategy(&self, strategy: Arc<dyn AggregationStrategy>) {
        let mut s = self.strategy.write();
        *s = strategy;
    }

    /// Get global model for a specific round (None = latest).
    pub fn get_model(&self, round: Option<u32>) -> Option<Vec<f32>> {
        match round {
            Some(r) => {
                let rounds = self.rounds.read();
                rounds.get(&r).and_then(|s| s.result.as_ref().map(|r| r.parameters.clone()))
            }
            None => self.global_model.read().clone(),
        }
    }

    /// List recent round states.
    pub fn list_rounds(&self, limit: usize) -> Vec<RoundStateSummary> {
        let rounds = self.rounds.read();
        let mut keys: Vec<u32> = rounds.keys().cloned().collect();
        keys.sort_by(|a, b| b.cmp(a)); // newest first
        keys.into_iter().take(limit).filter_map(|k| {
            rounds.get(&k).map(|s| RoundStateSummary {
                round: s.round,
                num_updates: s.updates.len(),
                status: match s.status {
                    RoundStatus::Collecting => "collecting".to_string(),
                    RoundStatus::Ready => "ready".to_string(),
                    RoundStatus::Completed => "completed".to_string(),
                },
                client_ids: s.updates.iter().map(|u| u.client_id.clone()).collect(),
            })
        }).collect()
    }

    /// Get the number of clients in the current or most recent completed round.
    pub fn current_num_clients(&self) -> i32 {
        let rounds = self.rounds.read();
        let current = *self.current_round.read();
        // Try current round first, then walk backwards
        for r in (1..=current).rev() {
            if let Some(state) = rounds.get(&r) {
                if let Some(ref result) = state.result {
                    return result.num_clients as i32;
                }
            }
        }
        0
    }
}

/// Summary of a round state (for gRPC responses).
#[derive(Debug, Clone)]
pub struct RoundStateSummary {
    pub round: u32,
    pub num_updates: usize,
    pub status: String,
    pub client_ids: Vec<String>,
}

// ════════════════════════════════════════════════════════════════
// Differential Privacy
// ════════════════════════════════════════════════════════════════

/// Trait for differential privacy mechanisms.
pub trait DifferentialPrivacy: Send + Sync {
    /// Add noise to parameters in-place.
    fn add_noise(&self, parameters: &mut [f32], epsilon: f64, delta: f64);

    /// Name of the DP mechanism.
    fn name(&self) -> &str;
}

/// Gaussian noise mechanism (DP-SGD style).
pub struct GaussianDP {
    pub clip_norm: f64,
}

impl DifferentialPrivacy for GaussianDP {
    fn add_noise(&self, parameters: &mut [f32], epsilon: f64, delta: f64) {
        // Clip gradient norm
        let norm: f64 = parameters.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
        if norm > self.clip_norm {
            let scale = self.clip_norm / norm;
            for p in parameters.iter_mut() {
                *p = (*p as f64 * scale) as f32;
            }
        }

        // Add Gaussian noise: sigma = clip_norm * sqrt(2 * ln(1.25/delta)) / epsilon
        let sigma = self.clip_norm * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon;
        let mut rng = rand::thread_rng();
        for p in parameters.iter_mut() {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            *p += (z * sigma) as f32;
        }
    }

    fn name(&self) -> &str { "gaussian_dp" }
}

/// Laplace noise mechanism.
pub struct LaplaceDP {
    pub clip_norm: f64,
}

impl DifferentialPrivacy for LaplaceDP {
    fn add_noise(&self, parameters: &mut [f32], epsilon: f64, _delta: f64) {
        // Clip gradient norm
        let norm: f64 = parameters.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
        if norm > self.clip_norm {
            let scale = self.clip_norm / norm;
            for p in parameters.iter_mut() {
                *p = (*p as f64 * scale) as f32;
            }
        }

        // Add Laplace noise: b = clip_norm / epsilon
        let b = self.clip_norm / epsilon;
        let mut rng = rand::thread_rng();
        for p in parameters.iter_mut() {
            let u: f64 = rng.gen();
            let sign = if u < 0.5 { -1.0_f64 } else { 1.0_f64 };
            let laplace = sign * (1.0_f64 - 2.0_f64 * u.abs()).ln().abs() * b;
            *p += laplace as f32;
        }
    }

    fn name(&self) -> &str { "laplace_dp" }
}

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(client_id: &str, round: u32, params: Vec<f32>, samples: usize) -> ModelUpdate {
        ModelUpdate {
            client_id: client_id.to_string(),
            round,
            parameters: params,
            num_samples: samples,
            train_loss: None,
            entropy: None,
            submitted_at: Utc::now().to_rfc3339(),
        }
    }

    #[tokio::test]
    async fn test_fedavg_basic() {
        let strategy = FedAvg;
        let updates = vec![
            make_update("c1", 1, vec![1.0, 2.0, 3.0], 100),
            make_update("c2", 1, vec![3.0, 4.0, 5.0], 300),
        ];
        let result = strategy.aggregate(&updates, None).await.unwrap();
        // Weighted: c1=100/400=0.25, c2=300/400=0.75
        assert_eq!(result.parameters[0], 0.25 * 1.0 + 0.75 * 3.0); // 2.5
        assert_eq!(result.num_clients, 2);
        assert_eq!(result.total_samples, 400);
    }

    #[tokio::test]
    async fn test_fedavg_equal_weights() {
        let strategy = FedAvg;
        let updates = vec![
            make_update("c1", 1, vec![2.0, 4.0], 50),
            make_update("c2", 1, vec![4.0, 8.0], 50),
        ];
        let result = strategy.aggregate(&updates, None).await.unwrap();
        // Equal weights: simple average
        assert!((result.parameters[0] - 3.0).abs() < 0.001);
        assert!((result.parameters[1] - 6.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_ewa_low_entropy_gets_higher_weight() {
        let strategy = EWA::new(2.0);
        let mut u1 = make_update("c1", 1, vec![1.0], 100);
        u1.entropy = Some(0.5); // Low entropy (confident)
        let mut u2 = make_update("c2", 1, vec![3.0], 100);
        u2.entropy = Some(2.0); // High entropy (uncertain)

        let result = strategy.aggregate(&[u1, u2], None).await.unwrap();
        // c1 should have higher weight than c2
        let w1 = result.client_weights.get("c1").unwrap();
        let w2 = result.client_weights.get("c2").unwrap();
        assert!(w1 > w2, "Low-entropy client should have higher weight: {} vs {}", w1, w2);
    }

    #[tokio::test]
    async fn test_fl_engine_full_round() {
        let engine = FlEngine::new(Arc::new(FedAvg), 2);

        let round = engine.start_round();
        assert_eq!(round, 1);

        engine.submit_update(make_update("c1", 1, vec![1.0, 2.0], 100)).unwrap();
        engine.submit_update(make_update("c2", 1, vec![3.0, 4.0], 200)).unwrap();

        let result = engine.aggregate_round(1).await.unwrap();
        assert_eq!(result.num_clients, 2);

        let model = engine.global_model().unwrap();
        assert_eq!(model.len(), 2);
    }

    #[tokio::test]
    async fn test_fl_engine_duplicate_submission() {
        let engine = FlEngine::new(Arc::new(FedAvg), 2);
        let round = engine.start_round();

        engine.submit_update(make_update("c1", round, vec![1.0], 100)).unwrap();
        let result = engine.submit_update(make_update("c1", round, vec![2.0], 100));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fl_engine_insufficient_clients() {
        let engine = FlEngine::new(Arc::new(FedAvg), 3);
        let round = engine.start_round();

        engine.submit_update(make_update("c1", round, vec![1.0], 100)).unwrap();
        engine.submit_update(make_update("c2", round, vec![2.0], 100)).unwrap();

        let result = engine.aggregate_round(round).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fedprox_same_as_fedavg() {
        let fedprox = FedProx { mu: 0.01 };
        let updates = vec![
            make_update("c1", 1, vec![1.0, 2.0], 100),
            make_update("c2", 1, vec![3.0, 4.0], 100),
        ];
        let result = fedprox.aggregate(&updates, None).await.unwrap();
        // Same as FedAvg with equal weights
        assert!((result.parameters[0] - 2.0).abs() < 0.001);
        assert!(result.strategy.contains("fedprox"));
    }

    #[test]
    fn test_gaussian_dp_adds_noise() {
        let dp = GaussianDP { clip_norm: 1.0 };
        let mut params = vec![1.0f32, 2.0f32, 3.0f32];
        let original: Vec<f32> = params.clone();
        dp.add_noise(&mut params, 10.0, 1e-5);
        // Parameters should have changed
        assert_ne!(params, original);
    }

    #[test]
    fn test_laplace_dp_adds_noise() {
        let dp = LaplaceDP { clip_norm: 1.0 };
        let mut params = vec![1.0f32, 2.0f32, 3.0f32];
        let original: Vec<f32> = params.clone();
        dp.add_noise(&mut params, 10.0, 1e-5);
        assert_ne!(params, original);
    }

    #[test]
    fn test_gaussian_dp_clips_norm() {
        let dp = GaussianDP { clip_norm: 1.0 };
        let mut params = vec![100.0f32, 0.0f32, 0.0f32];
        // Norm = 100, should be clipped to 1.0
        dp.add_noise(&mut params, 100.0, 1e-5);
        // After clipping + noise, magnitude should be much smaller than 100
        let norm: f64 = params.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
        assert!(norm < 10.0, "Norm should be clipped: got {}", norm);
    }
}
