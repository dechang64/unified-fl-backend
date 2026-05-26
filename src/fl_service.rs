//! FL gRPC Service Implementation
//!
//! Implements the FederatedLearning service defined in federated_ai.proto.

use std::sync::Arc;
use tonic::{Request, Response, Status};

use crate::proto::{
    federated_learning_server::FederatedLearning,
    SubmitUpdateRequest, SubmitUpdateResponse,
    AggregateRequest, AggregateResponse,
    GetModelRequest, GetModelResponse,
    ListRoundsRequest, ListRoundsResponse,
    FlRoundState,
};
use crate::fl_engine::{FlEngine, ModelUpdate};

pub struct FlServiceImpl {
    engine: Arc<FlEngine>,
}

impl FlServiceImpl {
    pub fn new(engine: Arc<FlEngine>) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl FederatedLearning for FlServiceImpl {
    async fn submit_update(
        &self,
        req: Request<SubmitUpdateRequest>,
    ) -> Result<Response<SubmitUpdateResponse>, Status> {
        let req = req.into_inner();
        let update = req.update.ok_or_else(|| {
            Status::invalid_argument("update field is required")
        })?;

        let model_update = ModelUpdate {
            client_id: update.client_id,
            round: update.round,
            parameters: update.parameters,
            num_samples: update.num_samples as usize,
            train_loss: if update.train_loss != 0.0 { Some(update.train_loss) } else { None },
            entropy: if update.entropy != 0.0 { Some(update.entropy) } else { None },
            submitted_at: chrono::Utc::now().to_rfc3339(),
        };

        let current_round = self.engine.current_round();
        let updates_count = self.engine.round_update_count(model_update.round);

        match self.engine.submit_update(model_update) {
            Ok(()) => Ok(Response::new(SubmitUpdateResponse {
                accepted: true,
                current_round,
                updates_received: updates_count as i32 + 1,
            })),
            Err(e) => Err(Status::invalid_argument(format!("Submit failed: {}", e))),
        }
    }

    async fn aggregate(
        &self,
        req: Request<AggregateRequest>,
    ) -> Result<Response<AggregateResponse>, Status> {
        let req = req.into_inner();
        let round = if req.round > 0 { req.round } else { self.engine.current_round() };

        // Configure strategy if specified
        if !req.strategy.is_empty() {
            let strategy: Arc<dyn crate::fl_engine::AggregationStrategy> = match req.strategy.as_str() {
                "fedavg" => Arc::new(crate::fl_engine::FedAvg),
                "fedprox" => Arc::new(crate::fl_engine::FedProx::new(req.fedprox_mu)),
                "ewa" => Arc::new(crate::fl_engine::EWA::new(req.ewa_alpha)),
                _ => return Err(Status::invalid_argument(format!("Unknown strategy: {}", req.strategy))),
            };
            self.engine.set_strategy(strategy);
        }

        let result = self.engine.aggregate_round(round).await
            .map_err(|e| Status::internal(format!("Aggregation failed: {}", e)))?;

        Ok(Response::new(AggregateResponse {
            global_parameters: result.parameters,
            round: result.round,
            num_clients: result.num_clients as i32,
            total_samples: result.total_samples as i64,
            strategy: result.strategy,
            client_weights: result.client_weights,
            aggregation_time_ms: 0.0,
        }))
    }

    async fn get_model(
        &self,
        req: Request<GetModelRequest>,
    ) -> Result<Response<GetModelResponse>, Status> {
        let req = req.into_inner();
        let round = if req.round > 0 { Some(req.round) } else { None };

        match self.engine.get_model(round) {
            Some(params) => {
                let current_round = self.engine.current_round();
                let num_clients = self.engine.current_num_clients();
                let strategy = self.engine.strategy_name();
                Ok(Response::new(GetModelResponse {
                    parameters: params,
                    round: current_round,
                    num_clients,
                    strategy,
                }))
            }
            None => Err(Status::not_found("No global model available")),
        }
    }

    async fn list_rounds(
        &self,
        req: Request<ListRoundsRequest>,
    ) -> Result<Response<ListRoundsResponse>, Status> {
        let req = req.into_inner();
        let limit = if req.limit > 0 { req.limit as usize } else { 10 };
        let rounds = self.engine.list_rounds(limit);

        Ok(Response::new(ListRoundsResponse {
            rounds: rounds.into_iter().map(|s| FlRoundState {
                round: s.round,
                num_updates: s.num_updates as i32,
                status: s.status,
                client_ids: s.client_ids,
            }).collect(),
        }))
    }
}
