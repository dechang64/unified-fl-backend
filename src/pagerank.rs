//! PageRank computation on GraphStore
//!
//! Implements classic PageRank (power iteration) and Federated PageRank
//! (FedAvg aggregation of local PageRank scores).
//!
//! Classic PageRank:
//!   PR(v) = (1-d)/N + d * Σ_{u∈in(v)} PR(u) / |out(u)|
//!
//! Federated PageRank (FedPR):
//!   PR_global(v) = Σ_k (n_k / N) * PR_k(v)

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::traits::GraphStore;

// ════════════════════════════════════════════════════════════════
// Data Types
// ════════════════════════════════════════════════════════════════

/// PageRank computation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResult {
    /// Node ID → PageRank score.
    pub scores: HashMap<String, f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
}

/// Configuration for PageRank computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankConfig {
    /// Damping factor (0.85 is classic).
    pub damping: f64,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (L1 norm).
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping: 0.85,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Federated PageRank aggregator.
///
/// Each client computes PageRank on its local subgraph,
/// then submits scores for FedAvg-weighted aggregation.
#[derive(Debug, Clone, Default)]
pub struct FederatedPageRank {
    clients: Vec<(HashMap<String, f64>, usize)>, // (scores, node_count)
    global_scores: HashMap<String, f64>,
}

// ════════════════════════════════════════════════════════════════
// PageRank Computation
// ════════════════════════════════════════════════════════════════

/// Compute PageRank on a GraphStore using power iteration.
///
/// Since GraphStore doesn't expose `find_nodes()`, callers must provide
/// the list of node IDs to include in the computation.
pub async fn compute_pagerank(
    graph: &Arc<dyn GraphStore>,
    node_ids: &[String],
    config: &PageRankConfig,
) -> PageRankResult {
    let n = node_ids.len();

    if n == 0 {
        return PageRankResult {
            scores: HashMap::new(),
            iterations: 0,
            converged: true,
        };
    }

    // Build adjacency: out_degree and in_edges
    let mut out_degree: HashMap<String, usize> = HashMap::new();
    let mut in_edges: HashMap<String, Vec<String>> = HashMap::new();

    for node_id in node_ids {
        out_degree.insert(node_id.clone(), 0);
        in_edges.insert(node_id.clone(), Vec::new());

        // Get outgoing neighbors (edge_label = None → all edges)
        if let Ok(neighbors) = graph.neighbors(node_id, None).await {
            let degree = neighbors.len();
            out_degree.insert(node_id.clone(), degree);
            for neighbor in neighbors {
                if let Some(in_list) = in_edges.get_mut(&neighbor.id) {
                    in_list.push(node_id.clone());
                }
            }
        }
    }

    // Power iteration
    let init_score = 1.0 / (n as f64);
    let mut scores: HashMap<String, f64> = node_ids.iter().map(|id| (id.clone(), init_score)).collect();

    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..config.max_iterations {
        iterations += 1;
        let mut new_scores: HashMap<String, f64> = HashMap::new();

        for node_id in node_ids {
            let incoming_sum: f64 = in_edges[node_id]
                .iter()
                .map(|src| {
                    let src_out = out_degree[src].max(1) as f64;
                    scores[src] / src_out
                })
                .sum();

            new_scores.insert(
                node_id.clone(),
                (1.0 - config.damping) / (n as f64) + config.damping * incoming_sum,
            );
        }

        // Check convergence (L1 norm)
        let l1: f64 = node_ids
            .iter()
            .map(|id| (new_scores[id] - scores[id]).abs())
            .sum();

        scores = new_scores;
        if l1 < config.tolerance {
            converged = true;
            break;
        }
    }

    info!("Computed PageRank for {} nodes in {} iterations (converged={})", n, iterations, converged);

    PageRankResult {
        scores,
        iterations,
        converged,
    }
}

// ════════════════════════════════════════════════════════════════
// Federated PageRank
// ════════════════════════════════════════════════════════════════

impl FederatedPageRank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a client's local PageRank scores.
    pub fn add_client(&mut self, scores: HashMap<String, f64>, node_count: usize) {
        self.clients.push((scores, node_count));
    }

    /// Aggregate all client scores using FedAvg (weighted by node count).
    pub fn aggregate(&mut self) -> &HashMap<String, f64> {
        if self.clients.is_empty() {
            self.global_scores.clear();
            return &self.global_scores;
        }

        let total_nodes: usize = self.clients.iter().map(|(_, nc)| *nc).sum();
        if total_nodes == 0 {
            self.global_scores.clear();
            return &self.global_scores;
        }

        let mut global: HashMap<String, f64> = HashMap::new();
        for (scores, node_count) in &self.clients {
            let weight = *node_count as f64 / total_nodes as f64;
            for (node_id, score) in scores {
                *global.entry(node_id.clone()).or_insert(0.0) += weight * score;
            }
        }

        self.global_scores = global;
        info!(
            "Federated PageRank aggregated: {} nodes from {} clients",
            self.global_scores.len(),
            self.clients.len()
        );
        &self.global_scores
    }

    /// Get the PageRank score for a node.
    pub fn get_score(&self, node_id: &str) -> f64 {
        self.global_scores.get(node_id).copied().unwrap_or(0.0)
    }

    /// Top-k nodes by global PageRank score.
    pub fn top_k(&self, k: usize) -> Vec<(String, f64)> {
        let mut items: Vec<_> = self.global_scores.iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        items.into_iter().take(k).map(|(id, s)| (id.clone(), *s)).collect()
    }

    /// Number of clients that have submitted scores.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_pagerank_two_clients() {
        let mut fed_pr = FederatedPageRank::new();

        let mut scores_a = HashMap::new();
        scores_a.insert("CN001".into(), 0.3);
        scores_a.insert("CN003".into(), 0.7);
        fed_pr.add_client(scores_a, 2);

        let mut scores_b = HashMap::new();
        scores_b.insert("CN002".into(), 0.3);
        scores_b.insert("CN003".into(), 0.7);
        fed_pr.add_client(scores_b, 2);

        let global = fed_pr.aggregate();
        assert!(global.get("CN003").unwrap() > global.get("CN001").unwrap());
        assert!(global.get("CN003").unwrap() > global.get("CN002").unwrap());
    }

    #[test]
    fn test_federated_pagerank_empty() {
        let mut fed_pr = FederatedPageRank::new();
        let global = fed_pr.aggregate();
        assert!(global.is_empty());
    }

    #[test]
    fn test_federated_pagerank_top_k() {
        let mut fed_pr = FederatedPageRank::new();
        let mut scores = HashMap::new();
        scores.insert("A".into(), 0.5);
        scores.insert("B".into(), 0.3);
        scores.insert("C".into(), 0.2);
        fed_pr.add_client(scores, 3);
        fed_pr.aggregate();

        let top2 = fed_pr.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, "A");
        assert_eq!(top2[1].0, "B");
    }
}
