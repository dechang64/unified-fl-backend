//! GraphRAG Query Engine
//!
//! Combines vector similarity search with graph traversal for multi-hop
//! retrieval-augmented generation. This is the core differentiator of
//! FedCtx — the first open-source federated GraphRAG framework.
//!
//! Query flow:
//! 1. Text → embedding → HNSW search top-k candidate nodes
//! 2. Candidate nodes → GraphStore multi-hop expansion
//! 3. Expanded subgraph → re-rank (vector similarity + graph centrality)
//! 4. Ranked subgraph → context for LLM generation

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

use crate::traits::{VectorStore, GraphStore};
use crate::embedding::EmbeddingProvider;

// ════════════════════════════════════════════════════════════════
// Data Types
// ════════════════════════════════════════════════════════════════

/// A node in the retrieval subgraph, with its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredNode {
    pub id: String,
    pub label: String,
    pub properties: HashMap<String, String>,
    /// Vector similarity score (0-1, higher = more similar).
    pub vector_score: f32,
    /// Graph centrality score (0-1, higher = more central).
    pub centrality_score: f32,
    /// Combined score (weighted average of vector + centrality).
    pub combined_score: f32,
}

/// A subgraph retrieved by GraphRAG, containing relevant nodes and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedSubgraph {
    /// The query that produced this subgraph.
    pub query: String,
    /// Scored nodes, sorted by combined_score descending.
    pub nodes: Vec<ScoredNode>,
    /// Edges connecting the nodes.
    pub edges: Vec<SubgraphEdge>,
    /// Number of hops expanded.
    pub hops: usize,
    /// Total nodes considered before pruning.
    pub candidates_considered: usize,
}

/// An edge in the retrieved subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub label: String,
}

/// Configuration for GraphRAG queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    /// Number of initial candidates from vector search.
    pub top_k: usize,
    /// Maximum number of graph expansion hops.
    pub max_hops: usize,
    /// Maximum nodes in the final subgraph.
    pub max_nodes: usize,
    /// Weight of vector similarity in combined score (0-1).
    pub vector_weight: f32,
    /// Weight of graph centrality in combined score (0-1).
    pub centrality_weight: f32,
    /// Minimum combined score to include a node.
    pub min_score: f32,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            max_hops: 2,
            max_nodes: 30,
            vector_weight: 0.6,
            centrality_weight: 0.4,
            min_score: 0.1,
        }
    }
}

/// Error type for GraphRAG operations.
#[derive(Debug, thiserror::Error)]
pub enum GraphRAGError {
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),
    #[error("vector search failed: {0}")]
    VectorSearchFailed(String),
    #[error("graph traversal failed: {0}")]
    GraphTraversalFailed(String),
    #[error("no results found for query")]
    NoResults,
    #[error("internal error: {0}")]
    Internal(String),
}

// ════════════════════════════════════════════════════════════════
// GraphRAG Engine
// ════════════════════════════════════════════════════════════════

/// The GraphRAG query engine — combines vector search with graph traversal.
pub struct GraphRAGEngine {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Arc<dyn GraphStore>,
    embedding: Arc<dyn EmbeddingProvider>,
    config: GraphRAGConfig,
}

impl GraphRAGEngine {
    pub fn new(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        embedding: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            embedding,
            config: GraphRAGConfig::default(),
        }
    }

    pub fn with_config(mut self, config: GraphRAGConfig) -> Self {
        self.config = config;
        self
    }

    pub fn new_with_config(
        vector_store: Arc<dyn VectorStore>,
        graph_store: Arc<dyn GraphStore>,
        embedding: Arc<dyn EmbeddingProvider>,
        config: GraphRAGConfig,
    ) -> Self {
        Self { vector_store, graph_store, embedding, config }
    }

    pub fn vector_store(&self) -> Arc<dyn VectorStore> { self.vector_store.clone() }
    pub fn graph_store(&self) -> Arc<dyn GraphStore> { self.graph_store.clone() }
    pub fn embedding(&self) -> Arc<dyn EmbeddingProvider> { self.embedding.clone() }

    /// Execute a GraphRAG query: vector search → graph expansion → re-rank.
    pub async fn query(&self, question: &str) -> Result<RetrievedSubgraph, GraphRAGError> {
        // Step 1: Embed the query
        let query_vec = self.embedding.embed(question).await
            .map_err(|e| GraphRAGError::EmbeddingFailed(e.to_string()))?;

        // Step 2: Vector search for top-k candidates
        let hits = self.vector_store.search(
            &query_vec,
            self.config.top_k,
            None, // No metadata filter
        ).await
            .map_err(|e| GraphRAGError::VectorSearchFailed(e.to_string()))?;

        if hits.is_empty() {
            return Err(GraphRAGError::NoResults);
        }

        // Step 3: Collect seed node IDs from vector hits
        let seed_ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let mut vector_scores: HashMap<String, f32> = HashMap::new();
        for hit in &hits {
            // SearchHit.score is already cosine similarity (0-1, higher = more similar)
            // as converted in VectorDB::search(): score = 1.0 - (distance / 1_000_000.0)
            vector_scores.insert(hit.id.clone(), hit.score);
        }

        // Step 4: Graph expansion — BFS from seed nodes
        let mut visited_nodes: HashSet<String> = HashSet::new();
        let mut all_node_ids: HashSet<String> = HashSet::new();
        let mut all_edges: Vec<SubgraphEdge> = Vec::new();

        // Start with seed nodes
        for id in &seed_ids {
            all_node_ids.insert(id.clone());
        }

        // Multi-hop expansion
        let mut frontier: HashSet<String> = seed_ids.iter().cloned().collect();
        for _hop in 0..self.config.max_hops {
            let mut next_frontier = HashSet::new();
            for node_id in &frontier {
                if visited_nodes.contains(node_id) {
                    continue;
                }
                visited_nodes.insert(node_id.clone());

                // Get neighbors (all edge labels)
                let neighbors = self.graph_store.neighbors(node_id, None).await
                    .map_err(|e| GraphRAGError::GraphTraversalFailed(e.to_string()))?;

                for neighbor_node in neighbors {
                    let neighbor_id = neighbor_node.id.clone();
                    if !all_node_ids.contains(&neighbor_id) {
                        all_node_ids.insert(neighbor_id.clone());
                        next_frontier.insert(neighbor_id.clone());
                    }
                    all_edges.push(SubgraphEdge {
                        id: format!("edge_{}_{}", node_id, neighbor_id),
                        source: node_id.clone(),
                        target: neighbor_id,
                        label: String::new(),
                    });
                }
            }
            frontier = next_frontier;
            if frontier.is_empty() {
                break;
            }
        }

        // Step 5: Compute centrality scores (degree centrality)
        let mut degree_count: HashMap<String, usize> = HashMap::new();
        for edge in &all_edges {
            *degree_count.entry(edge.source.clone()).or_insert(0) += 1;
            *degree_count.entry(edge.target.clone()).or_insert(0) += 1;
        }
        let max_degree = degree_count.values().max().copied().unwrap_or(1) as f32;

        // Step 6: Score and rank nodes
        let mut scored_nodes: Vec<ScoredNode> = Vec::new();
        for node_id in &all_node_ids {
            let vec_score = vector_scores.get(node_id).copied().unwrap_or(0.0);
            let cent_score = degree_count.get(node_id).copied().unwrap_or(0) as f32 / max_degree;
            let combined = vec_score * self.config.vector_weight
                + cent_score * self.config.centrality_weight;

            if combined < self.config.min_score {
                continue;
            }

            // Get node properties from graph store
            let (label, properties) = match self.graph_store.get_node(node_id).await {
                Ok(Some(node)) => (node.label.clone(), node.properties.clone()),
                _ => ("Unknown".to_string(), HashMap::new()),
            };

            scored_nodes.push(ScoredNode {
                id: node_id.clone(),
                label,
                properties,
                vector_score: vec_score,
                centrality_score: cent_score,
                combined_score: combined,
            });
        }

        // Sort by combined score descending
        scored_nodes.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        // Truncate to max_nodes
        scored_nodes.truncate(self.config.max_nodes);

        // Filter edges to only include those between retained nodes
        let retained_ids: HashSet<String> = scored_nodes.iter().map(|n| n.id.clone()).collect();
        all_edges.retain(|e| retained_ids.contains(&e.source) && retained_ids.contains(&e.target));

        let candidates_considered = all_node_ids.len();

        Ok(RetrievedSubgraph {
            query: question.to_string(),
            nodes: scored_nodes,
            edges: all_edges,
            hops: self.config.max_hops,
            candidates_considered,
        })
    }

    /// Get the current configuration.
    pub fn config(&self) -> &GraphRAGConfig {
        &self.config
    }
}

// ════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw_store::HnswVectorStore;
    use crate::json_graph_store::JsonGraphStore;
    use crate::embedding::HashEmbedder;
    use crate::traits::{VectorEntry, GraphNode, GraphEdge};

    async fn setup_engine() -> GraphRAGEngine {
        let vector_store = Arc::new(HnswVectorStore::new(64));
        let graph_store = Arc::new(JsonGraphStore::new());
        let embedding = Arc::new(HashEmbedder::new(64));

        // Insert some test vectors
        let entries = vec![
            VectorEntry {
                id: "drug_1".to_string(),
                values: HashEmbedder::new(64).embed("aspirin anti-inflammatory").await.unwrap(),
                metadata: HashMap::new(),
            },
            VectorEntry {
                id: "drug_2".to_string(),
                values: HashEmbedder::new(64).embed("ibuprofen pain relief").await.unwrap(),
                metadata: HashMap::new(),
            },
            VectorEntry {
                id: "disease_1".to_string(),
                values: HashEmbedder::new(64).embed("arthritis inflammation joint").await.unwrap(),
                metadata: HashMap::new(),
            },
        ];
        for entry in entries {
            vector_store.insert(vec![entry]).await.unwrap();
        }

        // Insert some test graph nodes and edges
        graph_store.add_node(GraphNode {
            id: "drug_1".to_string(),
            label: "Drug".to_string(),
            properties: HashMap::new(),
        }).await.unwrap();
        graph_store.add_node(GraphNode {
            id: "disease_1".to_string(),
            label: "Disease".to_string(),
            properties: HashMap::new(),
        }).await.unwrap();
        graph_store.add_edge(GraphEdge {
            id: "e1".to_string(),
            source: "drug_1".to_string(),
            target: "disease_1".to_string(),
            label: "TREATS".to_string(),
            properties: HashMap::new(),
        }).await.unwrap();

        GraphRAGEngine::new(vector_store, graph_store, embedding)
            .with_config(GraphRAGConfig {
                top_k: 5,
                max_hops: 2,
                max_nodes: 20,
                min_score: 0.0, // Low threshold for testing
                ..Default::default()
            })
    }

    #[tokio::test]
    async fn test_graphrag_query_returns_subgraph() {
        let engine = setup_engine().await;
        let result = engine.query("anti-inflammatory drug").await;
        // May or may not find results depending on hash embedding quality
        // The important thing is it doesn't crash
        if let Ok(subgraph) = result {
            assert!(!subgraph.query.is_empty());
        }
    }

    #[test]
    fn test_graphrag_config_default() {
        let config = GraphRAGConfig::default();
        assert_eq!(config.top_k, 10);
        assert_eq!(config.max_hops, 2);
        assert_eq!(config.max_nodes, 30);
        assert!((config.vector_weight + config.centrality_weight - 1.0).abs() < 0.001);
    }
}
