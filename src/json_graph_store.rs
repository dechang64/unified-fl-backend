//! JSON-backed GraphStore implementation
//!
//! In-memory graph using HashMaps with parking_lot::RwLock for high concurrency.
//! Non-poisoning: if a thread panics while holding a lock, the lock is not poisoned.

use std::collections::HashMap;
use std::path::Path;
use async_trait::async_trait;
use parking_lot::RwLock;

use crate::traits::*;

/// In-memory graph store backed by HashMaps.
pub struct JsonGraphStore {
    nodes: RwLock<HashMap<String, GraphNode>>,
    edges: RwLock<HashMap<String, GraphEdge>>,
    /// Adjacency list: node_id → [(edge_id, target_node_id)]
    adjacency: RwLock<HashMap<String, Vec<(String, String)>>>,
    /// Reverse adjacency: node_id → [(edge_id, source_node_id)]
    reverse_adj: RwLock<HashMap<String, Vec<(String, String)>>>,
    /// Label index for fast label-based queries
    label_index: RwLock<HashMap<String, Vec<String>>>,
}

impl Default for JsonGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            edges: RwLock::new(HashMap::new()),
            adjacency: RwLock::new(HashMap::new()),
            reverse_adj: RwLock::new(HashMap::new()),
            label_index: RwLock::new(HashMap::new()),
        }
    }

    /// Load graph from a JSON string (array of nodes and edges).
    pub fn load_from_json(&self, json: &str) -> Result<(), GraphStoreError> {
        #[derive(serde::Deserialize)]
        struct GraphData {
            nodes: Vec<GraphNode>,
            edges: Vec<GraphEdge>,
        }
        let data: GraphData = serde_json::from_str(json)
            .map_err(|e| GraphStoreError::Internal(format!("JSON parse error: {e}")))?;

        for node in data.nodes {
            self.add_node_sync(node);
        }
        for edge in data.edges {
            self.add_edge_sync(edge);
        }
        Ok(())
    }

    fn add_node_sync(&self, node: GraphNode) {
        let label = node.label.clone();
        let id = node.id.clone();
        self.nodes.write().insert(id.clone(), node);
        self.label_index.write().entry(label).or_default().push(id);
    }

    fn add_edge_sync(&self, edge: GraphEdge) {
        let edge_id = edge.id.clone();
        let source = edge.source.clone();
        let target = edge.target.clone();
        self.edges.write().insert(edge_id.clone(), edge);
        self.adjacency.write()
            .entry(source.clone())
            .or_default()
            .push((edge_id.clone(), target.clone()));
        self.reverse_adj.write()
            .entry(target)
            .or_default()
            .push((edge_id, source));
    }

    /// Persist graph to a JSON file.
    pub fn persist(&self, path: &Path) -> Result<(), std::io::Error> {
        let nodes = self.nodes.read();
        let edges = self.edges.read();
        let data = serde_json::json!({
            "nodes": nodes.values().collect::<Vec<_>>(),
            "edges": edges.values().collect::<Vec<_>>(),
        });
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&data)?)
    }

    /// Restore graph from a JSON file.
    pub fn restore(&self, path: &Path) -> Result<(), std::io::Error> {
        if !path.exists() {
            return Ok(());
        }
        let json = std::fs::read_to_string(path)?;
        self.load_from_json(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

#[async_trait]
impl GraphStore for JsonGraphStore {
    async fn add_node(&self, node: GraphNode) -> Result<(), GraphStoreError> {
        self.add_node_sync(node);
        Ok(())
    }

    async fn add_edge(&self, edge: GraphEdge) -> Result<(), GraphStoreError> {
        // Validate source and target exist
        {
            let nodes = self.nodes.read();
            if !nodes.contains_key(&edge.source) {
                return Err(GraphStoreError::NodeNotFound(edge.source.clone()));
            }
            if !nodes.contains_key(&edge.target) {
                return Err(GraphStoreError::NodeNotFound(edge.target.clone()));
            }
        }
        self.add_edge_sync(edge);
        Ok(())
    }

    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, GraphStoreError> {
        Ok(self.nodes.read().get(id).cloned())
    }

    async fn neighbors(&self, id: &str, edge_label: Option<&str>) -> Result<Vec<GraphNode>, GraphStoreError> {
        let adj = self.adjacency.read();
        let edges = self.edges.read();
        let nodes = self.nodes.read();

        let Some(adj_list) = adj.get(id) else {
            return Ok(vec![]);
        };

        let mut result = Vec::new();
        for (edge_id, target_id) in adj_list {
            if let Some(label) = edge_label {
                if let Some(edge) = edges.get(edge_id) {
                    if edge.label != label {
                        continue;
                    }
                }
            }
            if let Some(node) = nodes.get(target_id) {
                result.push(node.clone());
            }
        }
        Ok(result)
    }

    async fn traverse(&self, start_id: &str, edge_label: Option<&str>, max_hops: usize) -> Result<GraphPath, GraphStoreError> {
        let mut visited_nodes = vec![];
        let mut visited_edges = vec![];

        {
            let nodes = self.nodes.read();
            let edges = self.edges.read();
            let adj = self.adjacency.read();

            let start_node = nodes.get(start_id)
                .ok_or_else(|| GraphStoreError::NodeNotFound(start_id.to_string()))?
                .clone();
            visited_nodes.push(start_node);

            let mut frontier = vec![start_id.to_string()];
            let mut seen = std::collections::HashSet::new();
            seen.insert(start_id.to_string());

            for _hop in 0..max_hops {
                let mut next_frontier = vec![];
                for node_id in &frontier {
                    if let Some(adj_list) = adj.get(node_id) {
                        for (edge_id, target_id) in adj_list {
                            if seen.contains(target_id) {
                                continue;
                            }
                            seen.insert(target_id.clone());

                            if let Some(edge) = edges.get(edge_id) {
                                if let Some(label) = edge_label {
                                    if edge.label != label {
                                        continue;
                                    }
                                }
                                visited_edges.push(edge.clone());
                            }
                            if let Some(node) = nodes.get(target_id) {
                                visited_nodes.push(node.clone());
                                next_frontier.push(target_id.clone());
                            }
                        }
                    }
                }
                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }
        }

        Ok(GraphPath {
            nodes: visited_nodes,
            edges: visited_edges,
            total_weight: 1.0, // Unweighted graph, each edge has weight 1.0
        })
    }

    async fn stats(&self) -> GraphStoreStats {
        self.stats_blocking()
    }
}

impl JsonGraphStore {
    /// Synchronous stats access (for non-async contexts like REST handlers).
    pub fn stats_blocking(&self) -> GraphStoreStats {
        let nodes = self.nodes.read();
        let edges = self.edges.read();
        let _labels = self.label_index.read();

        let mut node_labels = std::collections::HashSet::new();
        let mut edge_labels = std::collections::HashSet::new();
        for node in nodes.values() {
            node_labels.insert(node.label.clone());
        }
        for edge in edges.values() {
            edge_labels.insert(edge.label.clone());
        }

        GraphStoreStats {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            node_labels: node_labels.into_iter().collect(),
            edge_labels: edge_labels.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, label: &str) -> GraphNode {
        GraphNode { id: id.to_string(), label: label.to_string(), properties: HashMap::new() }
    }

    fn make_edge(id: &str, source: &str, target: &str, label: &str) -> GraphEdge {
        GraphEdge { id: id.to_string(), source: source.to_string(), target: target.to_string(), label: label.to_string(), properties: HashMap::new() }
    }

    #[tokio::test]
    async fn test_add_and_get_node() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("fabry", "Disease")).await.unwrap();
        let node = store.get_node("fabry").await.unwrap();
        assert!(node.is_some());
        assert_eq!(node.unwrap().label, "Disease");
    }

    #[tokio::test]
    async fn test_traverse() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("fabry", "Disease")).await.unwrap();
        store.add_node(make_node("mayo", "Hospital")).await.unwrap();
        store.add_node(make_node("nih", "Hospital")).await.unwrap();
        store.add_edge(make_edge("e1", "fabry", "mayo", "SPECIALIZES_IN")).await.unwrap();
        store.add_edge(make_edge("e2", "fabry", "nih", "SPECIALIZES_IN")).await.unwrap();

        let path = store.traverse("fabry", Some("SPECIALIZES_IN"), 2).await.unwrap();
        assert_eq!(path.nodes.len(), 3); // fabry + mayo + nih
        assert_eq!(path.edges.len(), 2);
    }

    #[tokio::test]
    async fn test_neighbors() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("fabry", "Disease")).await.unwrap();
        store.add_node(make_node("mayo", "Hospital")).await.unwrap();
        store.add_edge(make_edge("e1", "fabry", "mayo", "SPECIALIZES_IN")).await.unwrap();

        let neighbors = store.neighbors("fabry", Some("SPECIALIZES_IN")).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, "mayo");
    }

    #[tokio::test]
    async fn test_stats() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("fabry", "Disease")).await.unwrap();
        store.add_node(make_node("mayo", "Hospital")).await.unwrap();
        store.add_edge(make_edge("e1", "fabry", "mayo", "SPECIALIZES_IN")).await.unwrap();

        let stats = store.stats().await;
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
        assert!(stats.node_labels.contains(&"Disease".to_string()));
        assert!(stats.node_labels.contains(&"Hospital".to_string()));
    }

    #[tokio::test]
    async fn test_traverse_max_hops() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("a", "A")).await.unwrap();
        store.add_node(make_node("b", "B")).await.unwrap();
        store.add_node(make_node("c", "C")).await.unwrap();
        store.add_edge(make_edge("e1", "a", "b", "LINKS")).await.unwrap();
        store.add_edge(make_edge("e2", "b", "c", "LINKS")).await.unwrap();

        let path = store.traverse("a", None, 1).await.unwrap();
        assert_eq!(path.nodes.len(), 2); // a + b

        let path = store.traverse("a", None, 2).await.unwrap();
        assert_eq!(path.nodes.len(), 3); // a + b + c
    }

    #[tokio::test]
    async fn test_edge_to_missing_node_fails() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("a", "A")).await.unwrap();
        // Edge to non-existent node should fail
        let result = store.add_edge(make_edge("e1", "a", "missing", "LINKS")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_label_index() {
        let store = JsonGraphStore::new();
        store.add_node(make_node("d1", "Disease")).await.unwrap();
        store.add_node(make_node("d2", "Disease")).await.unwrap();
        store.add_node(make_node("h1", "Hospital")).await.unwrap();

        let stats = store.stats().await;
        assert!(stats.node_labels.contains(&"Disease".to_string()));
        assert!(stats.node_labels.contains(&"Hospital".to_string()));
    }
}
