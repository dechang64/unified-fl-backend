//! JSON-backed GraphStore implementation
//!
//! In-memory graph using HashMaps. Suitable for Phase 0 prototyping.
//! Will be replaced by GraphLite (ISO GQL) in Phase 2.

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;
use async_trait::async_trait;

use crate::traits::*;

/// In-memory graph store backed by HashMaps.
pub struct JsonGraphStore {
    nodes: RwLock<HashMap<String, GraphNode>>,
    edges: RwLock<HashMap<String, GraphEdge>>,
    /// Adjacency list: node_id → [(edge_id, target_node_id)]
    adjacency: RwLock<HashMap<String, Vec<(String, String)>>>,
    /// Reverse adjacency: node_id → [(edge_id, source_node_id)]
    reverse_adj: RwLock<HashMap<String, Vec<(String, String)>>>,
}

impl JsonGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            edges: RwLock::new(HashMap::new()),
            adjacency: RwLock::new(HashMap::new()),
            reverse_adj: RwLock::new(HashMap::new()),
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
            .map_err(|e| GraphStoreError::QueryError(format!("JSON parse error: {e}")))?;

        for node in data.nodes {
            self.add_node_sync(node);
        }
        for edge in data.edges {
            self.add_edge_sync(edge);
        }
        Ok(())
    }

    fn add_node_sync(&self, node: GraphNode) {
        self.nodes.write().unwrap().insert(node.id.clone(), node);
    }

    fn add_edge_sync(&self, edge: GraphEdge) {
        let edge_id = edge.id.clone();
        let source = edge.source.clone();
        let target = edge.target.clone();

        self.adjacency
            .write()
            .unwrap()
            .entry(source.clone())
            .or_default()
            .push((edge_id.clone(), target.clone()));

        self.reverse_adj
            .write()
            .unwrap()
            .entry(target.clone())
            .or_default()
            .push((edge_id.clone(), source));

        self.edges.write().unwrap().insert(edge_id, edge);
    }

    // ── Persistence ──

    /// Persist graph to a JSON file.
    pub fn persist(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let nodes_guard = self.nodes.read().unwrap();
        let edges_guard = self.edges.read().unwrap();
        let nodes: Vec<&GraphNode> = nodes_guard.values().collect();
        let edges: Vec<&GraphEdge> = edges_guard.values().collect();
        let data = serde_json::json!({
            "nodes": nodes,
            "edges": edges,
        });
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)
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
        self.add_edge_sync(edge);
        Ok(())
    }

    async fn traverse(
        &self,
        start_id: &str,
        edge_label: Option<&str>,
        max_hops: usize,
    ) -> Result<GraphPath, GraphStoreError> {
        let nodes_map = self.nodes.read().unwrap();
        let edges_map = self.edges.read().unwrap();
        let adj = self.adjacency.read().unwrap();

        if !nodes_map.contains_key(start_id) {
            return Err(GraphStoreError::NodeNotFound(start_id.to_string()));
        }

        let mut visited_nodes = vec![];
        let mut visited_edges = vec![];
        let mut visited_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut queue = vec![(start_id.to_string(), 0usize)];
        visited_ids.insert(start_id.to_string());

        while let Some((current_id, hops)) = queue.pop() {
            if let Some(node) = nodes_map.get(&current_id) {
                visited_nodes.push(node.clone());
            }
            if hops >= max_hops {
                continue;
            }

            if let Some(neighbors) = adj.get(&current_id) {
                for (edge_id, target_id) in neighbors {
                    if visited_ids.contains(target_id) {
                        continue;
                    }
                    if let Some(edge) = edges_map.get(edge_id) {
                        // Filter by edge label if specified
                        if let Some(label) = edge_label {
                            if edge.label != label {
                                continue;
                            }
                        }
                        visited_edges.push(edge.clone());
                        visited_ids.insert(target_id.clone());
                        queue.push((target_id.clone(), hops + 1));
                    }
                }
            }
        }

        let edge_count = visited_edges.len();
        Ok(GraphPath {
            nodes: visited_nodes,
            edges: visited_edges,
            total_weight: edge_count as f32,
        })
    }

    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, GraphStoreError> {
        Ok(self.nodes.read().unwrap().get(id).cloned())
    }

    async fn neighbors(
        &self,
        node_id: &str,
        edge_label: Option<&str>,
    ) -> Result<Vec<GraphNode>, GraphStoreError> {
        let nodes_map = self.nodes.read().unwrap();
        let edges_map = self.edges.read().unwrap();
        let adj = self.adjacency.read().unwrap();

        let mut result = vec![];
        if let Some(neighbors) = adj.get(node_id) {
            for (edge_id, target_id) in neighbors {
                if let Some(edge) = edges_map.get(edge_id) {
                    if let Some(label) = edge_label {
                        if edge.label != label {
                            continue;
                        }
                    }
                    if let Some(node) = nodes_map.get(target_id) {
                        result.push(node.clone());
                    }
                }
            }
        }
        Ok(result)
    }

    async fn stats(&self) -> GraphStoreStats {
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();

        let node_labels: Vec<String> = nodes
            .values()
            .map(|n| n.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let edge_labels: Vec<String> = edges
            .values()
            .map(|e| e.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        GraphStoreStats {
            total_nodes: nodes.len(),
            total_edges: edges.len(),
            node_labels,
            edge_labels,
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, label: &str) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            label: label.to_string(),
            properties: HashMap::new(),
        }
    }

    fn make_edge(id: &str, source: &str, target: &str, label: &str) -> GraphEdge {
        GraphEdge {
            id: id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            label: label.to_string(),
            properties: HashMap::new(),
        }
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

        // max_hops=1 should only reach b
        let path = store.traverse("a", None, 1).await.unwrap();
        assert_eq!(path.nodes.len(), 2); // a + b

        // max_hops=2 should reach c
        let path = store.traverse("a", None, 2).await.unwrap();
        assert_eq!(path.nodes.len(), 3); // a + b + c
    }
}
