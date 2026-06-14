//! Knowledge Graph Builder — Extract entities and relations from PageIndex trees and patent data
//!
//! Extraction strategies (MVP):
//!   1. Structural: PageIndex tree parent/child → CONTAINS edges
//!   2. Entity: Rule-based NER (Chinese/English patterns)
//!   3. Co-occurrence: Entities in same section → RELATES_TO edges
//!   4. Patent: Citations → REFERENCES, IPC → BELONGS_TO
//!
//! All nodes/edges are written to GraphStore (JsonGraphStore or any implementation).

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::traits::{GraphStore, GraphNode, GraphEdge, GraphStoreError};
use crate::pageindex::PageIndexNode;

// ════════════════════════════════════════════════════════════════
// Data Types
// ════════════════════════════════════════════════════════════════

/// Node type labels in the KG.
pub mod node_type {
    pub const DOCUMENT: &str = "Document";
    pub const SECTION: &str = "Section";
    pub const ENTITY: &str = "Entity";
    pub const CONCEPT: &str = "Concept";
    pub const PATENT: &str = "Patent";
    pub const TECHNOLOGY: &str = "Technology";
    pub const IPC: &str = "IPC";
}

/// Edge type labels in the KG.
pub mod edge_type {
    pub const CONTAINS: &str = "contains";
    pub const REFERENCES: &str = "references";
    pub const RELATES_TO: &str = "relates_to";
    pub const MENTIONS: &str = "mentions";
    pub const BELONGS_TO: &str = "belongs_to";
    pub const DEPENDS_ON: &str = "depends_on";
}

/// A patent record for KG construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatentRecord {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub abstract_text: String,
    #[serde(default)]
    pub cited_patents: Vec<String>,
    #[serde(default)]
    pub ipc_codes: Vec<String>,
}

/// Statistics from KG building.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KGBuildStats {
    pub nodes_added: usize,
    pub edges_added: usize,
    pub entities_extracted: usize,
    pub cooccurrence_edges: usize,
}

// ════════════════════════════════════════════════════════════════
// Entity Extraction (Rule-based NER)
// ════════════════════════════════════════════════════════════════

/// Technology keywords for entity extraction.
const TECH_KEYWORDS: &[&str] = &[
    "federated learning", "differential privacy", "machine learning",
    "deep learning", "neural network", "reinforcement learning",
    "knowledge graph", "natural language processing", "computer vision",
    "transfer learning", "attention mechanism", "transformer",
    "convolutional", "recurrent", "generative", "discriminative",
    "encryption", "blockchain", "secure aggregation",
    "联邦学习", "差分隐私", "机器学习", "深度学习", "神经网络",
    "强化学习", "知识图谱", "自然语言处理", "计算机视觉",
    "迁移学习", "注意力机制", "安全聚合", "加密",
    "联邦训练", "模型聚合", "梯度下降", "反向传播",
];

/// Extract technology entities from text.
fn extract_tech_entities(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut entities = Vec::new();
    for keyword in TECH_KEYWORDS {
        if lower.contains(keyword) {
            entities.push(keyword.to_string());
        }
    }
    entities
}

// ════════════════════════════════════════════════════════════════
// KG Builder
// ════════════════════════════════════════════════════════════════

/// Knowledge Graph Builder.
///
/// Builds a KG by extracting entities and relations from documents,
/// PageIndex trees, and patent data, writing to a GraphStore.
pub struct KGBuilder {
    graph: Arc<dyn GraphStore>,
    /// Track entities per source for co-occurrence.
    source_entities: HashMap<String, Vec<String>>,
    /// Counter for generating unique edge IDs.
    edge_counter: usize,
}

impl KGBuilder {
    pub fn new(graph: Arc<dyn GraphStore>) -> Self {
        Self {
            graph,
            source_entities: HashMap::new(),
            edge_counter: 0,
        }
    }

    fn next_edge_id(&mut self) -> String {
        self.edge_counter += 1;
        format!("edge_{}", self.edge_counter)
    }

    /// Build KG from a PageIndex tree structure.
    pub async fn build_from_pageindex_tree(
        &mut self,
        nodes: &[PageIndexNode],
        doc_id: &str,
        doc_name: &str,
    ) -> Result<KGBuildStats, GraphStoreError> {
        let mut stats = KGBuildStats {
            nodes_added: 0,
            edges_added: 0,
            entities_extracted: 0,
            cooccurrence_edges: 0,
        };

        // Add document node
        let doc_node = GraphNode {
            id: doc_id.to_string(),
            label: node_type::DOCUMENT.to_string(),
            properties: {
                let mut p = HashMap::new();
                p.insert("name".into(), doc_name.to_string());
                p
            },
        };
        self.graph.add_node(doc_node).await?;
        stats.nodes_added += 1;

        // Process tree nodes recursively
        self.process_tree_nodes(nodes, doc_id, &mut stats).await?;

        // Build co-occurrence edges
        stats.cooccurrence_edges = self.build_cooccurrence_edges().await?;
        stats.edges_added += stats.cooccurrence_edges;

        info!(
            "Built KG from PageIndex tree: {} nodes, {} edges, {} entities",
            stats.nodes_added, stats.edges_added, stats.entities_extracted
        );

        Ok(stats)
    }

    fn process_tree_nodes<'a>(
        &'a mut self,
        nodes: &'a [PageIndexNode],
        parent_id: &'a str,
        stats: &'a mut KGBuildStats,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), GraphStoreError>> + 'a>> {
        Box::pin(async move {
        for node in nodes {
            let section_id = format!("section::{}", node.node_id);

            // Add section node
            let section_node = GraphNode {
                id: section_id.clone(),
                label: node_type::SECTION.to_string(),
                properties: {
                    let mut p = HashMap::new();
                    p.insert("title".into(), node.title.clone());
                    p.insert("summary".into(), node.summary.clone());
                    p.insert("start_index".into(), node.start_index.to_string());
                    p.insert("end_index".into(), node.end_index.to_string());
                    p
                },
            };
            self.graph.add_node(section_node).await?;
            stats.nodes_added += 1;

            // Add CONTAINS edge: parent → section
            let edge = GraphEdge {
                id: self.next_edge_id(),
                source: parent_id.to_string(),
                target: section_id.clone(),
                label: edge_type::CONTAINS.to_string(),
                properties: HashMap::new(),
            };
            self.graph.add_edge(edge).await?;
            stats.edges_added += 1;

            // Extract entities from title + summary
            let text = format!("{} {}", node.title, node.summary);
            let entities = extract_tech_entities(&text);

            for entity_name in &entities {
                let entity_id = format!("entity::{}", entity_name.replace(' ', "_"));

                // Add entity node if not exists (we try, ignore error if duplicate)
                let entity_node = GraphNode {
                    id: entity_id.clone(),
                    label: node_type::ENTITY.to_string(),
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("name".into(), entity_name.clone());
                        p.insert("type".into(), "technology".to_string());
                        p
                    },
                };
                let _ = self.graph.add_node(entity_node).await;
                stats.nodes_added += 1;

                // Add MENTIONS edge: section → entity
                let edge = GraphEdge {
                    id: self.next_edge_id(),
                    source: section_id.clone(),
                    target: entity_id.clone(),
                    label: edge_type::MENTIONS.to_string(),
                    properties: HashMap::new(),
                };
                self.graph.add_edge(edge).await?;
                stats.edges_added += 1;

                // Track for co-occurrence
                self.source_entities
                    .entry(section_id.clone())
                    .or_default()
                    .push(entity_id.clone());

                stats.entities_extracted += 1;
            }

            // Recurse into children
            if !node.children.is_empty() {
                self.process_tree_nodes(&node.children, &section_id, stats).await?;
            }
        }
        Ok(())
        })
    }

    /// Build KG from patent records.
    pub async fn build_from_patents(
        &mut self,
        patents: &[PatentRecord],
    ) -> Result<KGBuildStats, GraphStoreError> {
        let mut stats = KGBuildStats {
            nodes_added: 0,
            edges_added: 0,
            entities_extracted: 0,
            cooccurrence_edges: 0,
        };

        for patent in patents {
            let patent_id = format!("patent::{}", patent.id);

            // Add patent node
            let patent_node = GraphNode {
                id: patent_id.clone(),
                label: node_type::PATENT.to_string(),
                properties: {
                    let mut p = HashMap::new();
                    p.insert("title".into(), patent.title.clone());
                    p.insert("abstract".into(), patent.abstract_text.clone());
                    p
                },
            };
            self.graph.add_node(patent_node).await?;
            stats.nodes_added += 1;

            // Cited patents → REFERENCES edges
            for cited_id in &patent.cited_patents {
                let cited_node_id = format!("patent::{}", cited_id);

                // Add placeholder node if not exists
                let cited_node = GraphNode {
                    id: cited_node_id.clone(),
                    label: node_type::PATENT.to_string(),
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("placeholder".into(), "true".to_string());
                        p
                    },
                };
                let _ = self.graph.add_node(cited_node).await;
                stats.nodes_added += 1;

                let edge = GraphEdge {
                    id: self.next_edge_id(),
                    source: patent_id.clone(),
                    target: cited_node_id,
                    label: edge_type::REFERENCES.to_string(),
                    properties: HashMap::new(),
                };
                self.graph.add_edge(edge).await?;
                stats.edges_added += 1;
            }

            // IPC codes → BELONGS_TO edges
            for ipc in &patent.ipc_codes {
                let ipc_id = format!("ipc::{}", ipc);

                let ipc_node = GraphNode {
                    id: ipc_id.clone(),
                    label: node_type::IPC.to_string(),
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("code".into(), ipc.clone());
                        p
                    },
                };
                let _ = self.graph.add_node(ipc_node).await;
                stats.nodes_added += 1;

                let edge = GraphEdge {
                    id: self.next_edge_id(),
                    source: patent_id.clone(),
                    target: ipc_id,
                    label: edge_type::BELONGS_TO.to_string(),
                    properties: HashMap::new(),
                };
                self.graph.add_edge(edge).await?;
                stats.edges_added += 1;
            }

            // Extract entities from title + abstract
            let text = format!("{} {}", patent.title, patent.abstract_text);
            let entities = extract_tech_entities(&text);

            for entity_name in &entities {
                let entity_id = format!("entity::{}", entity_name.replace(' ', "_"));

                let entity_node = GraphNode {
                    id: entity_id.clone(),
                    label: node_type::ENTITY.to_string(),
                    properties: {
                        let mut p = HashMap::new();
                        p.insert("name".into(), entity_name.clone());
                        p.insert("type".into(), "technology".to_string());
                        p
                    },
                };
                let _ = self.graph.add_node(entity_node).await;
                stats.nodes_added += 1;

                let edge = GraphEdge {
                    id: self.next_edge_id(),
                    source: patent_id.clone(),
                    target: entity_id.clone(),
                    label: edge_type::MENTIONS.to_string(),
                    properties: HashMap::new(),
                };
                self.graph.add_edge(edge).await?;
                stats.edges_added += 1;

                self.source_entities
                    .entry(patent_id.clone())
                    .or_default()
                    .push(entity_id);

                stats.entities_extracted += 1;
            }
        }

        // Build co-occurrence edges
        stats.cooccurrence_edges = self.build_cooccurrence_edges().await?;
        stats.edges_added += stats.cooccurrence_edges;

        info!(
            "Built KG from patents: {} nodes, {} edges, {} entities",
            stats.nodes_added, stats.edges_added, stats.entities_extracted
        );

        Ok(stats)
    }

    /// Build co-occurrence edges between entities mentioned in the same source.
    async fn build_cooccurrence_edges(&mut self) -> Result<usize, GraphStoreError> {
        let mut count = 0;
        let mut seen: HashMap<(String, String), bool> = HashMap::new();

        // Clone entity data to avoid borrow conflict
        let source_entities: Vec<Vec<String>> = self.source_entities.values().cloned().collect();

        for entities in &source_entities {
            for i in 0..entities.len() {
                for j in (i + 1)..entities.len() {
                    let pair = if entities[i] < entities[j] {
                        (entities[i].clone(), entities[j].clone())
                    } else {
                        (entities[j].clone(), entities[i].clone())
                    };

                    if seen.contains_key(&pair) {
                        continue;
                    }
                    seen.insert(pair.clone(), true);

                    let edge = GraphEdge {
                        id: self.next_edge_id(),
                        source: pair.0,
                        target: pair.1,
                        label: edge_type::RELATES_TO.to_string(),
                        properties: HashMap::new(),
                    };
                    self.graph.add_edge(edge).await?;
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_graph_store::JsonGraphStore;

    #[tokio::test]
    async fn test_build_from_pageindex_tree() {
        let graph = Arc::new(JsonGraphStore::new());
        let mut builder = KGBuilder::new(graph.clone());

        let tree = vec![PageIndexNode {
            node_id: "0000".into(),
            title: "Introduction to Federated Learning".into(),
            summary: "Overview of federated learning and differential privacy".into(),
            start_index: 1,
            end_index: 5,
            children: vec![PageIndexNode {
                node_id: "0001".into(),
                title: "Differential Privacy Methods".into(),
                summary: "Laplace mechanism and secure aggregation".into(),
                start_index: 2,
                end_index: 4,
                children: vec![],
            }],
        }];

        let stats = builder.build_from_pageindex_tree(&tree, "doc1", "Test Doc").await.unwrap();

        assert!(stats.nodes_added >= 3); // doc + 2 sections
        assert!(stats.edges_added >= 2); // doc→section edges
        assert!(stats.entities_extracted > 0); // "federated learning", "differential privacy"
    }

    #[tokio::test]
    async fn test_build_from_patents() {
        let graph = Arc::new(JsonGraphStore::new());
        let mut builder = KGBuilder::new(graph.clone());

        let patents = vec![
            PatentRecord {
                id: "CN001".into(),
                title: "联邦学习隐私保护方法".into(),
                abstract_text: "基于差分隐私的联邦学习训练框架".into(),
                cited_patents: vec!["CN003".into()],
                ipc_codes: vec!["G06N".into()],
            },
            PatentRecord {
                id: "CN002".into(),
                title: "分布式模型聚合系统".into(),
                abstract_text: "安全聚合协议用于联邦学习".into(),
                cited_patents: vec!["CN001".into(), "CN003".into()],
                ipc_codes: vec!["G06N".into(), "H04L".into()],
            },
        ];

        let stats = builder.build_from_patents(&patents).await.unwrap();

        assert!(stats.nodes_added >= 4); // 2 patents + cited + IPC
        assert!(stats.edges_added >= 3); // references + belongs_to
        assert!(stats.entities_extracted > 0); // "联邦学习", "差分隐私"
    }

    #[tokio::test]
    async fn test_cooccurrence_edges() {
        let graph = Arc::new(JsonGraphStore::new());
        let mut builder = KGBuilder::new(graph.clone());

        let patents = vec![PatentRecord {
            id: "P1".into(),
            title: "Federated Learning with Differential Privacy".into(),
            abstract_text: "Combining machine learning and secure aggregation".into(),
            cited_patents: vec![],
            ipc_codes: vec![],
        }];

        let stats = builder.build_from_patents(&patents).await.unwrap();

        // "federated learning" and "differential privacy" co-occur → RELATES_TO edge
        assert!(stats.cooccurrence_edges > 0);
    }
}
