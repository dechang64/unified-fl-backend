//! PageIndex Tree Index — Hierarchical document index for structural retrieval
//!
//! Inspired by VectifyAI/PageIndex: builds a tree index from documents
//! and uses keyword/structure matching for retrieval (no vectors needed).
//!
//! Tree structure:
//!   Node { node_id, title, summary, start_index, end_index, children }
//!
//! This is the structural retrieval channel in the hybrid search pipeline.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::info;

// ════════════════════════════════════════════════════════════════
// Data Types
// ════════════════════════════════════════════════════════════════

/// A node in the PageIndex tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndexNode {
    pub node_id: String,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    pub start_index: usize,
    pub end_index: usize,
    #[serde(default)]
    pub children: Vec<PageIndexNode>,
}

/// A document indexed by PageIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndexDocument {
    pub doc_id: String,
    pub doc_name: String,
    pub page_count: usize,
    pub root_nodes: Vec<PageIndexNode>,
}

/// A search result from PageIndex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndexHit {
    pub doc_id: String,
    pub node_id: String,
    pub title: String,
    pub summary: String,
    pub score: f64,
    pub start_index: usize,
    pub end_index: usize,
}

/// PageIndex tree store — manages indexed documents.
pub struct PageIndexStore {
    documents: HashMap<String, PageIndexDocument>,
    /// Flat node index: (doc_id, node_id) → node path for fast lookup
    node_index: HashMap<(String, String), NodePath>,
}

/// Path to a node in the tree (for breadcrumb/context).
#[derive(Debug, Clone)]
struct NodePath {
    titles: Vec<String>,
    summary: String,
    start_index: usize,
    end_index: usize,
}

impl PageIndexStore {
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            node_index: HashMap::new(),
        }
    }

    /// Index a document with its tree structure.
    pub fn index_document(&mut self, doc: PageIndexDocument) -> String {
        let doc_id = doc.doc_id.clone();
        // Build flat node index
        self.index_nodes_recursive(&doc.root_nodes, &doc_id, Vec::new());
        self.documents.insert(doc_id.clone(), doc);
        info!("PageIndex: indexed document, total={}", self.documents.len());
        doc_id
    }

    fn index_nodes_recursive(
        &mut self,
        nodes: &[PageIndexNode],
        doc_id: &str,
        parent_titles: Vec<String>,
    ) {
        for node in nodes {
            let mut path = parent_titles.clone();
            path.push(node.title.clone());

            self.node_index.insert(
                (doc_id.to_string(), node.node_id.clone()),
                NodePath {
                    titles: path.clone(),
                    summary: node.summary.clone(),
                    start_index: node.start_index,
                    end_index: node.end_index,
                },
            );

            self.index_nodes_recursive(&node.children, doc_id, path);
        }
    }

    /// Search the tree index by keywords.
    ///
    /// Scoring: title overlap (3x) + summary overlap (2x) + depth bonus.
    pub fn search(&self, query: &str, doc_id: Option<&str>, top_k: usize) -> Vec<PageIndexHit> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut hits: Vec<PageIndexHit> = Vec::new();

        for (doc_id_key, doc) in &self.documents {
            if let Some(did) = doc_id {
                if did != doc_id_key {
                    continue;
                }
            }
            self.search_in_nodes(&doc.root_nodes, doc_id_key, &query_terms, &mut hits, 0);
        }

        // Sort by score descending
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);
        hits
    }

    fn search_in_nodes(
        &self,
        nodes: &[PageIndexNode],
        doc_id: &str,
        query_terms: &[&str],
        hits: &mut Vec<PageIndexHit>,
        depth: usize,
    ) {
        for node in nodes {
            let title_lower = node.title.to_lowercase();
            let summary_lower = node.summary.to_lowercase();

            let title_overlap = query_terms.iter().filter(|t| title_lower.contains(*t)).count() as f64;
            let summary_overlap = query_terms.iter().filter(|t| summary_lower.contains(*t)).count() as f64;

            if title_overlap + summary_overlap > 0.0 {
                let depth_bonus = 0.1 * (depth as f64);
                let score = 3.0 * title_overlap + 2.0 * summary_overlap + depth_bonus;

                hits.push(PageIndexHit {
                    doc_id: doc_id.to_string(),
                    node_id: node.node_id.clone(),
                    title: node.title.clone(),
                    summary: node.summary.clone(),
                    score: score.min(1.0),
                    start_index: node.start_index,
                    end_index: node.end_index,
                });
            }

            self.search_in_nodes(&node.children, doc_id, query_terms, hits, depth + 1);
        }
    }

    /// Get document count.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Get a document by ID.
    pub fn get_document(&self, doc_id: &str) -> Option<&PageIndexDocument> {
        self.documents.get(doc_id)
    }
}

impl Default for PageIndexStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_doc() -> PageIndexDocument {
        PageIndexDocument {
            doc_id: "doc1".into(),
            doc_name: "test.pdf".into(),
            page_count: 10,
            root_nodes: vec![
                PageIndexNode {
                    node_id: "0000".into(),
                    title: "Introduction".into(),
                    summary: "Overview of federated learning".into(),
                    start_index: 1,
                    end_index: 3,
                    children: vec![PageIndexNode {
                        node_id: "0001".into(),
                        title: "Background".into(),
                        summary: "Differential privacy in FL".into(),
                        start_index: 1,
                        end_index: 2,
                        children: vec![],
                    }],
                },
                PageIndexNode {
                    node_id: "0002".into(),
                    title: "Monetary Policy".into(),
                    summary: "Central bank decisions".into(),
                    start_index: 4,
                    end_index: 7,
                    children: vec![],
                },
            ],
        }
    }

    #[test]
    fn test_index_and_search() {
        let mut store = PageIndexStore::new();
        store.index_document(make_test_doc());
        assert_eq!(store.doc_count(), 1);

        let hits = store.search("federated learning", None, 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].node_id, "0000"); // "Introduction" matches
    }

    #[test]
    fn test_search_by_doc_id() {
        let mut store = PageIndexStore::new();
        store.index_document(make_test_doc());

        let hits = store.search("Monetary", Some("doc1"), 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].node_id, "0002");
    }

    #[test]
    fn test_search_no_match() {
        let mut store = PageIndexStore::new();
        store.index_document(make_test_doc());

        let hits = store.search("xylophone zebra", None, 5);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_search_child_node() {
        let mut store = PageIndexStore::new();
        store.index_document(make_test_doc());

        let hits = store.search("differential privacy", None, 5);
        assert!(hits.iter().any(|h| h.node_id == "0001"));
    }
}
