//! Hybrid Search — Query Router + RRF Fusion + PageRank Reranking
//!
//! The core retrieval pipeline:
//!   1. Query Router: classify query as structural / semantic / hybrid
//!   2. Dual-channel retrieval: PageIndex (structural) + HNSW (semantic)
//!   3. RRF Fusion: merge results from both channels
//!   4. PageRank Reranking: boost results from important KG nodes
//!
//! RRF (Reciprocal Rank Fusion):
//!   score(d) = Σ_r 1 / (k + rank_r(d))
//!
//! PageRank Reranking:
//!   final_score = α * rrf_normalized + (1-α) * pagerank_normalized

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::traits::VectorStore;
use crate::pageindex::{PageIndexStore, PageIndexHit};

// ════════════════════════════════════════════════════════════════
// Query Router
// ════════════════════════════════════════════════════════════════

/// Query classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Structural document query → PageIndex tree search.
    Structural,
    /// Semantic similarity query → HNSW vector search.
    Semantic,
    /// Mixed signals → both channels + RRF fusion.
    Hybrid,
}

/// Routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingResult {
    pub query_type: QueryType,
    pub confidence: f64,
    pub reason: String,
}

/// Route a query to the appropriate retrieval channel.
pub fn route_query(query: &str, has_pageindex: bool, has_hnsw: bool) -> RoutingResult {
    let q = query.to_lowercase();

    // Structural patterns
    let structural_patterns = [
        regex::Regex::new(r"(第[一二三四五六七八九十\d]+[章节条])").ok(),
        regex::Regex::new(r"(section|chapter|part)\s+\d+").ok(),
        regex::Regex::new(r"(目录|索引|table of contents)").ok(),
        regex::Regex::new(r"(哪一页|在哪页|which page)").ok(),
        regex::Regex::new(r"(摘要|summary|abstract)").ok(),
        regex::Regex::new(r"(定义|什么是|what is .+ defined)").ok(),
        regex::Regex::new(r"(法规|条例|regulation|act)").ok(),
    ];

    // Semantic patterns
    let semantic_patterns = [
        regex::Regex::new(r"(相似|类似|similar|like)").ok(),
        regex::Regex::new(r"(相关|related|relevant)").ok(),
        regex::Regex::new(r"(比较|对比|compare|versus)").ok(),
        regex::Regex::new(r"(推荐|recommend|suggest)").ok(),
        regex::Regex::new(r"(趋势|走势|trend)").ok(),
        regex::Regex::new(r"(排名|排行|rank)").ok(),
        regex::Regex::new(r"(最优|最佳|best|optimal)").ok(),
    ];

    let mut structural_score = 0.0_f64;
    let mut semantic_score = 0.0_f64;

    for pat in &structural_patterns {
        if let Some(re) = pat {
            if re.is_match(&q) {
                structural_score += 1.0;
            }
        }
    }

    for pat in &semantic_patterns {
        if let Some(re) = pat {
            if re.is_match(&q) {
                semantic_score += 1.0;
            }
        }
    }

    // Length heuristic
    if query.len() < 15 {
        semantic_score += 0.5;
    } else if query.len() > 50 {
        structural_score += 0.5;
    }

    // Fallback if one channel is unavailable
    if !has_pageindex && structural_score > 0.0 {
        semantic_score += structural_score;
        structural_score = 0.0;
    }
    if !has_hnsw && semantic_score > 0.0 {
        structural_score += semantic_score;
        semantic_score = 0.0;
    }

    let total = structural_score + semantic_score;
    if total == 0.0 {
        return RoutingResult {
            query_type: QueryType::Hybrid,
            confidence: 0.3,
            reason: "No clear routing signal, defaulting to hybrid".into(),
        };
    }

    let structural_ratio = structural_score / total;
    let semantic_ratio = semantic_score / total;

    if structural_ratio > 0.65 {
        RoutingResult {
            query_type: QueryType::Structural,
            confidence: structural_ratio,
            reason: format!("Structural patterns matched (score={structural_score:.1})"),
        }
    } else if semantic_ratio > 0.65 {
        RoutingResult {
            query_type: QueryType::Semantic,
            confidence: semantic_ratio,
            reason: format!("Semantic patterns matched (score={semantic_score:.1})"),
        }
    } else {
        RoutingResult {
            query_type: QueryType::Hybrid,
            confidence: structural_ratio.max(semantic_ratio),
            reason: format!("Mixed signals (struct={structural_score:.1}, semantic={semantic_score:.1})"),
        }
    }
}

// ════════════════════════════════════════════════════════════════
// Unified Search Result
// ════════════════════════════════════════════════════════════════

/// A unified search result from any channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchHit {
    pub id: String,
    pub title: String,
    pub score: f64,
    pub rrf_score: Option<f64>,
    pub pagerank_score: Option<f64>,
    pub final_score: Option<f64>,
    pub sources: Vec<String>,
    pub metadata: HashMap<String, String>,
}

// ════════════════════════════════════════════════════════════════
// RRF Fusion
// ════════════════════════════════════════════════════════════════

const RRF_K: f64 = 60.0;

/// Fuse results from multiple channels using Reciprocal Rank Fusion.
pub fn rrf_fuse(
    pageindex_results: &[PageIndexHit],
    hnsw_results: &[(String, f32, HashMap<String, String>)],
) -> Vec<HybridSearchHit> {
    let mut rrf_scores: HashMap<String, f64> = HashMap::new();
    let mut result_map: HashMap<String, HybridSearchHit> = HashMap::new();

    // PageIndex results
    for (rank, hit) in pageindex_results.iter().enumerate() {
        let id = format!("{}::{}", hit.doc_id, hit.node_id);
        let rrf = 1.0 / (RRF_K + (rank + 1) as f64);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += rrf;

        result_map.entry(id.clone()).or_insert_with(|| HybridSearchHit {
            id: id.clone(),
            title: hit.title.clone(),
            score: hit.score,
            rrf_score: None,
            pagerank_score: None,
            final_score: None,
            sources: vec!["pageindex".into()],
            metadata: {
                let mut m = HashMap::new();
                m.insert("doc_id".into(), hit.doc_id.clone());
                m.insert("node_id".into(), hit.node_id.clone());
                m
            },
        });

        if let Some(entry) = result_map.get_mut(&id) {
            if !entry.sources.contains(&"pageindex".into()) {
                entry.sources.push("pageindex".into());
            }
        }
    }

    // HNSW results
    for (rank, (id, dist, meta)) in hnsw_results.iter().enumerate() {
        let rrf = 1.0 / (RRF_K + (rank + 1) as f64);
        *rrf_scores.entry(id.clone()).or_insert(0.0) += rrf;

        result_map.entry(id.clone()).or_insert_with(|| HybridSearchHit {
            id: id.clone(),
            title: meta.get("title").cloned().unwrap_or_default(),
            score: *dist as f64,
            rrf_score: None,
            pagerank_score: None,
            final_score: None,
            sources: vec!["hnsw".into()],
            metadata: meta.clone(),
        });

        if let Some(entry) = result_map.get_mut(id) {
            if !entry.sources.contains(&"hnsw".into()) {
                entry.sources.push("hnsw".into());
            }
        }
    }

    // Sort by RRF score and attach
    let mut fused: Vec<HybridSearchHit> = result_map.into_values().collect();
    for hit in &mut fused {
        hit.rrf_score = rrf_scores.get(&hit.id).copied();
    }
    fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));

    fused
}

// ════════════════════════════════════════════════════════════════
// PageRank Reranking
// ════════════════════════════════════════════════════════════════

/// Rerank fused results using PageRank scores.
///
/// final_score = α * rrf_normalized + (1-α) * pagerank_normalized
pub fn pagerank_rerank(
    results: &mut [HybridSearchHit],
    pagerank_scores: &HashMap<String, f64>,
    alpha: f64,
) {
    if results.is_empty() {
        return;
    }

    let rrf_max = results.iter().filter_map(|r| r.rrf_score).fold(0.0_f64, f64::max).max(1e-10);

    let pr_scores: Vec<f64> = results
        .iter()
        .map(|r| {
            let direct = pagerank_scores.get(&r.id).copied().unwrap_or(0.0);
            if direct > 0.0 {
                return direct;
            }
            // Try prefix matching
            for (key, val) in pagerank_scores {
                if key.ends_with(&format!("::{}", r.id)) || r.id.ends_with(&format!("::{}", key)) {
                    return *val;
                }
            }
            0.0
        })
        .collect();

    let pr_max = pr_scores.iter().copied().fold(0.0_f64, f64::max).max(1e-10);

    for (i, hit) in results.iter_mut().enumerate() {
        let rrf_norm = hit.rrf_score.unwrap_or(0.0) / rrf_max;
        let pr_norm = pr_scores[i] / pr_max;
        hit.pagerank_score = Some(pr_scores[i]);
        hit.final_score = Some(alpha * rrf_norm + (1.0 - alpha) * pr_norm);
    }

    results.sort_by(|a, b| {
        b.final_score.unwrap_or(0.0).partial_cmp(&a.final_score.unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ════════════════════════════════════════════════════════════════
// Hybrid Search Engine
// ════════════════════════════════════════════════════════════════

/// The hybrid search engine — orchestrates the full pipeline.
pub struct HybridSearchEngine {
    pub pageindex: PageIndexStore,
    pub pagerank_scores: HashMap<String, f64>,
    pub rrf_k: f64,
    pub pagerank_alpha: f64,
}

impl HybridSearchEngine {
    pub fn new() -> Self {
        Self {
            pageindex: PageIndexStore::new(),
            pagerank_scores: HashMap::new(),
            rrf_k: RRF_K,
            pagerank_alpha: 0.7,
        }
    }

    /// Set PageRank scores (computed externally).
    pub fn set_pagerank_scores(&mut self, scores: HashMap<String, f64>) {
        self.pagerank_scores = scores;
    }

    /// Full hybrid search pipeline.
    pub async fn search(
        &self,
        query: &str,
        vector_store: &Arc<dyn VectorStore>,
        query_vector: Option<&[f32]>,
        k: usize,
    ) -> (RoutingResult, Vec<HybridSearchHit>) {
        let has_pi = self.pageindex.doc_count() > 0;
        let has_hnsw = query_vector.is_some();

        // 1. Route
        let routing = route_query(query, has_pi, has_hnsw);
        info!("HybridSearch: query='{}' → {:?} ({})", query, routing.query_type, routing.reason);

        // 2. Retrieve from channels
        let pi_results = match routing.query_type {
            QueryType::Structural | QueryType::Hybrid => {
                self.pageindex.search(query, None, k)
            }
            QueryType::Semantic => vec![],
        };

        let hnsw_results = match (routing.query_type, query_vector) {
            (QueryType::Semantic | QueryType::Hybrid, Some(vec)) => {
                match vector_store.search(vec, k, None).await {
                    Ok(hits) => hits.into_iter().map(|h| (h.id, h.score, h.metadata)).collect(),
                    Err(_) => vec![],
                }
            }
            _ => vec![],
        };

        // 3. RRF fusion
        let mut fused = rrf_fuse(&pi_results, &hnsw_results);

        // 4. PageRank reranking
        if !self.pagerank_scores.is_empty() {
            pagerank_rerank(&mut fused, &self.pagerank_scores, self.pagerank_alpha);
        }

        fused.truncate(k);
        (routing, fused)
    }
}

impl Default for HybridSearchEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_structural() {
        let r = route_query("第三章讲了什么？", true, true);
        assert_eq!(r.query_type, QueryType::Structural);
    }

    #[test]
    fn test_route_semantic() {
        let r = route_query("找跟联邦学习相似的论文", true, true);
        assert_eq!(r.query_type, QueryType::Semantic);
    }

    #[test]
    fn test_route_hybrid() {
        let r = route_query("比较第三章和第五章的方法", true, true);
        assert_eq!(r.query_type, QueryType::Hybrid);
    }

    #[test]
    fn test_rrf_fuse_basic() {
        let pi_hits = vec![PageIndexHit {
            doc_id: "doc1".into(),
            node_id: "0000".into(),
            title: "Introduction".into(),
            summary: "".into(),
            score: 0.9,
            start_index: 1,
            end_index: 5,
        }];

        let hnsw_hits = vec![
            ("doc1::0000".into(), 0.85_f32, HashMap::new()),
            ("doc2".into(), 0.7_f32, HashMap::new()),
        ];

        let fused = rrf_fuse(&pi_hits, &hnsw_hits);
        assert_eq!(fused.len(), 2); // doc1::0000 + doc2
        // doc1::0000 appears in both channels → highest RRF
        assert!(fused[0].rrf_score.unwrap() > fused[1].rrf_score.unwrap());
        assert!(fused[0].sources.contains(&"pageindex".into()));
        assert!(fused[0].sources.contains(&"hnsw".into()));
    }

    #[test]
    fn test_pagerank_rerank() {
        let mut results = vec![
            HybridSearchHit {
                id: "doc1".into(),
                title: "Doc 1".into(),
                score: 0.9,
                rrf_score: Some(0.032),
                pagerank_score: None,
                final_score: None,
                sources: vec![],
                metadata: HashMap::new(),
            },
            HybridSearchHit {
                id: "doc2".into(),
                title: "Doc 2".into(),
                score: 0.7,
                rrf_score: Some(0.016),
                pagerank_score: None,
                final_score: None,
                sources: vec![],
                metadata: HashMap::new(),
            },
        ];

        let mut pr = HashMap::new();
        pr.insert("doc2".into(), 0.8);
        pr.insert("doc1".into(), 0.2);

        pagerank_rerank(&mut results, &pr, 0.3);

        // doc2 has much higher PageRank, with alpha=0.3 it should win
        assert_eq!(results[0].id, "doc2");
        assert!(results[0].final_score.is_some());
    }

    #[test]
    fn test_rrf_empty() {
        let fused = rrf_fuse(&[], &[]);
        assert!(fused.is_empty());
    }
}
