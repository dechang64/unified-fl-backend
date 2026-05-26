//! Prometheus Metrics Endpoint
//!
//! Exposes `/metrics` for Prometheus scraping.
//! Tracks: vector operations, graph operations, FL rounds, memory operations,
//! audit events, and system health.

use axum::{extract::State, http::StatusCode, response::IntoResponse};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Global metrics collector.
#[derive(Debug)]
pub struct Metrics {
    pub vector_inserts: AtomicU64,
    pub vector_searches: AtomicU64,
    pub vector_deletes: AtomicU64,
    pub graph_nodes_added: AtomicU64,
    pub graph_edges_added: AtomicU64,
    pub graph_traversals: AtomicU64,
    pub fl_updates_submitted: AtomicU64,
    pub fl_rounds_aggregated: AtomicU64,
    pub memory_remembers: AtomicU64,
    pub memory_recalls: AtomicU64,
    pub audit_appends: AtomicU64,
    pub start_time: Instant,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            vector_inserts: AtomicU64::new(0),
            vector_searches: AtomicU64::new(0),
            vector_deletes: AtomicU64::new(0),
            graph_nodes_added: AtomicU64::new(0),
            graph_edges_added: AtomicU64::new(0),
            graph_traversals: AtomicU64::new(0),
            fl_updates_submitted: AtomicU64::new(0),
            fl_rounds_aggregated: AtomicU64::new(0),
            memory_remembers: AtomicU64::new(0),
            memory_recalls: AtomicU64::new(0),
            audit_appends: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Increment a counter.
    pub fn inc(&self, counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Render metrics in Prometheus exposition format.
    pub fn render(&self) -> String {
        let uptime = self.start_time.elapsed().as_secs();
        let mut lines = Vec::new();

        // Vector operations
        lines.push("# HELP fedctx_vector_operations_total Total vector operations".to_string());
        lines.push("# TYPE fedctx_vector_operations_total counter".to_string());
        lines.push(format!("fedctx_vector_operations_total{{op=\"insert\"}} {}", self.vector_inserts.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_vector_operations_total{{op=\"search\"}} {}", self.vector_searches.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_vector_operations_total{{op=\"delete\"}} {}", self.vector_deletes.load(Ordering::Relaxed)));

        // Graph operations
        lines.push("# HELP fedctx_graph_operations_total Total graph operations".to_string());
        lines.push("# TYPE fedctx_graph_operations_total counter".to_string());
        lines.push(format!("fedctx_graph_operations_total{{op=\"add_node\"}} {}", self.graph_nodes_added.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_graph_operations_total{{op=\"add_edge\"}} {}", self.graph_edges_added.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_graph_operations_total{{op=\"traverse\"}} {}", self.graph_traversals.load(Ordering::Relaxed)));

        // FL operations
        lines.push("# HELP fedctx_fl_operations_total Total federated learning operations".to_string());
        lines.push("# TYPE fedctx_fl_operations_total counter".to_string());
        lines.push(format!("fedctx_fl_operations_total{{op=\"submit_update\"}} {}", self.fl_updates_submitted.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_fl_operations_total{{op=\"aggregate\"}} {}", self.fl_rounds_aggregated.load(Ordering::Relaxed)));

        // Memory operations
        lines.push("# HELP fedctx_memory_operations_total Total memory operations".to_string());
        lines.push("# TYPE fedctx_memory_operations_total counter".to_string());
        lines.push(format!("fedctx_memory_operations_total{{op=\"remember\"}} {}", self.memory_remembers.load(Ordering::Relaxed)));
        lines.push(format!("fedctx_memory_operations_total{{op=\"recall\"}} {}", self.memory_recalls.load(Ordering::Relaxed)));

        // Audit operations
        lines.push("# HELP fedctx_audit_appends_total Total audit log appends".to_string());
        lines.push("# TYPE fedctx_audit_appends_total counter".to_string());
        lines.push(format!("fedctx_audit_appends_total {}", self.audit_appends.load(Ordering::Relaxed)));

        // System
        lines.push("# HELP fedctx_uptime_seconds Server uptime in seconds".to_string());
        lines.push("# TYPE fedctx_uptime_seconds gauge".to_string());
        lines.push(format!("fedctx_uptime_seconds {}", uptime));

        lines.join("\n")
    }
}

/// Shared metrics state.
pub type SharedMetrics = Arc<Metrics>;

/// Axum handler for `/metrics`.
pub async fn metrics_handler(State(metrics): State<SharedMetrics>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics.render(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_render() {
        let metrics = Metrics::new();
        metrics.inc(&metrics.vector_inserts);
        metrics.inc(&metrics.vector_inserts);
        metrics.inc(&metrics.vector_searches);
        
        let output = metrics.render();
        assert!(output.contains("fedctx_vector_operations_total{op=\"insert\"} 2"));
        assert!(output.contains("fedctx_vector_operations_total{op=\"search\"} 1"));
        assert!(output.contains("fedctx_uptime_seconds"));
    }

    #[test]
    fn test_metrics_concurrent_increment() {
        let metrics = Arc::new(Metrics::new());
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let m = metrics.clone();
                std::thread::spawn(move || {
                    m.inc(&m.vector_inserts);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(metrics.vector_inserts.load(Ordering::Relaxed), 10);
    }
}
