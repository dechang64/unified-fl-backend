//! Integration Tests — End-to-End verification of all subsystems.

use std::sync::Arc;

fn test_runtime() -> unified_fl_backend::FedCtxRuntime {
    unified_fl_backend::FedCtxRuntime::new(64, 100, None)
}

#[tokio::test]
async fn test_vector_insert_and_search() {
    let rt = test_runtime();
    let db = rt.vector_db();
    let entry = unified_fl_backend::vector_db::VectorEntry {
        id: "test_1".to_string(),
        values: vec![1.0; 64],
        metadata: Default::default(),
    };
    {
        let mut db_guard = db.write();
        db_guard.insert(entry).unwrap();
    }
    let results = {
        let db_guard = db.read();
        db_guard.search(&[1.0; 64], 5)
    };
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test_1");
}

#[tokio::test]
async fn test_graph_add_and_traverse() {
    let rt = test_runtime();
    let graph = rt.graph_store();
    use unified_fl_backend::traits::{GraphStore, GraphNode, GraphEdge};

    graph.add_node(GraphNode { id: "a".into(), label: "A".into(), properties: Default::default() }).await.unwrap();
    graph.add_node(GraphNode { id: "b".into(), label: "B".into(), properties: Default::default() }).await.unwrap();
    graph.add_edge(GraphEdge { id: "e1".into(), source: "a".into(), target: "b".into(), label: "LINKS".into(), properties: Default::default() }).await.unwrap();

    let path = graph.traverse("a", None, 1).await.unwrap();
    assert_eq!(path.nodes.len(), 2);
}

#[tokio::test]
async fn test_memory_remember_and_recall() {
    let rt = test_runtime();
    let memory = rt.memory_store();
    use unified_fl_backend::traits::MemoryStore;

    memory.remember(
        "test memory content",
        unified_fl_backend::traits::MemoryRole::User,
        None,
        1.0,
        std::collections::HashMap::new(),
    ).await.unwrap();

    let hits = memory.recall("test memory", 5, None).await.unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0].entry.content, "test memory content");
}

#[tokio::test]
async fn test_audit_append_and_verify() {
    let rt = test_runtime();
    let audit = rt.audit_chain();
    let mut audit_guard = audit.write();
    let block = audit_guard.append(
        unified_fl_backend::audit::EventType::Custom { name: "test_event".into() },
        "test_node",
        serde_json::json!({"key": "value"}),
    );
    assert!(block.sequence > 0);
    assert!(audit_guard.verify().is_ok());
}

#[tokio::test]
async fn test_fl_engine_fedavg() {
    use unified_fl_backend::fl_engine::{FlEngine, FedAvg, ModelUpdate};

    let engine = FlEngine::new(Arc::new(FedAvg), 2);
    let round = engine.start_round();
    let now = chrono::Utc::now().to_rfc3339();

    engine.submit_update(ModelUpdate {
        client_id: "c1".into(), round, parameters: vec![1.0, 2.0],
        num_samples: 100, train_loss: Some(0.5), entropy: None, submitted_at: now.clone(),
    }).unwrap();
    engine.submit_update(ModelUpdate {
        client_id: "c2".into(), round, parameters: vec![3.0, 4.0],
        num_samples: 100, train_loss: Some(0.3), entropy: None, submitted_at: now,
    }).unwrap();

    let result = engine.aggregate_round(round).await.unwrap();
    assert_eq!(result.num_clients, 2);
    assert!((result.parameters[0] - 2.0).abs() < 0.001);
}

#[tokio::test]
async fn test_config_roundtrip() {
    use unified_fl_backend::config::FedCtxConfig;
    let config = FedCtxConfig::default();
    let toml = config.to_toml_string().unwrap();
    let parsed = FedCtxConfig::load_from_str(&toml).unwrap();
    assert_eq!(parsed.server.grpc_port, config.server.grpc_port);
}

#[tokio::test]
async fn test_embedding_hash() {
    use unified_fl_backend::embedding::{EmbeddingProvider, HashEmbedder};
    let embedder = HashEmbedder::new(64);
    let v1 = embedder.embed("hello").await.unwrap();
    let v2 = embedder.embed("hello").await.unwrap();
    assert_eq!(v1, v2);
    assert_eq!(v1.len(), 64);
}

#[tokio::test]
async fn test_metrics() {
    use unified_fl_backend::metrics::Metrics;
    let metrics = Metrics::new();
    metrics.inc(&metrics.vector_inserts);
    metrics.inc(&metrics.vector_inserts);
    metrics.inc(&metrics.vector_searches);
    let output = metrics.render();
    assert!(output.contains("fedctx_vector_operations_total{op=\"insert\"} 2"));
}
