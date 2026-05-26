//! FedCtx — Federated Semantic Infrastructure Entry Point
//!
//! Three modes:
//! 1. Default: gRPC + HTTP REST servers with scenario plugins
//! 2. --mcp: MCP stdio server (for Claude Desktop / Cursor integration)
//! 3. --mcp-http: MCP Streamable HTTP server (for remote deployment)

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

use unified_fl_backend::FedCtxRuntime;
use unified_fl_backend::FedCtxMcpServer;
use unified_fl_backend::MedGpsPlugin;
use unified_fl_backend::grpc_service;
use rmcp::ServiceExt;

#[derive(Parser)]
#[command(name = "fedctx")]
#[command(about = "FedCtx — Federated Semantic Infrastructure: HNSW + Graph + Audit + gRPC + REST + MCP")]
struct Cli {
    /// Vector dimension
    #[arg(long, default_value = "384")]
    dimension: usize,

    /// gRPC port
    #[arg(long, default_value = "50051")]
    grpc_port: u16,

    /// HTTP REST port
    #[arg(long, default_value = "8080")]
    http_port: u16,

    /// Max audit chain blocks
    #[arg(long, default_value = "10000")]
    max_audit_blocks: usize,

    /// Data directory for persistence (vectors.bin, graph.json, audit.json)
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Auto-persist interval in seconds (0 = disabled)
    #[arg(long, default_value = "300")]
    auto_persist_secs: u64,

    /// Embedding model directory (ONNX format). If not set, uses hash embedder.
    #[arg(long)]
    embedding_model_dir: Option<PathBuf>,

    /// Disable MedGps plugin
    #[arg(long)]
    no_med_gps: bool,

    /// Run as MCP stdio server
    #[arg(long)]
    mcp: bool,

    /// Run as MCP HTTP server on given port
    #[arg(long)]
    mcp_http: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into())
        )
        .init();

    let cli = Cli::parse();

    if cli.mcp {
        return run_mcp_stdio(cli).await;
    }
    if let Some(port) = cli.mcp_http {
        return run_mcp_http(cli, port).await;
    }

    // ── Create runtime with options ──
    let mut runtime = FedCtxRuntime::with_options(
        cli.dimension,
        cli.max_audit_blocks,
        cli.data_dir.clone(),
        cli.embedding_model_dir.clone(),
        Some(cli.auto_persist_secs),
    );

    // ── Register plugins ──
    if !cli.no_med_gps {
        runtime.register(Box::new(MedGpsPlugin::new()));
    }

    // ── Initialize plugins ──
    runtime.initialize_plugins().await;

    // ── Start auto-persist ──
    let persist_handle = runtime.start_auto_persist();

    // ── Start gRPC server ──
    let db = runtime.vector_db();
    let audit = runtime.audit_chain();
    let fl_engine = runtime.fl_engine();
    let embedding = runtime.stores().embedding.clone()
        .expect("embedding provider should be initialized");
    let vector_store_trait = runtime.stores().vector_store.clone();
    let graph_store_trait = runtime.stores().graph_store.clone();
    let embedding_for_graphrag = runtime.stores().embedding.clone().unwrap();
    let grpc_addr = format!("0.0.0.0:{}", cli.grpc_port);
    let grpc_handle = tokio::spawn(async move {
        let mut builder = tonic::transport::Server::builder();
        let vector_db_service =
            unified_fl_backend::proto::vector_database_server::VectorDatabaseServer::new(
                grpc_service::VectorDBService::new(db, audit.clone()),
            );
        let audit_service =
            unified_fl_backend::proto::audit_service_server::AuditServiceServer::new(
                grpc_service::AuditServiceImpl::new(audit),
            );
        // FL gRPC service
        let fl_service = unified_fl_backend::proto::federated_learning_server::FederatedLearningServer::new(
            unified_fl_backend::fl_service::FlServiceImpl::new(fl_engine),
        );
        // Embedding gRPC service
        let embedding_service = unified_fl_backend::proto::embedding_service_server::EmbeddingServiceServer::new(
            unified_fl_backend::embedding_service::EmbeddingServiceImpl::new(embedding),
        );
        // GraphRAG gRPC service
        let graphrag_engine = Arc::new(unified_fl_backend::graphrag::GraphRAGEngine::new(
            vector_store_trait,
            graph_store_trait,
            embedding_for_graphrag,
        ));
        let graphrag_service = unified_fl_backend::proto::graph_rag_service_server::GraphRagServiceServer::new(
            unified_fl_backend::graphrag_service::GraphRAGServiceImpl::new(graphrag_engine),
        );
        if let Err(e) = builder
            .add_service(vector_db_service)
            .add_service(audit_service)
            .add_service(fl_service)
            .add_service(embedding_service)
            .add_service(graphrag_service)
            .serve(grpc_addr.parse().unwrap())
            .await
        {
            tracing::error!("gRPC server error: {}", e);
        }
    });

    // ── Start HTTP REST server ──
    let http_db = runtime.vector_db();
    let http_audit = runtime.audit_chain();
    let http_graph = Some(runtime.graph_store());
    let http_memory = Some(runtime.memory_store());
    let http_embedding = runtime.stores().embedding.clone();
    let http_addr = format!("0.0.0.0:{}", cli.http_port);
    let plugin_routes = runtime.collect_rest_routes();
    let http_handle = tokio::spawn(async move {
        let app = unified_fl_backend::rest_api::create_router(
            http_db, http_audit, http_graph, http_memory, http_embedding,
        ).merge(plugin_routes);
        let listener = tokio::net::TcpListener::bind(&http_addr).await.unwrap();
        info!("HTTP REST server listening on http://{}", http_addr);
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    // ── Wait for shutdown ──
    tokio::select! {
        _ = grpc_handle => info!("gRPC server stopped"),
        _ = http_handle => info!("HTTP server stopped"),
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal, draining...");
            // Give inflight requests a moment to complete
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    // Graceful shutdown: persist + stop plugins
    if let Some(handle) = persist_handle {
        handle.abort();
    }
    runtime.shutdown().await;
    info!("FedCtx shut down gracefully");
    Ok(())
}

async fn run_mcp_stdio(cli: Cli) -> anyhow::Result<()> {
    let data_dir = cli.data_dir.clone();
    let mut runtime = FedCtxRuntime::new(
        cli.dimension,
        cli.max_audit_blocks,
        cli.data_dir,
    );

    if !cli.no_med_gps {
        runtime.register(Box::new(MedGpsPlugin::new()));
    }
    runtime.initialize_plugins().await;

    let mut mcp_server = FedCtxMcpServer::new(
        runtime.vector_db(),
        runtime.graph_store(),
        runtime.audit_chain(),
        runtime.memory_store(),
    );
    if let Some(dir) = data_dir {
        mcp_server = mcp_server.with_data_dir(dir);
    }

    let service = mcp_server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

async fn run_mcp_http(cli: Cli, port: u16) -> anyhow::Result<()> {
    let data_dir = cli.data_dir.clone();
    let mut runtime = FedCtxRuntime::new(
        cli.dimension,
        cli.max_audit_blocks,
        cli.data_dir,
    );

    if !cli.no_med_gps {
        runtime.register(Box::new(MedGpsPlugin::new()));
    }
    runtime.initialize_plugins().await;

    let data_dir_for_mcp = data_dir.clone();
    let mcp_service = rmcp::transport::streamable_http_server::StreamableHttpService::new(
        // Service factory: creates a new FedCtxMcpServer per session
        move || {
            let mut server = FedCtxMcpServer::new(
                runtime.vector_db(),
                runtime.graph_store(),
                runtime.audit_chain(),
                runtime.memory_store(),
            );
            if let Some(ref dir) = data_dir_for_mcp {
                server = server.with_data_dir(dir.clone());
            }
            Ok(server)
        },
        // In-memory session manager
        std::sync::Arc::new(
            rmcp::transport::streamable_http_server::session::local::LocalSessionManager::default()
        ),
        rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default()
            .with_stateful_mode(true)
            .with_sse_keep_alive(Some(std::time::Duration::from_secs(30))),
    );

    let app = axum::Router::new()
        .route("/mcp", axum::routing::any({
            let service = mcp_service.clone();
            move |req: axum::extract::Request| {
                let service = service.clone();
                async move {
                    service.handle(req).await
                }
            }
        }))
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("MCP HTTP server listening on http://{}/mcp", addr);
    axum::serve(listener, app).await?;
    Ok(())
}
