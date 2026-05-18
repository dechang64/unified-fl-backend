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

    /// SQLite database path (empty = in-memory only)
    #[arg(long)]
    db_path: Option<String>,

    /// Data directory for persistence (vectors.bin, graph.json, audit.json)
    #[arg(long)]
    data_dir: Option<String>,

    /// Disable MedGps scenario plugin
    #[arg(long)]
    no_med_gps: bool,

    /// Run as MCP stdio server (for Claude Desktop / Cursor)
    #[arg(long)]
    mcp: bool,

    /// Run as MCP Streamable HTTP server on the given port
    #[arg(long)]
    mcp_http: Option<u16>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)  // MCP uses stdout, so log to stderr
        .with_ansi(false)
        .init();

    let cli = Cli::parse();
    info!("FedCtx v{} starting...", env!("CARGO_PKG_VERSION"));

    // ── Initialize runtime ──
    let data_dir = cli.data_dir.map(PathBuf::from);
    let mut runtime = Arc::new(
        FedCtxRuntime::new(cli.dimension, cli.max_audit_blocks, data_dir)
    );

    // ── Register scenario plugins ──
    let med_gps_plugin = MedGpsPlugin::new();
    let med_gps_grpc = med_gps_plugin.create_grpc_service();
    if !cli.no_med_gps {
        Arc::get_mut(&mut runtime).unwrap().register(Box::new(med_gps_plugin));
    }

    // Initialize all plugins
    runtime.initialize_plugins().await.map_err(|e| anyhow::anyhow!("{}", e))?;
    
    // List registered plugins
    for (id, name, desc) in runtime.list_plugins() {
        info!("  Plugin: {} — {} ({})", id, name, desc);
    }

    // ── MCP stdio mode ──
    if cli.mcp {
        info!("Starting MCP stdio server...");
        let mcp_server = FedCtxMcpServer::new(
            runtime.vector_db(),
            runtime.graph_store(),
            runtime.audit_chain(),
            runtime.memory_store(),
        );
        let transport = rmcp::transport::stdio();
        let service = mcp_server.serve(transport).await.map_err(|e| {
            anyhow::anyhow!("MCP server error: {:?}", e)
        })?;
        service.waiting().await?;
        return Ok(());
    }

    // ── Get shared stores ──
    let db = runtime.vector_db();
    let audit = runtime.audit_chain();

    // ── Start gRPC server ──
    let grpc_db = db.clone();
    let grpc_audit = audit.clone();
    let grpc_addr = format!("0.0.0.0:{}", cli.grpc_port);
    let grpc_handle = tokio::spawn(async move {
        let vector_db_server = grpc_service::create_grpc_server(grpc_db.clone(), grpc_audit.clone());
        let audit_server = grpc_service::create_audit_server(grpc_audit.clone());

        let mut builder = tonic::transport::Server::builder()
            .add_service(vector_db_server)
            .add_service(audit_server);

        // Add scenario gRPC services (compile-time registration)
        if let Some(med_gps) = med_gps_grpc {
            builder = builder.add_service(med_gps);
        }

        info!("gRPC server listening on {}", grpc_addr);
        if let Err(e) = builder.serve(grpc_addr.parse().unwrap()).await {
            tracing::error!("gRPC server error: {}", e);
        }
    });

    // ── Start HTTP REST server ──
    let http_db = db.clone();
    let http_audit = audit.clone();
    let http_addr = format!("0.0.0.0:{}", cli.http_port);
    let rt = runtime.clone();
    let http_handle = tokio::spawn(async move {
        let mut app = unified_fl_backend::rest_api::create_router(http_db, http_audit);
        
        // Merge scenario REST routes
        let scenario_routes = rt.collect_rest_routes();
        app = app.merge(scenario_routes);

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
        _ = tokio::signal::ctrl_c() => info!("Received shutdown signal"),
    }

    runtime.shutdown().await;
    info!("FedCtx shut down gracefully");
    Ok(())
}
