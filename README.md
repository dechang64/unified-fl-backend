# FedCtx

**Federated Semantic Infrastructure** — a self-contained Rust runtime for privacy-preserving, semantically-aware federated applications.

[![Rust](https://img.shields.io/badge/Rust-1.95-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-60%2F60-brightgreen.svg)]()

---

## What is FedCtx?

FedCtx is a **single-binary infrastructure** that provides the core building blocks for federated intelligence systems — vector search, knowledge graphs, audit trails, memory, and federated learning — all accessible via gRPC, REST, and MCP.

The key insight: **data stays where it is; models and knowledge move.** FedCtx implements this principle at the infrastructure level, not as an afterthought.

### Why does this exist?

Existing federated learning frameworks (Flower, PySyft, FedML) focus on the training loop. But real federated systems need more than aggregation — they need:

- **Semantic recall** across distributed data (HNSW vector search)
- **Provenance & audit** for regulatory compliance (SHA-256 chain)
- **Knowledge graphs** for domain relationships
- **Memory** that respects forgetting curves (Ebbinghaus)
- **Multiple access protocols** (gRPC for services, REST for web, MCP for AI agents)

FedCtx provides all of these as a single, self-contained Rust binary with zero external dependencies at runtime.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                          │
│        MedGpsPlugin · Custom ScenarioPlugins · AI Agents           │
├─────────────────────────────────────────────────────────────────────┤
│                       ScenarioPlugin Trait                          │
├──────────┬───────────┬───────────┬──────────┬──────────────────────┤
│ VectorDB │ GraphStore│ AuditLog  │ Memory   │ EmbeddingProvider    │
│ (HNSW)   │ (JSON)    │ (SHA-256) │ (HNSW +  │ (Hash / ONNX /      │
│          │           │ + Ed25519 │ Ebbinghaus│  sentence-transform) │
│          │           │ signatures│ forgetting│                      │
│          │           │           │ curve)   │                      │
├──────────┴───────────┴───────────┴──────────┴──────────────────────┤
│                     CoreStores (trait objects)                      │
├─────────────────────────────────────────────────────────────────────┤
│  gRPC :50051  │  REST API :8080  │  MCP stdio  │  MCP HTTP :3000  │
│  (tonic)      │  (axum, 16      │  (rmcp,     │  (rmcp SSE,       │
│               │   endpoints)     │  14 tools)  │  Streamable HTTP) │
├─────────────────────────────────────────────────────────────────────┤
│              Persistence (SQLite WAL + JSON + Binary)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Capability | Implementation | Details |
|---|---|---|
| **Vector Search** | Self-implemented HNSW | f64-precision cosine distance, metadata filtering, upsert |
| **Knowledge Graph** | In-memory with adjacency index | BFS traversal, label indexing, neighbor queries |
| **Audit Trail** | SHA-256 chain + Ed25519 signatures | Anchor-based eviction, Merkle root, tamper detection |
| **Memory Store** | HNSW-backed + Ebbinghaus forgetting | Session timelines, importance boosting, retention scoring |
| **Federated Learning** | FedAvg / FedProx / EWA strategies | Differential privacy (Gaussian/Laplace), entropy-weighted aggregation |
| **GraphRAG** | Graph + Vector hybrid retrieval | Subgraph extraction + semantic ranking |
| **Embedding** | Hash (zero-dep) / ONNX Runtime | Pluggable trait, batch support |
| **Persistence** | SQLite WAL + JSON + Binary | Crash-safe writes, auto-persist interval |
| **Observability** | Prometheus metrics endpoint | Vector/graph/FL/memory/audit counters, uptime |

---

## Quick Start

### Build

```bash
# Requires Rust 1.75+ and protoc
cargo build --release
# Binary: target/release/fedctx (9.3 MB)
```

### Run

```bash
# Default: gRPC + REST servers
./fedctx --data-dir ./data

# With persistence (auto-save every 5 min)
./fedctx --data-dir ./data --auto-persist-secs 300

# MCP stdio mode (for Claude Desktop / Cursor)
./fedctx --mcp

# MCP HTTP mode (for remote AI agents)
./fedctx --mcp-http 3000

# Custom dimension + ports
./fedctx --dimension 768 --grpc-port 50052 --http-port 8081
```

### Docker

```bash
docker compose up -d
```

---

## REST API (16 endpoints)

### Core
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (lightweight) |
| GET | `/api/stats` | System statistics |
| POST | `/api/search` | Vector similarity search |
| POST | `/api/text_search` | Text → embedding → search |
| POST | `/api/vectors` | Insert vector |
| GET | `/api/audit?limit=50` | Audit trail (paginated) |

### Graph
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/graph/nodes` | Add graph node |
| POST | `/api/graph/edges` | Add graph edge |
| GET | `/api/graph/traverse?start_id=&max_hops=3` | BFS traversal |
| GET | `/api/graph/neighbors?node_id=` | Get neighbors |

### Memory
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/memory/remember` | Store memory |
| POST | `/api/memory/recall` | Semantic recall |
| GET | `/api/memory/recent?limit=10` | Recent memories |
| POST | `/api/memory/forget` | Delete memory |
| GET | `/api/memory/stats` | Memory statistics |

---

## MCP Tools (14)

Accessible via stdio or Streamable HTTP (`/mcp`):

| Category | Tools |
|----------|-------|
| **Vector** | `vector_search`, `insert_vector` |
| **Graph** | `graph_traverse`, `graph_neighbors`, `add_graph_node`, `add_graph_edge` |
| **Memory** | `memory_remember`, `memory_recall`, `memory_recent`, `memory_forget`, `memory_stats` |
| **Audit** | `audit_recent` |
| **System** | `stats`, `persist` |

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "fedctx": {
      "command": "/path/to/fedctx",
      "args": ["--mcp", "--data-dir", "/path/to/data"]
    }
  }
}
```

### Remote MCP Client

```json
{
  "mcpServers": {
    "fedctx": {
      "url": "http://your-server:3000/mcp"
    }
  }
}
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dimension` | 384 | Vector dimension |
| `--grpc-port` | 50051 | gRPC listen port |
| `--http-port` | 8080 | REST API listen port |
| `--max-audit-blocks` | 10000 | Max audit chain blocks |
| `--data-dir` | None | Persistence directory |
| `--auto-persist-secs` | 300 | Auto-persist interval (0 = disabled) |
| `--embedding-model-dir` | None | ONNX sentence-transformers model path |
| `--no-med-gps` | false | Disable MedGps scenario plugin |
| `--mcp` | false | Run as MCP stdio server |
| `--mcp-http <port>` | - | Run as MCP Streamable HTTP server |

---

## Project Structure

```
fedctx/
├── src/
│   ├── main.rs              # Entry point (3 modes: gRPC+REST / MCP stdio / MCP HTTP)
│   ├── runtime.rs           # FedCtxRuntime — plugin registry + server bootstrap
│   ├── traits.rs            # Core trait definitions (VectorStore, GraphStore, etc.)
│   ├── vector_db.rs         # HNSW vector database (f64 cosine, metadata, upsert)
│   ├── json_graph_store.rs  # In-memory graph with adjacency + label indexes
│   ├── audit.rs             # SHA-256 audit chain with Ed25519 signatures
│   ├── memory_store.rs      # HNSW-backed memory with Ebbinghaus forgetting
│   ├── fl_engine.rs         # FL aggregation (FedAvg/FedProx/EWA) + DP
│   ├── embedding.rs         # Pluggable embedding (Hash / ONNX)
│   ├── graphrag.rs          # GraphRAG hybrid retrieval
│   ├── rest_api.rs          # 16 REST endpoints (axum)
│   ├── mcp_server.rs        # 14 MCP tools (rmcp)
│   ├── grpc_service.rs      # gRPC service definitions
│   ├── storage.rs           # SQLite persistence layer
│   ├── metrics.rs           # Prometheus metrics
│   ├── config.rs            # TOML configuration
│   └── med_gps_*.rs         # MedGps scenario plugin
├── proto/
│   └── federated_ai.proto   # gRPC service definitions
├── python/sdk/              # Python client SDK
├── config/                  # Default configuration
├── deploy/                  # Cloud deployment scripts
├── tests/                   # Integration tests
└── docs/                    # Architecture diagrams
```

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Runtime | Rust 1.95 | Zero-cost abstractions, memory safety |
| gRPC | tonic 0.13 + prost | High-performance RPC |
| HTTP | axum 0.8 | Async REST framework |
| Vector Search | hnsw 0.11 | Self-implemented HNSW index |
| Crypto | sha2 + ed25519-dalek | Audit chain integrity |
| MCP | rmcp 1.7 | Model Context Protocol (stdio + HTTP) |
| Persistence | rusqlite 0.32 | SQLite WAL mode |
| Concurrency | parking_lot | Non-poisoning RwLock |

---

## Design Principles

1. **Data stays, knowledge moves** — the core tenet of federated intelligence
2. **Core capabilities are self-implemented** — HNSW, audit chain, memory; not outsourced to external services
3. **Trait-based abstraction** — swap implementations without changing application code
4. **Graceful degradation** — HashEmbedder when ONNX is unavailable; template routes when LLM is down
5. **Audit by default** — every mutation is recorded in a tamper-evident chain
6. **Single binary** — no external services required at runtime

---

## Testing

```bash
# Unit + integration tests
cargo test

# With clippy linting
cargo clippy && cargo test

# Release build
cargo build --release
```

**Current status: 60/60 tests passing, 0 clippy warnings.**

---

## License

[Apache-2.0](LICENSE)

---

## Citation

If you use FedCtx in your research, please cite:

```bibtex
@software{fedctx2026,
  title = {FedCtx: Federated Semantic Infrastructure},
  author = {Dechang Yang},
  year = {2026},
  url = {https://github.com/dechang64/unified-fl-backend}
}
```
