# Unified FL Backend

Self-implemented Rust backend for federated intelligence applications.

## Quick Start

```bash
# Docker (recommended)
docker compose up -d

# Verify
curl http://localhost:8080/health
curl http://localhost:8080/api/stats
```

## Architecture

```
┌─────────────────────────────────────────────┐
│           unified-fl-backend                │
│                                             │
│  ┌─────────────┐  ┌──────────────────────┐  │
│  │  gRPC :50051│  │  REST API :8080      │  │
│  │  (tonic)    │  │  (axum)              │  │
│  └──────┬──────┘  └──────────┬───────────┘  │
│         │                    │              │
│  ┌──────┴────────────────────┴───────────┐  │
│  │  Core Services                        │  │
│  │  ├ HNSW VectorDB (self-implemented)   │  │
│  │  ├ SHA-256 Audit Chain                │  │
│  │  └ SQLite Persistence                 │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## API Endpoints

### REST (port 8080)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/stats` | DB statistics |
| POST | `/api/search` | Vector search |
| POST | `/api/vectors` | Insert vectors |
| GET | `/api/audit` | Audit trail |

### gRPC (port 50051)

- `VectorDatabase` service: Insert, Search, Delete, Stats
- `AuditService` service: GetTrail, Health

## Python SDK

```bash
pip install -r python/sdk/requirements.txt
python python/sdk/unified_fl_client.py
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `FL_DIMENSION` | 384 | Vector dimension |
| `FL_GRPC_PORT` | 50051 | gRPC port |
| `FL_HTTP_PORT` | 8080 | HTTP port |
| `FL_DB_PATH` | /data/vectors.db | SQLite path |
| `FL_MAX_AUDIT_BLOCKS` | 10000 | Max audit blocks |
| `RUST_LOG` | info | Log level |

## Deployment

### AWS
```bash
bash deploy/aws/deploy.sh
```

### Tencent Cloud
```bash
bash deploy/tencent/deploy.sh
```

### Docker
```bash
docker compose up -d --build
```

## Tech Stack

- **Rust 1.95** — core runtime
- **tonic 0.13** — gRPC
- **axum 0.8** — HTTP REST
- **hnsw 0.11** — self-implemented HNSW vector index
- **rusqlite 0.32** — SQLite persistence
- **sha2 0.10** — SHA-256 audit chain
