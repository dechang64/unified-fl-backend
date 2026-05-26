"""
FedCtx Python SDK
=================
Official Python client for FedCtx — Federated Semantic Infrastructure.

Supports:
  - VectorDB: insert, search, delete, stats
  - GraphStore: add_node, add_edge, traverse, neighbors
  - Memory: remember, recall, recent, forget, stats
  - FL: submit_update, aggregate, get_model, list_rounds
  - GraphRAG: query (vector + graph retrieval)
  - Embedding: embed texts
  - Audit: trail, verify
  - System: health, stats, metrics, persist

Transport: REST (default) or gRPC.
"""

import httpx
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════════
# Data Types
# ════════════════════════════════════════════════════════════════

@dataclass
class Vector:
    id: str
    values: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphNode:
    id: str
    label: str
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    label: str
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    id: str
    content: str
    role: str  # "user" | "assistant" | "system"
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    timestamp: str = ""


@dataclass
class ModelUpdate:
    client_id: str
    round: int
    parameters: List[float]
    num_samples: int
    train_loss: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class ScoredNode:
    id: str
    label: str
    properties: Dict[str, str] = field(default_factory=dict)
    vector_score: float = 0.0
    centrality_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class GraphRAGResult:
    query: str
    nodes: List[ScoredNode] = field(default_factory=list)
    edges: List[Dict[str, str]] = field(default_factory=list)
    total_candidates: int = 0
    total_expanded: int = 0
    retrieval_time_ms: float = 0.0


# ════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════

class FedCtxClient:
    """Python client for FedCtx backend.

    Usage:
        with FedCtxClient() as client:
            client.insert([Vector("v1", [0.1, 0.2, ...])])
            hits = client.search([0.1, 0.2, ...], k=5)
    """

    def __init__(
        self,
        rest_addr: str = "http://localhost:8080",
        timeout: float = 30.0,
    ):
        self.rest_addr = rest_addr
        self._http = httpx.Client(base_url=rest_addr, timeout=timeout)

    # ── System ──

    def health(self) -> Dict[str, Any]:
        """Check backend health."""
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        resp = self._http.get("/api/stats")
        resp.raise_for_status()
        return resp.json()

    def metrics(self) -> str:
        """Get Prometheus metrics (raw text)."""
        resp = self._http.get("/metrics")
        resp.raise_for_status()
        return resp.text

    def persist(self) -> Dict[str, Any]:
        """Trigger manual data persistence."""
        resp = self._http.post("/api/persist")
        resp.raise_for_status()
        return resp.json()

    # ── VectorDB ──

    def insert(self, vectors: List[Vector]) -> int:
        """Insert vectors into the database."""
        payload = [{"id": v.id, "values": v.values, "metadata": v.metadata} for v in vectors]
        resp = self._http.post("/api/vectors", json=payload)
        resp.raise_for_status()
        return resp.json().get("inserted", 0)

    def search(self, query: List[float], k: int = 5) -> List[SearchHit]:
        """Search for k nearest neighbors."""
        resp = self._http.post("/api/search", json={"query": query, "k": k})
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchHit(id=r["id"], score=r["score"], metadata=r.get("metadata", {}))
            for r in data.get("results", [])
        ]

    # ── GraphStore ──

    def add_node(self, node: GraphNode) -> bool:
        """Add a node to the graph."""
        resp = self._http.post("/api/graph/nodes", json={
            "id": node.id, "label": node.label, "properties": node.properties,
        })
        resp.raise_for_status()
        return resp.json().get("success", False)

    def add_edge(self, edge: GraphEdge) -> bool:
        """Add an edge to the graph."""
        resp = self._http.post("/api/graph/edges", json={
            "id": edge.id, "source": edge.source, "target": edge.target,
            "label": edge.label, "properties": edge.properties,
        })
        resp.raise_for_status()
        return resp.json().get("success", False)

    def traverse(self, start_id: str, edge_label: Optional[str] = None, max_hops: int = 3) -> Dict[str, Any]:
        """Traverse the graph from a starting node."""
        resp = self._http.post("/api/graph/traverse", json={
            "start_id": start_id, "edge_label": edge_label, "max_hops": max_hops,
        })
        resp.raise_for_status()
        return resp.json()

    # ── Memory ──

    def remember(self, content: str, role: str = "user", tags: Optional[List[str]] = None,
                 importance: float = 1.0) -> str:
        """Store a memory entry. Returns the memory ID."""
        resp = self._http.post("/api/memory/remember", json={
            "content": content, "role": role, "tags": tags or [],
            "importance": importance,
        })
        resp.raise_for_status()
        return resp.json().get("id", "")

    def recall(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recall memories by semantic similarity."""
        resp = self._http.post("/api/memory/recall", json={"query": query, "k": k})
        resp.raise_for_status()
        return resp.json().get("results", [])

    def recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memories."""
        resp = self._http.get("/api/memory/recent", params={"limit": limit})
        resp.raise_for_status()
        return resp.json().get("results", [])

    def forget(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        resp = self._http.delete(f"/api/memory/{memory_id}")
        resp.raise_for_status()
        return resp.json().get("success", False)

    # ── Federated Learning ──

    def submit_update(self, update: ModelUpdate) -> Dict[str, Any]:
        """Submit a model update for FL aggregation."""
        resp = self._http.post("/api/fl/submit", json={
            "client_id": update.client_id,
            "round": update.round,
            "parameters": update.parameters,
            "num_samples": update.num_samples,
            "train_loss": update.train_loss,
            "entropy": update.entropy,
        })
        resp.raise_for_status()
        return resp.json()

    def aggregate(self, round: int, strategy: str = "fedavg",
                  min_clients: int = 2, **kwargs) -> Dict[str, Any]:
        """Trigger FL aggregation for a round."""
        payload = {
            "round": round, "strategy": strategy, "min_clients": min_clients,
        }
        payload.update(kwargs)
        resp = self._http.post("/api/fl/aggregate", json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_model(self, round: int = 0) -> Dict[str, Any]:
        """Get the global model parameters."""
        resp = self._http.get("/api/fl/model", params={"round": round})
        resp.raise_for_status()
        return resp.json()

    def list_rounds(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List FL round states."""
        resp = self._http.get("/api/fl/rounds", params={"limit": limit})
        resp.raise_for_status()
        return resp.json().get("rounds", [])

    # ── GraphRAG ──

    def graphrag_query(self, query: str, top_k: int = 10, max_hops: int = 2,
                       max_nodes: int = 30, min_score: float = 0.1) -> GraphRAGResult:
        """Query using GraphRAG: vector search + graph expansion + re-ranking."""
        resp = self._http.post("/api/graphrag/query", json={
            "query": query, "top_k": top_k, "max_hops": max_hops,
            "max_nodes": max_nodes, "min_score": min_score,
        })
        resp.raise_for_status()
        data = resp.json()
        return GraphRAGResult(
            query=data.get("query", query),
            nodes=[
                ScoredNode(
                    id=n["id"], label=n.get("label", ""),
                    properties=n.get("properties", {}),
                    vector_score=n.get("vector_score", 0),
                    centrality_score=n.get("centrality_score", 0),
                    combined_score=n.get("combined_score", 0),
                )
                for n in data.get("nodes", [])
            ],
            edges=data.get("edges", []),
            total_candidates=data.get("total_candidates", 0),
            total_expanded=data.get("total_expanded", 0),
            retrieval_time_ms=data.get("retrieval_time_ms", 0),
        )

    # ── Embedding ──

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts into vectors."""
        resp = self._http.post("/api/embed", json={"texts": texts})
        resp.raise_for_status()
        data = resp.json()
        return [r["values"] for r in data.get("results", [])]

    # ── Audit ──

    def audit_trail(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent audit trail."""
        resp = self._http.get("/api/audit", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    # ── Lifecycle ──

    def close(self):
        """Close connections."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ════════════════════════════════════════════════════════════════
# Quick Test
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    with FedCtxClient() as client:
        print("=== Health ===")
        print(json.dumps(client.health(), indent=2))

        print("\n=== Inserting 100 vectors (dim=384) ===")
        vectors = [
            Vector(
                id=f"vec_{i:04d}",
                values=[random.gauss(0, 1) for _ in range(384)],
                metadata={"source": f"client_{i % 3}", "type": "test"},
            )
            for i in range(100)
        ]
        inserted = client.insert(vectors)
        print(f"  Inserted: {inserted}")

        print("\n=== Search (k=5) ===")
        query = [random.gauss(0, 1) for _ in range(384)]
        hits = client.search(query, k=5)
        for h in hits:
            print(f"  {h.id}: score={h.score:.4f}, metadata={h.metadata}")

        print("\n=== Stats ===")
        print(json.dumps(client.stats(), indent=2))

        print("\n=== Metrics ===")
        metrics_text = client.metrics()
        for line in metrics_text.split("\n")[:5]:
            print(f"  {line}")

        print("\n✅ SDK test complete!")
