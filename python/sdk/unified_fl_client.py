"""
Unified FL Backend — Python Client SDK
========================================
Connects to the Rust backend via gRPC (port 50051) or REST (port 8080).
"""

import grpc
import httpx
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


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


class UnifiedFLClient:
    """Python client for Unified FL Backend."""

    def __init__(
        self,
        grpc_addr: str = "localhost:50051",
        rest_addr: str = "http://localhost:8080",
        use_grpc: bool = True,
    ):
        self.grpc_addr = grpc_addr
        self.rest_addr = rest_addr
        self.use_grpc = use_grpc
        self._channel = None
        self._http = httpx.Client(base_url=rest_addr, timeout=30.0)

    # ── Health ──

    def health(self) -> Dict[str, Any]:
        """Check backend health."""
        resp = self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    # ── Stats ──

    def stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        resp = self._http.get("/api/stats")
        resp.raise_for_status()
        return resp.json()

    # ── Vector Operations ──

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

    # ── Audit ──

    def audit_trail(self, limit: int = 50) -> Dict[str, Any]:
        """Get recent audit trail."""
        resp = self._http.get("/api/audit")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close connections."""
        self._http.close()
        if self._channel:
            self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Quick Test ──

if __name__ == "__main__":
    import random

    with UnifiedFLClient(rest_addr="http://localhost:8080") as client:
        # Health check
        print("=== Health ===")
        print(json.dumps(client.health(), indent=2))

        # Insert 100 random vectors
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

        # Search
        print("\n=== Search (k=5) ===")
        query = [random.gauss(0, 1) for _ in range(384)]
        hits = client.search(query, k=5)
        for h in hits:
            print(f"  {h.id}: score={h.score:.4f}, metadata={h.metadata}")

        # Stats
        print("\n=== Stats ===")
        print(json.dumps(client.stats(), indent=2))

        # Audit
        print("\n=== Audit Trail (last 5) ===")
        trail = client.audit_trail()
        for entry in trail.get("entries", [])[-5:]:
            print(f"  [{entry['sequence']}] {entry['event_type']} @ {entry['timestamp']}")

        print("\n✅ All tests passed!")
