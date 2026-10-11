"""In-memory observability + cost estimation.

Every query appends one QueryRecord. The /metrics endpoint and the Streamlit
dashboard read aggregates from here. For production you would persist these to
a database; in-memory keeps the demo dependency-free.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings


@dataclass
class QueryRecord:
    timestamp: str
    query: str
    retrieved_chunk_count: int
    top_similarity_score: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    estimated_cost_usd: float
    model: str
    prompt_version: str
    sufficient_context: bool
    cost_fields: dict[str, float] = field(default_factory=dict)


class MetricsStore:
    def __init__(self, maxlen: int = 1000) -> None:
        self._records: deque[QueryRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, rec: QueryRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(r) for r in self._records]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        if not records:
            return {
                "query_count": 0,
                "avg_total_latency_ms": None,
                "avg_retrieval_latency_ms": None,
                "avg_llm_latency_ms": None,
                "avg_estimated_cost_usd": None,
                "avg_retrieved_chunks": None,
                "insufficient_context_rate": None,
            }
        n = len(records)
        return {
            "query_count": n,
            "avg_total_latency_ms": round(sum(r.total_latency_ms for r in records) / n, 2),
            "avg_retrieval_latency_ms": round(sum(r.retrieval_latency_ms for r in records) / n, 2),
            "avg_llm_latency_ms": round(sum(r.llm_latency_ms for r in records) / n, 2),
            "avg_estimated_cost_usd": round(sum(r.estimated_cost_usd for r in records) / n, 6),
            "avg_retrieved_chunks": round(sum(r.retrieved_chunk_count for r in records) / n, 2),
            "insufficient_context_rate": round(
                sum(0 if r.sufficient_context else 1 for r in records) / n, 3
            ),
        }


def estimate_cost(input_tokens: int, output_tokens: int, embedding_tokens: int) -> dict[str, float]:
    """Estimate query cost from configurable per-1M-token pricing.

    Returns zeros when pricing is not configured (we never fabricate numbers).
    """
    s = get_settings()
    embedding_cost = embedding_tokens / 1_000_000 * s.embedding_cost_per_1m
    llm_input_cost = input_tokens / 1_000_000 * s.llm_input_cost_per_1m
    llm_output_cost = output_tokens / 1_000_000 * s.llm_output_cost_per_1m
    return {
        "embedding_cost": round(embedding_cost, 8),
        "llm_input_cost": round(llm_input_cost, 8),
        "llm_output_cost": round(llm_output_cost, 8),
        "total_query_cost": round(embedding_cost + llm_input_cost + llm_output_cost, 8),
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Single process-wide store.
metrics_store = MetricsStore()
