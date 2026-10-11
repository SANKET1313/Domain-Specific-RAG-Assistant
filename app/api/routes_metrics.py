"""GET /metrics and GET /health."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.observability.metrics_store import metrics_store

router = APIRouter(tags=["metrics"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> dict:
    s = get_settings()
    return {
        "summary": metrics_store.summary(),
        "recent": metrics_store.all()[-25:],
        "config": {
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model,
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.embedding_model,
            "vector_provider": s.vector_provider,
            "top_k": s.top_k,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "prompt_version": s.prompt_version,
            "dataset_version": s.dataset_version,
            "domain": s.domain_name,
        },
    }
