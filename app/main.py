"""FastAPI application entry point.

Wires the routers together and configures structured logging. All business
logic lives in the app package, not in the frontend.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_documents,
    routes_metrics,
    routes_query,
    routes_upload,
)
from app.core.config import get_settings
from app.core.logging_config import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=f"{settings.domain_name} — RAG API",
    description="Domain-specific Retrieval-Augmented Generation backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_upload.router)
app.include_router(routes_query.router)
app.include_router(routes_documents.router)
app.include_router(routes_metrics.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.domain_name, "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.domain_name}


@app.get("/_stcore/health")
def stcore_health() -> dict[str, str]:
    """Compatibility alias for Streamlit health check path on Render."""
    return {"status": "ok", "service": settings.domain_name}

