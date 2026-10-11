"""Factory that returns the configured vector store."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.vectorstore.base import BaseVectorStore


@lru_cache
def get_vector_store() -> BaseVectorStore:
    provider = get_settings().vector_provider.lower()
    if provider == "qdrant":
        from app.vectorstore.qdrant_store import QdrantStore

        return QdrantStore()
    if provider == "pinecone":
        from app.vectorstore.pinecone_store import PineconeStore

        return PineconeStore()
    raise ValueError(f"Unknown VECTOR_PROVIDER: {provider}")
