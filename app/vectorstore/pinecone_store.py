"""Pinecone-backed vector store (same interface as Qdrant).

Kept intentionally minimal to demonstrate that the provider is replaceable.
"""
from __future__ import annotations

from typing import Any, Optional

from app.core.config import get_settings
from app.vectorstore.base import BaseVectorStore, SearchHit, VectorRecord


class PineconeStore(BaseVectorStore):
    def __init__(self) -> None:
        from pinecone import Pinecone

        settings = get_settings()
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index

    def ensure_collection(self, dim: int) -> None:
        from pinecone import ServerlessSpec

        existing = {i["name"] for i in self._pc.list_indexes()}
        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self._index = self._pc.Index(self._index_name)

    def _idx(self):
        if not hasattr(self, "_index"):
            self._index = self._pc.Index(self._index_name)
        return self._index

    def upsert(self, records: list[VectorRecord]) -> None:
        self._idx().upsert(
            vectors=[(r.id, r.vector, r.metadata) for r in records]
        )

    def search(
        self, vector: list[float], top_k: int, flt: Optional[dict[str, Any]] = None
    ) -> list[SearchHit]:
        res = self._idx().query(
            vector=vector, top_k=top_k, include_metadata=True, filter=flt or None
        )
        return [
            SearchHit(id=m["id"], score=m["score"], metadata=m.get("metadata", {}))
            for m in res.get("matches", [])
        ]

    def delete_by_document(self, document_id: str) -> None:
        self._idx().delete(filter={"document_id": {"$eq": document_id}})

    def delete_all(self) -> None:
        try:
            self._idx().delete(delete_all=True)
        except Exception:
            pass

    def list_documents(self) -> list[dict[str, Any]]:
        # Pinecone has no cheap scan; a production impl would track docs in a
        # side store (e.g. SQLite). Returned empty here to keep the demo simple.
        return []
