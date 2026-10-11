"""Qdrant-backed vector store (self-hostable, default provider)."""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import get_settings
from app.vectorstore.base import BaseVectorStore, SearchHit, VectorRecord

logger = logging.getLogger(__name__)


class QdrantStore(BaseVectorStore):
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        settings = get_settings()
        # Try connecting to the configured Qdrant server; fall back to
        # in-memory mode so the app works without Docker on local dev.
        try:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
            client.get_collections()  # quick connectivity check
            self._client = client
            logger.info("Connected to Qdrant server at %s", settings.qdrant_url)
        except Exception:
            try:
                import os
                os.makedirs("qdrant_storage", exist_ok=True)
                self._client = QdrantClient(path="qdrant_storage")
                logger.info("Using local persistent Qdrant storage at qdrant_storage/")
            except Exception as exc:
                logger.warning(
                    "Qdrant server at %s unreachable and local disk storage failed (%s) — using in-memory storage.",
                    settings.qdrant_url,
                    exc,
                )
                self._client = QdrantClient(":memory:")
        self._collection = settings.qdrant_collection

    def _collection_exists(self) -> bool:
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            return self._collection in existing
        except Exception:
            return False

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._collection_exists():
            try:
                coll_info = self._client.get_collection(self._collection)
                existing_dim = None
                vectors_cfg = coll_info.config.params.vectors
                if hasattr(vectors_cfg, "size"):
                    existing_dim = vectors_cfg.size
                elif isinstance(vectors_cfg, dict) and "size" in vectors_cfg:
                    existing_dim = vectors_cfg["size"]

                if existing_dim is not None and existing_dim != dim:
                    raise RuntimeError(
                        f"Qdrant collection '{self._collection}' dimension mismatch: "
                        f"existing collection has dim={existing_dim}, but active model requires dim={dim}. "
                        f"To prevent silent data loss, please migrate your data or set a new collection "
                        f"name via QDRANT_COLLECTION in your .env file."
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
        else:
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create Qdrant collection '{self._collection}': {exc}"
                ) from exc

    def upsert(self, records: list[VectorRecord]) -> None:
        from qdrant_client.models import PointStruct

        if not records:
            return

        points = [
            PointStruct(id=r.id, vector=r.vector, payload=r.metadata) for r in records
        ]
        try:
            self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise RuntimeError(f"Qdrant vector upsert failed: {exc}") from exc

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Return all chunk payloads stored for a specific document."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._collection_exists():
            return []

        chunks: list[dict[str, Any]] = []
        next_page = None
        qfilter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

        while True:
            try:
                points, next_page = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=qfilter,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=next_page,
                )
                for p in points:
                    chunks.append(p.payload or {})
                if next_page is None:
                    break
            except Exception as exc:
                logger.warning("Failed scrolling chunks for document %s: %s", document_id, exc)
                break

        return sorted(chunks, key=lambda c: c.get("chunk_id", ""))

    def get_section_chunks(
        self,
        section_id: str = "",
        section_heading: str = "",
        document_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve all chunk payloads stored for a specific section in sequential order."""
        if not self._collection_exists():
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must_conditions = []
        if document_id:
            must_conditions.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
        if section_id:
            must_conditions.append(FieldCondition(key="section_id", match=MatchValue(value=section_id)))
        elif section_heading:
            must_conditions.append(FieldCondition(key="section_heading", match=MatchValue(value=section_heading)))

        qfilter = Filter(must=must_conditions) if must_conditions else None
        chunks: list[dict[str, Any]] = []
        next_page = None

        while True:
            try:
                points, next_page = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=qfilter,
                    with_payload=True,
                    with_vectors=False,
                    limit=256,
                    offset=next_page,
                )
                for p in points:
                    chunks.append(p.payload or {})
                if next_page is None:
                    break
            except Exception as exc:
                logger.warning("Failed scrolling section chunks: %s", exc)
                break

        return sorted(
            chunks,
            key=lambda c: (
                c.get("document_id", ""),
                c.get("page_number", 0),
                c.get("section_chunk_index", 0),
                c.get("chunk_id", ""),
            ),
        )

    def get_adjacent_chunks(
        self,
        document_id: str,
        chunk_position: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """Retrieve adjacent chunk payloads for a given chunk position in a document."""
        if not self._collection_exists() or not document_id:
            return []

        doc_chunks = self.get_document_chunks(document_id)
        if not doc_chunks:
            return []

        target_positions = set(range(max(0, chunk_position - window), chunk_position + window + 1))
        neighbors = []
        for c in doc_chunks:
            pos = c.get("chunk_position", c.get("index"))
            if pos is not None and pos in target_positions:
                neighbors.append(c)

        return sorted(neighbors, key=lambda c: c.get("chunk_position", c.get("index", 0)))

    def search(
        self, vector: list[float], top_k: int, flt: Optional[dict[str, Any]] = None
    ) -> list[SearchHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._collection_exists():
            return []

        qfilter = None
        if flt:
            qfilter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in flt.items()]
            )
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
            hits = res.points
        else:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
        return [SearchHit(id=str(h.id), score=h.score, metadata=h.payload or {}) for h in hits]

    def delete_by_document(self, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        if not self._collection_exists():
            return

        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            ),
        )

    def delete_all(self) -> None:
        if not self._collection_exists():
            return
        try:
            self._client.delete_collection(self._collection)
        except Exception:
            try:
                from qdrant_client.models import Filter, FilterSelector
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=FilterSelector(filter=Filter()),
                )
            except Exception as exc:
                logger.warning("Failed to clear Qdrant collection: %s", exc)

    def list_documents(self) -> list[dict[str, Any]]:
        if not self._collection_exists():
            return []

        docs: dict[str, dict[str, Any]] = {}
        next_page = None
        while True:
            points, next_page = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            for p in points:
                md = p.payload or {}
                did = md.get("document_id")
                if not did:
                    continue
                entry = docs.setdefault(
                    did,
                    {
                        "document_id": did,
                        "document_name": md.get("document_name"),
                        "source": md.get("source"),
                        "domain": md.get("domain"),
                        "upload_time": md.get("upload_time"),
                        "chunk_count": 0,
                        "status": "indexed",
                    },
                )
                entry["chunk_count"] += 1
            if next_page is None:
                break
        return list(docs.values())
