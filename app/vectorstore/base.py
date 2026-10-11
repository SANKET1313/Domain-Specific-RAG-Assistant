"""Vector store abstraction.

A vector store persists chunk vectors + metadata and supports similarity
search. Qdrant is the default; Pinecone is provided to prove the seam.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(abc.ABC):
    @abc.abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Create the collection/index if it does not exist."""

    @abc.abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abc.abstractmethod
    def search(
        self, vector: list[float], top_k: int, flt: Optional[dict[str, Any]] = None
    ) -> list[SearchHit]: ...

    @abc.abstractmethod
    def delete_by_document(self, document_id: str) -> None: ...

    @abc.abstractmethod
    def list_documents(self) -> list[dict[str, Any]]:
        """Return aggregated per-document metadata (id, name, chunk count...)."""

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """Return all chunk payloads stored for a specific document."""
        return []

    def get_section_chunks(
        self,
        section_id: str = "",
        section_heading: str = "",
        document_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return all chunk payloads stored for a specific section in sequential order."""
        return []

    def get_adjacent_chunks(
        self,
        document_id: str,
        chunk_position: int,
        window: int = 1,
    ) -> list[dict[str, Any]]:
        """Return adjacent chunk payloads for a given chunk position in a document."""
        return []

    def delete_all(self) -> None:
        """Delete all records from the vector store."""
        pass
