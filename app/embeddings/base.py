"""Embedding provider abstraction.

Any embedding backend implements `embed_texts` / `embed_query` plus a `dim`
property. This lets us swap OpenAI for BGE (or anything else) without touching
the ingestion or retrieval code.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    token_count: int  # best-effort token usage for cost estimation


class BaseEmbedder(abc.ABC):
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Vector dimension; must match the vector DB collection size."""

    @abc.abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of documents/chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text]).vectors[0]
