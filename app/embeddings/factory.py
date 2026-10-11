"""Factory that returns the NVIDIA embedder."""
from __future__ import annotations

from functools import lru_cache

from app.embeddings.base import BaseEmbedder
from app.embeddings.nvidia_embeddings import NVIDIAEmbedder


@lru_cache
def get_embedder() -> BaseEmbedder:
    """Return the NVIDIA nemotron-3-embed-1b embedder."""
    return NVIDIAEmbedder()
