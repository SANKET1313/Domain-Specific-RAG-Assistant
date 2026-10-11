"""NVIDIA NIM embeddings provider (nemotron-3-embed-1b, dim=2048).

Integrates NVIDIA's NIM / OpenAI-compatible API for embeddings:
- input_type="passage" for indexing document chunks
- input_type="query" for search queries
- dimension 2048 for nemotron-3-embed-1b
- Automatic local deterministic fallback if NVIDIA_API_KEY is placeholder/missing,
  allowing full-fledged end-to-end testing without 500 errors.
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
from typing import Optional

from app.core.config import get_settings
from app.embeddings.base import BaseEmbedder, EmbeddingResult

logger = logging.getLogger(__name__)

# Known output dimensions for NVIDIA embedding models.
_DIMS = {
    "nemotron-3-embed-1b": 2048,
    "nvidia/nemotron-3-embed-1b": 2048,
    "NV-Embed-QA": 4096,
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
}

_PLACEHOLDER_KEYS = {
    "YOUR_NEW_NVIDIA_API_KEY",
    "your_nvidia_api_key_here",
    "YOUR_NVIDIA_API_KEY",
    "test-nvidia-key",
    "test-key",
    "nvapi-test-dummy",
    "dummy-key",
}


def _normalize_model_name(model: str) -> str:
    """Ensure model name has appropriate prefix for NVIDIA NIM endpoint."""
    if "/" not in model:
        return f"nvidia/{model}"
    return model


def _deterministic_mock_vector(text: str, dim: int = 2048) -> list[float]:
    """Generate deterministic, unit-normalized 2048-dim vector for local offline testing."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]


class NVIDIAEmbedder(BaseEmbedder):
    """Embedder using NVIDIA NIM API (nemotron-3-embed-1b).

    Attributes:
        dim: Vector output dimension (2048).
        is_mock: True if running in local fallback mode.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.nvidia_api_key
        self._raw_model = (
            model
            or settings.nvidia_model
            or settings.embedding_model
            or "nemotron-3-embed-1b"
        )
        self._model = _normalize_model_name(self._raw_model)
        self._base_url = (settings.nvidia_base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
        self._dim = (
            _DIMS.get(self._raw_model)
            or _DIMS.get(self._model)
            or settings.embedding_dim
            or 2048
        )

        is_missing = not self._api_key or not self._api_key.strip()
        is_placeholder = bool(self._api_key and self._api_key.strip() in _PLACEHOLDER_KEYS)

        if (is_missing or is_placeholder) and strict:
            if is_missing:
                raise RuntimeError("NVIDIA_API_KEY is not set. Please add NVIDIA_API_KEY to your .env file.")
            raise RuntimeError(
                "NVIDIA_API_KEY is still set to placeholder. Please replace it with your real NVIDIA API key in .env."
            )

        if is_missing or is_placeholder:
            self._is_mock = True
            self._client = None
            logger.warning(
                "NVIDIA_API_KEY is not set or still set to placeholder. "
                "Running in local deterministic fallback mode (dim=2048). "
                "Add your real NVIDIA API key to .env to connect to live NVIDIA NIM API."
            )
        else:
            self._is_mock = False
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key.strip(),
            )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._raw_model

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def _sanitize_error(self, err: Exception) -> str:
        """Strip sensitive credentials from error strings before propagation."""
        msg = str(err)
        if self._api_key and self._api_key in msg:
            msg = msg.replace(self._api_key, "[REDACTED_API_KEY]")
        return msg

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Embed document chunks with input_type='passage' for indexing."""
        if not texts:
            return EmbeddingResult(vectors=[], token_count=0)

        # Normalize texts while preserving exact 1:1 index alignment
        valid_texts = [t.strip() if (t and t.strip()) else "empty" for t in texts]

        if self._is_mock or self._client is None:
            vectors = [_deterministic_mock_vector(t, self._dim) for t in valid_texts]
            tokens = sum(len(t.split()) for t in valid_texts)
            return EmbeddingResult(vectors=vectors, token_count=tokens)

        # Batch in chunks of 32 to stay well within API token & body limits
        batch_size = 32
        all_vectors: list[list[float]] = []
        total_tokens = 0

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i : i + batch_size]
            try:
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    extra_body={"input_type": "passage"},
                )
                for item in resp.data:
                    vec = item.embedding
                    if len(vec) != self._dim:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {self._dim}, received {len(vec)} from {self._model}."
                        )
                    if not all(math.isfinite(x) for x in vec):
                        raise ValueError("Embedding vector contains NaN or Inf values.")
                    all_vectors.append(vec)

                if resp.usage:
                    total_tokens += getattr(resp.usage, "total_tokens", 0)
            except Exception as e:
                safe_err = self._sanitize_error(e)
                logger.error("Failed to generate embeddings from NVIDIA NIM: %s", safe_err)
                raise RuntimeError(f"NVIDIA embedding generation failed: {safe_err}") from None

        return EmbeddingResult(vectors=all_vectors, token_count=total_tokens)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with input_type='query' for similarity retrieval."""
        clean = text.strip() if text else ""
        if not clean:
            raise ValueError("Query string cannot be empty for embedding.")

        if self._is_mock or self._client is None:
            return _deterministic_mock_vector(clean, self._dim)

        try:
            resp = self._client.embeddings.create(
                model=self._model,
                input=[clean],
                extra_body={"input_type": "query"},
            )
            vec = resp.data[0].embedding
            if len(vec) != self._dim:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self._dim}, received {len(vec)} from {self._model}."
                )
            if not all(math.isfinite(x) for x in vec):
                raise ValueError("Query embedding vector contains NaN or Inf values.")
            return vec
        except Exception as e:
            safe_err = self._sanitize_error(e)
            logger.error("Failed to generate query embedding from NVIDIA NIM: %s", safe_err)
            raise RuntimeError(f"NVIDIA query embedding failed: {safe_err}") from None
