"""Shared test configuration: set dummy env so settings load without secrets."""
import os
import pytest

from app.core.config import get_settings
from app.embeddings.factory import get_embedder

os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
os.environ.setdefault("NVIDIA_MODEL", "nemotron-3-embed-1b")
os.environ.setdefault("EMBEDDING_PROVIDER", "nvidia")
os.environ.setdefault("VECTOR_PROVIDER", "qdrant")
os.environ.setdefault("LLM_PROVIDER", "modal")


@pytest.fixture(autouse=True)
def _reset_singleton_caches():
    get_settings.cache_clear()
    get_embedder.cache_clear()
    yield
    get_settings.cache_clear()
    get_embedder.cache_clear()
