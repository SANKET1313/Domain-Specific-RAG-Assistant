"""Unit and security tests for NVIDIA NIM embeddings integration."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from app.core.config import get_settings
from app.embeddings.factory import get_embedder
from app.embeddings.nvidia_embeddings import NVIDIAEmbedder, _normalize_model_name


def test_model_name_normalization():
    """Verify that bare model name is prefixed with nvidia/ for the NIM endpoint."""
    assert _normalize_model_name("nemotron-3-embed-1b") == "nvidia/nemotron-3-embed-1b"
    assert _normalize_model_name("nvidia/nemotron-3-embed-1b") == "nvidia/nemotron-3-embed-1b"


def test_rejects_missing_or_placeholder_key():
    """Verify that missing key or placeholders are rejected in strict mode."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": ""}):
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
            NVIDIAEmbedder(api_key="", strict=True)

    with patch.dict(os.environ, {"NVIDIA_API_KEY": "YOUR_NEW_NVIDIA_API_KEY"}):
        get_settings.cache_clear()
        with pytest.raises(RuntimeError, match="still set to placeholder"):
            NVIDIAEmbedder(api_key="YOUR_NEW_NVIDIA_API_KEY", strict=True)


def test_fallback_mock_mode_when_placeholder():
    """Verify that placeholder key enters resilient fallback mode without crashing."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "YOUR_NEW_NVIDIA_API_KEY"}):
        get_settings.cache_clear()
        embedder = NVIDIAEmbedder(api_key="YOUR_NEW_NVIDIA_API_KEY")
        assert embedder.is_mock is True
        assert embedder.dim == 2048
        res = embedder.embed_texts(["Test chunk"])
        assert len(res.vectors[0]) == 2048
        q = embedder.embed_query("Test query")
        assert len(q) == 2048


def test_dimension_is_2048():
    """Verify default dimension for nemotron-3-embed-1b is 2048."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test-dummy-key"}):
        get_settings.cache_clear()
        embedder = NVIDIAEmbedder(api_key="nvapi-test-dummy-key", model="nemotron-3-embed-1b")
        assert embedder.dim == 2048


def test_embed_texts_and_query_input_types():
    """Verify embed_texts passes input_type='passage' and embed_query passes input_type='query'."""
    dummy_key = "nvapi-test-secret-key-12345"
    with patch.dict(os.environ, {"NVIDIA_API_KEY": dummy_key}):
        get_settings.cache_clear()
        embedder = NVIDIAEmbedder(api_key=dummy_key, model="nemotron-3-embed-1b")

        # Mock the underlying OpenAI embeddings.create call
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.05] * 2048)]
        mock_resp.usage.total_tokens = 42

        embedder._client.embeddings.create = MagicMock(return_value=mock_resp)

        # Test chunk/document embedding (should use passage)
        res = embedder.embed_texts(["Chunk document text."])
        embedder._client.embeddings.create.assert_called_with(
            model="nvidia/nemotron-3-embed-1b",
            input=["Chunk document text."],
            extra_body={"input_type": "passage"},
        )
        assert len(res.vectors[0]) == 2048
        assert res.token_count == 42

        # Test query embedding (should use query)
        q_vec = embedder.embed_query("User search query?")
        embedder._client.embeddings.create.assert_called_with(
            model="nvidia/nemotron-3-embed-1b",
            input=["User search query?"],
            extra_body={"input_type": "query"},
        )
        assert len(q_vec) == 2048


def test_api_key_redaction_in_errors():
    """Security verification: Verify that API keys are never leaked in error messages."""
    secret_key = "nvapi-super-confidential-secret-key-999"
    with patch.dict(os.environ, {"NVIDIA_API_KEY": secret_key}):
        get_settings.cache_clear()
        embedder = NVIDIAEmbedder(api_key=secret_key, model="nemotron-3-embed-1b")

        # Simulate client exception that echoes back the secret key
        embedder._client.embeddings.create = MagicMock(
            side_effect=Exception(f"Failed request with key {secret_key}")
        )

        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed_texts(["Document text"])

        err_str = str(exc_info.value)
        assert secret_key not in err_str, "CRITICAL SECURITY BREACH: Secret API key was leaked in error message!"
        assert "[REDACTED_API_KEY]" in err_str


def test_factory_returns_nvidia_embedder():
    """Verify factory returns NVIDIAEmbedder when EMBEDDING_PROVIDER=nvidia."""
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "nvidia", "NVIDIA_API_KEY": "nvapi-test-dummy"}):
        get_settings.cache_clear()
        get_embedder.cache_clear()
        embedder = get_embedder()
        assert isinstance(embedder, NVIDIAEmbedder)
        assert embedder.dim == 2048
