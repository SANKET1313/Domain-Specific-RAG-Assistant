"""Factory that returns the configured LLM provider."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.llm.base import BaseLLM


@lru_cache
def get_llm() -> BaseLLM:
    provider = get_settings().llm_provider.lower()
    if provider == "nvidia":
        from app.llm.nvidia_provider import NvidiaLLMProvider

        return NvidiaLLMProvider()
    if provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if provider in ("modal", "huggingface"):
        from app.llm.modal_provider import ModalProvider

        return ModalProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
