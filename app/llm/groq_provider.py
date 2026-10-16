"""Groq LLM provider for high-speed Llama model inference."""
from __future__ import annotations

import logging
from typing import Optional

from app.core.config import get_settings
from app.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLM):
    """Groq API provider for running Llama models (e.g. llama-3.3-70b-versatile)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.groq_api_key
        self._model = model or settings.groq_model or "llama-3.3-70b-versatile"
        self._base_url = (base_url or settings.groq_base_url or "https://api.groq.com/openai/v1").rstrip("/")
        self._client = None

        if self._api_key and self._api_key.strip():
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key.strip(),
            )
        else:
            logger.info("GROQ_API_KEY is not configured; initialized in unconfigured state.")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        if self._client is None:
            settings = get_settings()
            key = self._api_key or settings.groq_api_key
            if not key or not key.strip():
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Please add GROQ_API_KEY to your .env file to use the Groq provider."
                )
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=key.strip(),
            )

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            model=self._model,
        )
