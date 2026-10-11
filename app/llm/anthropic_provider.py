"""Anthropic Claude provider (e.g. claude-3-haiku)."""
from __future__ import annotations

from app.core.config import get_settings
from app.llm.base import BaseLLM, LLMResponse


class AnthropicProvider(BaseLLM):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        import anthropic

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        resp = self._client.messages.create(
            model=self._model,
            system=system_prompt,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self._model,
        )
