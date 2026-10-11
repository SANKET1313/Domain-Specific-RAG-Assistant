"""OpenAI chat completion provider (e.g. gpt-4o-mini)."""
from __future__ import annotations

from app.core.config import get_settings
from app.llm.base import BaseLLM, LLMResponse


class OpenAIProvider(BaseLLM):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
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
