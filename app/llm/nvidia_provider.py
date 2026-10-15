"""NVIDIA NIM LLM provider.

Provides generative chat completions via NVIDIA NIM API (e.g., openai/gpt-oss-20b)
using the user's NVIDIA_API_KEY with configurable temperature (default 0.2).
Strictly adheres to security requirements: no keys logged or exposed.
"""
from __future__ import annotations

import logging
from typing import Any
import requests

from app.core.config import get_settings
from app.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


def _sanitize_error_message(msg: str, key: str | None) -> str:
    if key and key in msg:
        return msg.replace(key, "[REDACTED]")
    return msg


class NvidiaLLMProvider(BaseLLM):
    """LLM provider connecting to NVIDIA NIM chat completions API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = (settings.nvidia_api_key or "").strip()
        self._model = (
            settings.llm_model
            or (settings.nvidia_model if settings.nvidia_model and "embed" not in settings.nvidia_model.lower() else None)
            or "meta/llama-3.2-11b-vision-instruct"
        )
        self._fallback_model = getattr(settings, "llm_fallback_model", "openai/gpt-oss-20b")
        self._base_url = (settings.nvidia_base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
        self._default_temp = getattr(settings, "llm_temperature", 0.2)
        self._timeout = getattr(settings, "llm_timeout", 60)
        self._max_tokens = getattr(settings, "llm_max_tokens", 3000)

    def _call_model(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temp: float,
        timeout: int,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        endpoint = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        tokens_to_gen = max_tokens if max_tokens is not None else self._max_tokens
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temp,
            "max_tokens": tokens_to_gen,
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "") or ""
            content = (
                content.replace("\u202f", " ")
                .replace("\xa0", " ")
                .strip()
            )
            usage = data.get("usage", {})
            return LLMResponse(
                text=content,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                model=data.get("model", model_name),
            )

        safe_err = _sanitize_error_message(resp.text[:300], self._api_key)
        if resp.status_code in (401, 403):
            raise RuntimeError(f"NVIDIA NIM authentication failed: {safe_err}")
        if resp.status_code == 404:
            raise RuntimeError(f"NVIDIA model '{model_name}' not found: {safe_err}")
        raise RuntimeError(f"NVIDIA NIM chat API error (HTTP {resp.status_code}): {safe_err}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call NVIDIA NIM chat completions with the specified temperature and fallback handling."""
        temp = temperature if temperature is not None else self._default_temp

        if not self._api_key or self._api_key in ("YOUR_NEW_NVIDIA_API_KEY", "your_nvidia_api_key_here"):
            raise ValueError(
                "NVIDIA_API_KEY is not configured or is set to placeholder in .env. "
                "Please configure a valid NVIDIA API key."
            )

        # 1. Attempt generation with primary configured model
        try:
            return self._call_model(
                self._model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temp=temp,
                timeout=self._timeout,
                max_tokens=max_tokens,
            )
        except Exception as primary_exc:
            safe_err = _sanitize_error_message(str(primary_exc), self._api_key)
            logger.warning("Primary NVIDIA model '%s' failed: %s", self._model, safe_err)

            # 2. Attempt fallback model if different
            if self._fallback_model and self._fallback_model != self._model:
                try:
                    logger.info("Attempting fallback to NVIDIA model '%s'...", self._fallback_model)
                    return self._call_model(
                        self._fallback_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temp=temp,
                        timeout=self._timeout,
                        max_tokens=max_tokens,
                    )
                except Exception as fallback_exc:
                    safe_fb_err = _sanitize_error_message(str(fallback_exc), self._api_key)
                    logger.error("Fallback NVIDIA model '%s' also failed: %s", self._fallback_model, safe_fb_err)

            raise RuntimeError(f"NVIDIA NIM connection error: {safe_err}") from primary_exc
