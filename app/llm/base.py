"""LLM provider abstraction.

Every provider returns a uniform `LLMResponse` (text + token usage) so the RAG
pipeline, cost estimation, and logging never depend on a specific vendor SDK.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Generate a completion given a system and user prompt."""
