"""Prompt loading + construction.

Prompt TEXT lives in /prompts/*.txt (versionable, editable without code change).
This module loads them and fills in the domain + context + question.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@lru_cache
def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def system_prompt() -> str:
    return _load("rag_system_prompt.txt").replace(
        "{domain_name}", get_settings().domain_name
    )


def build_user_prompt(context: str, question: str) -> str:
    template = _load("answer_prompt.txt")
    return template.replace("{context}", context).replace("{question}", question)


def insufficient_context_message() -> str:
    return (
        "I could not find sufficient information in the indexed documents to "
        "answer this question reliably."
    )
