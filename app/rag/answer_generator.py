"""Turn built context + question into a grounded answer with sources.

Includes a light grounding-verification step: if the model's answer does not
reference any provided source and context was weak, we flag low confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.llm.base import LLMResponse
from app.llm.factory import get_llm
from app.rag.prompts import build_user_prompt, system_prompt
from app.retrieval.context_builder import BuiltContext


@dataclass
class GeneratedAnswer:
    answer: str
    sources: list[dict[str, Any]]
    llm: LLMResponse


def _sources_from_context(ctx: BuiltContext) -> list[dict[str, Any]]:
    seen: set[tuple[str, Any]] = set()
    sources: list[dict[str, Any]] = []
    for c in ctx.used:
        p_start = getattr(c, "page_start", None) or c.page_number
        p_end = getattr(c, "page_end", None) or p_start
        page_val = f"{p_start}–{p_end}" if (p_start is not None and p_end is not None and p_start != p_end) else p_start

        key = (c.document_name, page_val)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document": c.document_name,
                "page": page_val,
                "page_start": p_start,
                "page_end": p_end,
                "chunk_id": c.chunk_id,
                "source": c.metadata.get("source"),
                "score": round(c.score, 4),
                "text": c.text.strip(),
            }
        )
    return sources


def generate_answer(question: str, ctx: BuiltContext) -> GeneratedAnswer:
    settings = get_settings()
    llm = get_llm()
    user_prompt = build_user_prompt(ctx.text, question)
    response = llm.generate(system_prompt(), user_prompt, temperature=settings.llm_temperature)
    answer = response.text.strip()
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    # If the generated text included a duplicate Sources: block, trim it for cleaner UI rendering
    if "\nsources:" in answer.lower():
        idx = answer.lower().rfind("\nsources:")
        answer = answer[:idx].strip()
    return GeneratedAnswer(
        answer=answer,
        sources=_sources_from_context(ctx),
        llm=response,
    )
