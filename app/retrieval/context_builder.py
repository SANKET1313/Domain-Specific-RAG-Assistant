"""Assemble selected chunks into a structured, token-bounded context block.

Format is predictable, page-traceable, and enforces context token/character limits:

    SOURCE 1
    Document: {name}
    Page: {page_or_range}
    Section: {section_heading}
    Chunk: {chunk_id}
    {chunk_text}
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.retrieval.retriever import Candidate


@dataclass
class BuiltContext:
    text: str
    used: list[Candidate]
    is_truncated: bool = False
    token_count: int = 0


def build_context(selected: list[Candidate]) -> BuiltContext:
    settings = get_settings()
    blocks: list[str] = []
    used: list[Candidate] = []
    total = 0
    is_truncated = False

    for i, c in enumerate(selected, start=1):
        # Format page reference
        p_start = getattr(c, "page_start", None) or c.page_number
        p_end = getattr(c, "page_end", None) or p_start
        if p_start is not None and p_end is not None and p_start != p_end:
            page_str = f"{p_start}–{p_end}"
        elif p_start is not None:
            page_str = str(p_start)
        else:
            page_str = "n/a"

        sec_h = c.metadata.get("section_heading")
        sec_line = f"Section: {sec_h}\n" if sec_h else ""

        block = (
            f"SOURCE {i}\n"
            f"[Document: {c.document_name} | Page: {page_str}{f' | Section: {sec_h}' if sec_h else ''}]\n"
            f"Document: {c.document_name}\n"
            f"Page: {page_str}\n"
            f"{sec_line}"
            f"Chunk: {c.chunk_id}\n"
            f"{c.text.strip()}\n"
        )

        if total + len(block) > settings.max_context_chars:
            if used:
                blocks.append(
                    f"[Note: Context truncated at {settings.max_context_chars} characters. "
                    "Additional sources omitted to fit model context window.]"
                )
                is_truncated = True
                break
            else:
                # If first block exceeds budget, trim it to fit
                allowed = max(100, settings.max_context_chars - 150)
                trimmed_text = c.text.strip()[:allowed] + " ... [truncated]"
                block = (
                    f"SOURCE {i}\n"
                    f"[Document: {c.document_name} | Page: {page_str}{f' | Section: {sec_h}' if sec_h else ''}]\n"
                    f"Document: {c.document_name}\n"
                    f"Page: {page_str}\n"
                    f"{sec_line}"
                    f"Chunk: {c.chunk_id}\n"
                    f"{trimmed_text}\n"
                )
                blocks.append(block)
                used.append(c)
                is_truncated = True
                break

        blocks.append(block)
        used.append(c)
        total += len(block)

    full_text = "\n".join(blocks)
    approx_tokens = max(1, int(len(full_text) / 3.8))

    return BuiltContext(
        text=full_text,
        used=used,
        is_truncated=is_truncated,
        token_count=approx_tokens,
    )
