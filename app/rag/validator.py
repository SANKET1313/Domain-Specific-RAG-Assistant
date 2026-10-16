"""Validation of generated answers against retrieved context and sources.

Implements Section 13.F requirements:
1. Check that the answer is directly supported by the retrieved context.
2. Verify that every cited page number and document name corresponds to a
   retrieved chunk actually provided in the context.
3. Flag or remove any hallucinated source references.
4. If the retrieved context is genuinely insufficient after secondary retrieval,
   clearly state that the information was not found without making up an answer.
"""
from __future__ import annotations

import re
from typing import Any

from app.retrieval.context_builder import BuiltContext


def validate_answer_evidence(
    answer: str,
    sources: list[dict[str, Any]],
    ctx: BuiltContext,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Validate answer and sources against retrieved context.

    Returns:
        (validated_answer, validated_sources, validation_report)
    """
    if not ctx or not ctx.used:
        # No context used
        return (
            answer,
            [],
            {
                "is_supported": False,
                "grounding_score": 0.0,
                "citations_valid": True,
                "hallucinated_citations": [],
                "valid_sources_count": 0,
            },
        )

    # 1. Build authoritative sets of valid documents and page numbers from context
    valid_docs: set[str] = set()
    valid_pages_by_doc: dict[str, set[int]] = {}
    valid_all_pages: set[int] = set()

    for c in ctx.used:
        doc_name = (c.document_name or "").lower().strip()
        if doc_name:
            valid_docs.add(doc_name)
            if doc_name not in valid_pages_by_doc:
                valid_pages_by_doc[doc_name] = set()

        p_start = getattr(c, "page_start", None) or getattr(c, "page_number", 1)
        p_end = getattr(c, "page_end", None) or p_start
        try:
            p_start_int = int(p_start)
            p_end_int = int(p_end)
            for p in range(p_start_int, p_end_int + 1):
                valid_all_pages.add(p)
                if doc_name:
                    valid_pages_by_doc[doc_name].add(p)
        except (ValueError, TypeError):
            pass

    # 2. Check sources list against valid documents and pages
    validated_sources: list[dict[str, Any]] = []
    hallucinated_sources: list[dict[str, Any]] = []

    for s in sources:
        doc = str(s.get("document", "")).lower().strip()
        p_start = s.get("page_start") or s.get("page")
        is_doc_valid = (not doc) or (doc in valid_docs) or any(doc in vd or vd in doc for vd in valid_docs)
        
        is_page_valid = True
        if p_start is not None:
            try:
                p_int = int(str(p_start).split("–")[0].split("-")[0].strip())
                is_page_valid = p_int in valid_all_pages
            except (ValueError, TypeError):
                is_page_valid = True

        if is_doc_valid and is_page_valid:
            validated_sources.append(s)
        else:
            hallucinated_sources.append(s)

    # 3. Detect cited page numbers and document names directly inside the answer text
    # e.g., "Page 15", "Pages 12-14", "page 99"
    hallucinated_in_text: list[str] = []
    cited_pages = re.findall(r"\bpages?\s+(\d+)(?:\s*[\-–]\s*(\d+))?", answer, re.IGNORECASE)
    for p1_str, p2_str in cited_pages:
        try:
            p1 = int(p1_str)
            if p1 not in valid_all_pages:
                hallucinated_in_text.append(f"Page {p1}")
            if p2_str:
                p2 = int(p2_str)
                if p2 not in valid_all_pages:
                    hallucinated_in_text.append(f"Page {p2}")
        except (ValueError, TypeError):
            pass

    # Check for cited filenames in text (e.g., something.pdf, something.docx)
    cited_files = re.findall(r"\b([a-zA-Z0-9_\-]+\.(?:pdf|docx|txt))\b", answer, re.IGNORECASE)
    for fn in cited_files:
        fn_lower = fn.lower()
        if not any(fn_lower == vd or fn_lower in vd or vd in fn_lower for vd in valid_docs):
            hallucinated_in_text.append(f"Document {fn}")

    # 4. Check grounding / evidence support
    # Extract substantive tokens (len >= 4, alphanumeric) from answer and check against context
    ctx_lower = ctx.text.lower()
    answer_tokens = [
        t for t in re.findall(r"\b[a-zA-Z0-9]{4,}\b", answer.lower())
        if t not in {
            "this", "that", "these", "those", "from", "with", "about",
            "which", "there", "their", "where", "would", "could", "should",
            "state", "states", "document", "documents", "according", "provided",
            "answer", "based", "section", "page", "pages",
        }
    ]

    if answer_tokens:
        matched = sum(1 for t in answer_tokens if t in ctx_lower)
        grounding_score = round(matched / len(answer_tokens), 4)
    else:
        grounding_score = 1.0

    # Numbers / quantities check: numbers in answer should ideally appear in context
    answer_numbers = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", answer)
    unsupported_numbers = [n for n in answer_numbers if n not in ctx.text and n not in {"1", "2", "3", "4", "5", "10"}]

    all_hallucinations = [str(s) for s in hallucinated_sources] + hallucinated_in_text
    is_supported = (grounding_score >= 0.35 or len(answer_tokens) <= 5) and (len(hallucinated_in_text) == 0)

    # 5. Extract verbatim supporting evidence passage from context
    supporting_evidence: str | None = None
    primary_doc_id: str | None = None
    primary_doc_name: str | None = None
    source_pages: list[Any] = []

    # Check for insufficient context indicators in answer
    is_insufficient_text = any(
        phrase in answer.lower()
        for phrase in (
            "could not find sufficient",
            "could not be established",
            "does not contain sufficient",
            "insufficient information",
            "no evidence found",
            "not found in the indexed documents",
        )
    )

    if not is_supported or is_insufficient_text or not validated_sources:
        insufficient_evidence = True
        supporting_evidence = None
        if not is_supported and len(answer_tokens) > 5:
            answer = (
                "The retrieved passages do not contain sufficient evidence to answer "
                "this question reliably."
            )
            validated_sources = []
    else:
        insufficient_evidence = False
        # Find best supporting sentence / snippet across retrieved chunks
        best_sentence = ""
        best_overlap = -1
        target_tokens = set(answer_tokens)

        for c in ctx.used:
            doc_id = c.metadata.get("document_id")
            doc_name = c.document_name
            # Split into sentence candidates
            sentences = re.split(r"(?<=[.!?])\s+|\n+", c.text.strip())
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_tokens = set(re.findall(r"\b[a-zA-Z0-9]{4,}\b", s_clean.lower()))
                overlap = len(target_tokens & s_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s_clean
                    primary_doc_id = doc_id
                    primary_doc_name = doc_name

        if best_sentence:
            supporting_evidence = best_sentence
        elif ctx.used:
            # Fallback to the first chunk's leading text
            first_chunk = ctx.used[0]
            supporting_evidence = first_chunk.text.strip().split("\n")[0][:250]
            primary_doc_id = first_chunk.metadata.get("document_id")
            primary_doc_name = first_chunk.document_name

        # Extract unique sorted source pages
        for s in validated_sources:
            p_val = s.get("page") or s.get("page_start")
            if p_val is not None and p_val not in source_pages:
                source_pages.append(p_val)

        if not primary_doc_id and ctx.used:
            primary_doc_id = ctx.used[0].metadata.get("document_id")
        if not primary_doc_name and ctx.used:
            primary_doc_name = ctx.used[0].document_name

    validation_report = {
        "is_supported": is_supported and not insufficient_evidence,
        "grounding_score": grounding_score,
        "citations_valid": len(all_hallucinations) == 0,
        "hallucinated_citations": all_hallucinations,
        "unsupported_numbers": unsupported_numbers,
        "valid_sources_count": len(validated_sources),
        "supporting_evidence": supporting_evidence,
        "source_document_id": primary_doc_id,
        "source_document_name": primary_doc_name,
        "source_pages": source_pages,
        "insufficient_evidence": insufficient_evidence,
    }

    return answer, validated_sources, validation_report
