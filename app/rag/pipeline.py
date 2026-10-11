"""End-to-end RAG orchestration.

Implements the full context-engineering flow:

  Query
   -> Query cleaning / understanding
   -> Retrieval (vector search + section expansion)
   -> Retrieval filtering (similarity floor + de-dup)
   -> Relevance selection (top-k)
   -> Neighboring-chunk expansion (adjacent context preservation)
   -> Sufficiency check (detect insufficient context)
   -> Context assembly (structured, token-bounded, page-traceable)
   -> Prompt construction
   -> LLM generation
   -> Grounding handling (+ contradiction note)
   -> Final answer + sources
   -> Observability (structured diagnostic logs, latency, tokens, cost)
   -> Diagnostic debug mode for inspection
"""
from __future__ import annotations

import time
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.observability.metrics_store import (
    QueryRecord,
    estimate_cost,
    metrics_store,
    now_iso,
)
from app.rag.answer_generator import generate_answer
from app.rag.prompts import insufficient_context_message
from app.rag.validator import validate_answer_evidence
from app.retrieval.context_builder import build_context
from app.retrieval.query_understanding import analyze_query
from app.retrieval.reranker import filter_and_select, is_sufficient
from app.retrieval.retriever import retrieve

logger = get_logger("rag")


def answer_query(
    query: str,
    flt: Optional[dict[str, Any]] = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end RAG workflow with structured telemetry and optional debug inspection."""
    settings = get_settings()
    is_debug = debug or settings.debug_mode
    t0 = time.perf_counter()

    # --- Query Understanding ----------------------------------------------
    analysis = analyze_query(query)

    # --- Retrieval ---------------------------------------------------------
    candidates = retrieve(query, flt=flt)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # --- Reranking & Neighboring Chunk Expansion --------------------------
    selected = filter_and_select(candidates, query=query)
    top_score = selected[0].score if selected else 0.0

    # --- Insufficient Context Guard & Secondary Retrieval -----------------
    sufficient = is_sufficient(selected, query=query)
    secondary_attempted = False
    secondary_query = None

    if not sufficient:
        # Controlled secondary retrieval attempt (Section 13.C)
        alt_parts = []
        if analysis.keywords:
            focus_kws = [k for k in analysis.keywords if k not in {"explain", "describe", "details", "detail", "tell", "information", "find", "give"}]
            alt_parts.append(" ".join(focus_kws or analysis.keywords))
        for p in analysis.phrases:
            if p not in alt_parts and p != analysis.cleaned_query.lower():
                alt_parts.append(p)
        candidate_sec_query = " ".join(alt_parts).strip()

        if candidate_sec_query and candidate_sec_query.lower() != query.lower().strip():
            secondary_attempted = True
            secondary_query = candidate_sec_query
            logger.info("Executing secondary retrieval with query: '%s'", secondary_query)
            sec_candidates = retrieve(secondary_query, flt=flt)
            existing_ids = {c.chunk_id for c in candidates}
            new_candidates = [c for c in sec_candidates if c.chunk_id not in existing_ids]

            if new_candidates:
                candidates = candidates + new_candidates
                selected = filter_and_select(candidates, query=query)
                top_score = selected[0].score if selected else 0.0
                sufficient = is_sufficient(selected, query=query)

    if not sufficient:
        total_ms = (time.perf_counter() - t0) * 1000
        _record(
            query=query,
            selected=selected,
            candidates=candidates,
            top_score=top_score,
            retrieval_ms=retrieval_ms,
            llm_ms=0.0,
            total_ms=total_ms,
            input_tokens=0,
            output_tokens=0,
            embedding_tokens=len(analysis.cleaned_query.split()),
            model=settings.llm_model,
            sufficient=False,
            is_truncated=False,
            context_tokens=0,
        )

        resp: dict[str, Any] = {
            "answer": insufficient_context_message(),
            "sources": [],
            "sufficient_context": False,
            "contexts": [c.text for c in selected],
            "metrics": {
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "llm_latency_ms": 0.0,
                "total_latency_ms": round(total_ms, 2),
                "top_similarity_score": round(top_score, 4),
            },
        }

        if is_debug:
            resp["debug"] = {
                "original_query": query,
                "cleaned_query": analysis.cleaned_query,
                "query_type": analysis.query_type,
                "keywords": analysis.keywords,
                "phrases": analysis.phrases,
                "secondary_retrieval_attempted": secondary_attempted,
                "secondary_query": secondary_query,
                "candidates_count": len(candidates),
                "selected_count": len(selected),
                "candidates": [
                    {
                        "chunk_id": c.chunk_id,
                        "document": c.document_name,
                        "page_start": getattr(c, "page_start", c.page_number),
                        "page_end": getattr(c, "page_end", c.page_number),
                        "score": round(c.score, 4),
                    }
                    for c in candidates
                ],
                "selected": [
                    {
                        "chunk_id": c.chunk_id,
                        "document": c.document_name,
                        "page_start": getattr(c, "page_start", c.page_number),
                        "page_end": getattr(c, "page_end", c.page_number),
                        "score": round(c.score, 4),
                    }
                    for c in selected
                ],
                "sufficiency": False,
                "reason": "Insufficient similarity or keyword evidence found in context.",
            }

        return resp

    # --- Context Assembly -------------------------------------------------
    ctx = build_context(selected)

    # --- Answer Generation ------------------------------------------------
    t_llm = time.perf_counter()
    result = generate_answer(query, ctx)
    llm_ms = (time.perf_counter() - t_llm) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    # --- Answer & Citation Validation (Section 13.F) -----------------------
    validated_answer, validated_sources, val_report = validate_answer_evidence(
        result.answer, result.sources, ctx
    )

    emb_tokens = len(analysis.cleaned_query.split())
    cost = estimate_cost(result.llm.input_tokens, result.llm.output_tokens, emb_tokens)

    _record(
        query=query,
        selected=selected,
        candidates=candidates,
        top_score=top_score,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        total_ms=total_ms,
        input_tokens=result.llm.input_tokens,
        output_tokens=result.llm.output_tokens,
        embedding_tokens=emb_tokens,
        model=result.llm.model,
        sufficient=True,
        cost=cost,
        is_truncated=ctx.is_truncated,
        context_tokens=ctx.token_count,
    )

    resp = {
        "answer": validated_answer,
        "sources": validated_sources,
        "sufficient_context": True,
        "contexts": [c.text for c in selected],
        "validation": val_report,
        "metrics": {
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "llm_latency_ms": round(llm_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "input_tokens": result.llm.input_tokens,
            "output_tokens": result.llm.output_tokens,
            "estimated_cost_usd": cost["total_query_cost"],
            "top_similarity_score": round(top_score, 4),
            "context_tokens": ctx.token_count,
            "context_truncated": ctx.is_truncated,
        },
    }

    if is_debug:
        resp["debug"] = {
            "original_query": query,
            "cleaned_query": analysis.cleaned_query,
            "query_type": analysis.query_type,
            "keywords": analysis.keywords,
            "phrases": analysis.phrases,
            "target_section": analysis.target_section_id,
            "secondary_retrieval_attempted": secondary_attempted,
            "secondary_query": secondary_query,
            "candidates_retrieved": len(candidates),
            "candidates": [
                {
                    "chunk_id": c.chunk_id,
                    "document": c.document_name,
                    "page_start": getattr(c, "page_start", c.page_number),
                    "page_end": getattr(c, "page_end", c.page_number),
                    "score": round(c.score, 4),
                }
                for c in candidates
            ],
            "selected_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document": c.document_name,
                    "page_start": getattr(c, "page_start", c.page_number),
                    "page_end": getattr(c, "page_end", c.page_number),
                    "score": round(c.score, 4),
                }
                for c in selected
            ],
            "context_token_count": ctx.token_count,
            "context_truncated": ctx.is_truncated,
            "final_context_preview": ctx.text[:400] + ("..." if len(ctx.text) > 400 else ""),
            "full_context": ctx.text,
            "model": result.llm.model,
            "grounded": val_report.get("is_supported", True),
            "validation": val_report,
        }

    return resp


def _record(
    query: str,
    selected: list,
    candidates: list,
    top_score: float,
    retrieval_ms: float,
    llm_ms: float,
    total_ms: float,
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int,
    model: str,
    sufficient: bool,
    cost: Optional[dict] = None,
    is_truncated: bool = False,
    context_tokens: int = 0,
) -> None:
    settings = get_settings()
    cost = cost or estimate_cost(input_tokens, output_tokens, embedding_tokens)
    rec = QueryRecord(
        timestamp=now_iso(),
        query=query,
        retrieved_chunk_count=len(selected),
        top_similarity_score=round(top_score, 4),
        retrieval_latency_ms=round(retrieval_ms, 2),
        llm_latency_ms=round(llm_ms, 2),
        total_latency_ms=round(total_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        embedding_tokens=embedding_tokens,
        estimated_cost_usd=cost["total_query_cost"],
        model=model,
        prompt_version=settings.prompt_version,
        sufficient_context=sufficient,
        cost_fields=cost,
    )
    metrics_store.record(rec)

    # Diagnostic structured log
    candidate_ids = [c.chunk_id for c in candidates[:10]]
    selected_ids = [c.chunk_id for c in selected]
    page_ranges = [
        f"{getattr(c, 'page_start', c.page_number)}-{getattr(c, 'page_end', c.page_number)}"
        for c in selected
    ]

    logger.info(
        "query_served",
        extra={"extra": {
            "query": query,
            "retrieved_candidates_count": len(candidates),
            "selected_chunks_count": len(selected),
            "candidate_chunk_ids": candidate_ids,
            "selected_chunk_ids": selected_ids,
            "page_ranges": page_ranges,
            "top_similarity": rec.top_similarity_score,
            "retrieval_latency_ms": rec.retrieval_latency_ms,
            "llm_latency_ms": rec.llm_latency_ms,
            "total_latency_ms": rec.total_latency_ms,
            "input_tokens": rec.input_tokens,
            "output_tokens": rec.output_tokens,
            "context_tokens": context_tokens,
            "context_truncated": is_truncated,
            "model": rec.model,
            "sufficient_context": rec.sufficient_context,
            "estimated_cost_usd": rec.estimated_cost_usd,
        }},
    )
