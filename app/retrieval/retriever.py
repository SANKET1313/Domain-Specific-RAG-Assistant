"""Retrieval: clean query -> embed -> vector search -> candidate chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import get_settings
from app.embeddings.factory import get_embedder
from app.retrieval.query_understanding import analyze_query
from app.vectorstore.factory import get_vector_store


@dataclass
class Candidate:
    chunk_id: str
    document_name: str
    page_number: Any
    text: str
    score: float
    metadata: dict[str, Any]
    page_start: int = 1
    page_end: int = 1
    chunk_position: int = 0


def clean_query(query: str) -> str:
    """Normalize the query for embedding while preserving meaningful technical symbols."""
    return analyze_query(query).cleaned_query


def retrieve(query: str, flt: Optional[dict[str, Any]] = None) -> list[Candidate]:
    """Embed the query and return candidate chunks from the vector store with section expansion."""
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    analysis = analyze_query(query)
    vector = embedder.embed_query(analysis.cleaned_query)
    hits = store.search(vector, top_k=settings.retrieval_candidates, flt=flt)

    candidates: list[Candidate] = []
    for h in hits:
        md = h.metadata or {}
        p_num = md.get("page_number", 1)
        p_start = md.get("page_start", p_num)
        p_end = md.get("page_end", p_start)
        pos = md.get("chunk_position", md.get("index", 0))

        candidates.append(
            Candidate(
                chunk_id=md.get("chunk_id", h.id),
                document_name=md.get("document_name", "unknown"),
                page_number=p_num,
                text=md.get("chunk_text", ""),
                score=h.score,
                metadata=md,
                page_start=p_start,
                page_end=p_end,
                chunk_position=pos,
            )
        )

    # Heading-aware section expansion
    target_sec_id = analysis.target_section_id
    target_doc_id = flt.get("document_id") if flt else None

    if not target_sec_id and candidates:
        top_cand = candidates[0]
        top_sec = top_cand.metadata.get("section_id")
        top_sec_h = top_cand.metadata.get("section_heading", "")
        if top_sec and top_sec_h:
            norm_q = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
            norm_h = " ".join(re.findall(r"[a-z0-9]+", top_sec_h.lower()))
            if (norm_h and norm_h in norm_q) or (norm_q and norm_q in norm_h):
                target_sec_id = top_sec
                target_doc_id = top_cand.metadata.get("document_id")
            else:
                q_words = set(norm_q.split())
                h_words = set(norm_h.split())
                overlap = len(q_words & h_words)
                if overlap >= 3 and overlap / max(1, len(h_words)) >= 0.5:
                    target_sec_id = top_sec
                    target_doc_id = top_cand.metadata.get("document_id")

    if target_sec_id:
        if not target_doc_id and candidates:
            for c in candidates:
                if c.metadata.get("section_id") == target_sec_id:
                    target_doc_id = c.metadata.get("document_id")
                    break

        sec_chunks = store.get_section_chunks(section_id=target_sec_id, document_id=target_doc_id)
        if sec_chunks:
            sec_candidates: list[Candidate] = []
            seen_ids = set()
            for idx, sc in enumerate(sec_chunks):
                cid = sc.get("chunk_id", f"sec_chunk_{idx}")
                seen_ids.add(cid)
                p_num = sc.get("page_number", 1)
                p_start = sc.get("page_start", p_num)
                p_end = sc.get("page_end", p_start)
                pos = sc.get("chunk_position", sc.get("section_chunk_index", idx))

                sec_candidates.append(
                    Candidate(
                        chunk_id=cid,
                        document_name=sc.get("document_name", "unknown"),
                        page_number=p_num,
                        text=sc.get("chunk_text", ""),
                        score=round(0.95 - (0.001 * idx), 4),
                        metadata=sc,
                        page_start=p_start,
                        page_end=p_end,
                        chunk_position=pos,
                    )
                )

            other_candidates = [c for c in candidates if c.chunk_id not in seen_ids]
            candidates = sec_candidates + other_candidates

    return candidates
