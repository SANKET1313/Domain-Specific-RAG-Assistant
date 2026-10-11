"""Hybrid retrieval fusion (RRF) + relevance selection + sufficiency checking + neighboring chunk expansion.

Features:
- Dense semantic vector ranking.
- Sparse lexical ranking (term frequency overlap + exact phrase matching).
- Reciprocal Rank Fusion (RRF) combining dense and sparse scores.
- Threshold evaluation calibrated for both full questions and short keyword phrases.
- Near-duplicate chunk removal.
- Neighboring-chunk expansion: retrieves adjacent chunks from same document/section,
  deduplicating overlapping text and preserving page ranges.
- Detailed ranking diagnostic telemetry.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.config import get_settings
from app.retrieval.query_understanding import QueryAnalysis, analyze_query
from app.retrieval.retriever import Candidate
from app.vectorstore.factory import get_vector_store

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _compute_lexical_score(analysis: QueryAnalysis, chunk_text: str) -> float:
    """Calculate sparse / lexical matching score for a chunk."""
    if not chunk_text or not analysis.keywords:
        return 0.0

    lower_text = chunk_text.lower()
    score = 0.0

    # 1. Exact phrase match
    for p in analysis.phrases:
        if len(p) >= 3 and p in lower_text:
            score += 0.40
            break

    # 2. Keyword coverage
    matched_kw = sum(1 for kw in analysis.keywords if kw in lower_text)
    kw_coverage = matched_kw / len(analysis.keywords)
    score += 0.50 * kw_coverage

    return min(1.0, score)


def _deduplicate_overlap(base_text: str, adjacent_text: str) -> str:
    """Remove overlapping sentences between two adjacent chunks."""
    base_clean = base_text.strip()
    adj_clean = adjacent_text.strip()
    if not base_clean or not adj_clean:
        return adj_clean

    # Check for sentence overlap at boundary
    base_sentences = [s.strip() for s in base_clean.split("\n") if s.strip()]
    adj_sentences = [s.strip() for s in adj_clean.split("\n") if s.strip()]

    # If adjacent starts with something already in base_sentences, trim it
    start_idx = 0
    for idx, s in enumerate(adj_sentences[:4]):
        if s in base_clean:
            start_idx = idx + 1
        else:
            break

    if start_idx > 0 and start_idx < len(adj_sentences):
        return "\n".join(adj_sentences[start_idx:]).strip()

    return adj_clean


def expand_neighboring_chunks(
    selected: list[Candidate],
    max_total_chars: int = 8000,
) -> list[Candidate]:
    """Retrieve adjacent chunks for top relevant results from same document without crossing section boundaries."""
    settings = get_settings()
    if not settings.neighbor_expansion_enabled or not selected:
        return selected

    store = get_vector_store()
    expanded_candidates: list[Candidate] = list(selected)
    existing_ids = {c.chunk_id for c in selected}
    current_chars = sum(len(c.text) for c in selected)

    # Only expand the top highly relevant candidates (up to top 3)
    top_seeds = selected[: min(3, len(selected))]

    for seed in top_seeds:
        doc_id = seed.metadata.get("document_id")
        seed_pos = seed.metadata.get("chunk_position", seed.metadata.get("index"))
        seed_sec = seed.metadata.get("section_id")

        if not doc_id or seed_pos is None:
            continue

        try:
            neighbors = store.get_adjacent_chunks(
                document_id=doc_id,
                chunk_position=seed_pos,
                window=settings.neighbor_expansion_window,
            )
        except Exception as exc:
            logger.debug("Adjacent chunk lookup failed: %s", exc)
            continue

        for adj_meta in neighbors:
            adj_id = adj_meta.get("chunk_id")
            adj_pos = adj_meta.get("chunk_position", adj_meta.get("index"))
            if not adj_id or adj_id in existing_ids:
                continue

            # Section boundary check: do not cross into an unrelated section
            adj_sec = adj_meta.get("section_id")
            if seed_sec and adj_sec and seed_sec != adj_sec:
                continue

            adj_text = adj_meta.get("chunk_text", "")
            if not adj_text or len(adj_text.strip()) < 15:
                continue

            # Deduplicate overlap with seed text
            cleaned_adj_text = _deduplicate_overlap(seed.text, adj_text)
            if not cleaned_adj_text:
                continue

            if current_chars + len(cleaned_adj_text) > max_total_chars:
                break

            adj_p_num = adj_meta.get("page_number", 1)
            adj_p_start = adj_meta.get("page_start", adj_p_num)
            adj_p_end = adj_meta.get("page_end", adj_p_start)

            # Adjacent chunk score is linked to seed chunk with small decay
            adj_score = round(seed.score * 0.90, 4)

            cand_adj = Candidate(
                chunk_id=adj_id,
                document_name=adj_meta.get("document_name", seed.document_name),
                page_number=adj_p_num,
                text=cleaned_adj_text,
                score=adj_score,
                metadata=adj_meta,
                page_start=adj_p_start,
                page_end=adj_p_end,
                chunk_position=adj_pos or seed_pos,
            )

            # Insert adjacent chunk near seed in order
            insert_idx = expanded_candidates.index(seed) + (1 if adj_pos > seed_pos else 0)
            expanded_candidates.insert(insert_idx, cand_adj)
            existing_ids.add(adj_id)
            current_chars += len(cleaned_adj_text)

    return expanded_candidates


def filter_and_select(
    candidates: list[Candidate], query: Optional[str] = None
) -> list[Candidate]:
    """Fuse dense vector search with sparse keyword retrieval using RRF and select top-k chunks."""
    if not candidates:
        return []

    settings = get_settings()
    analysis = analyze_query(query or "")

    # 1. Compute sparse / lexical scores for all candidates
    lex_scores = [_compute_lexical_score(analysis, c.text) for c in candidates]

    # 2. Dense ranks (candidates are already sorted by vector score)
    dense_ranks = {c.chunk_id: rank for rank, c in enumerate(candidates, start=1)}

    # 3. Sparse ranks (sorted by lexical score descending)
    sparse_sorted = sorted(
        zip(candidates, lex_scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    sparse_ranks = {c.chunk_id: rank for rank, (c, _) in enumerate(sparse_sorted, start=1)}

    # 4. Reciprocal Rank Fusion (RRF)
    k = settings.rrf_k
    w_dense = settings.dense_weight
    w_sparse = settings.sparse_weight

    scored_candidates: list[tuple[float, Candidate]] = []
    for c, lex_sc in zip(candidates, lex_scores):
        r_dense = dense_ranks[c.chunk_id]
        r_sparse = sparse_ranks[c.chunk_id]

        if settings.hybrid_search_enabled:
            rrf_score = (w_dense / (k + r_dense)) + (w_sparse / (k + r_sparse))
            # Exact phrase bonus
            if lex_sc >= 0.40:
                rrf_score += 0.05
        else:
            rrf_score = c.score

        # Check floor: allow short keyword matches with lexical presence even if dense vector is dilute
        has_lex_evidence = lex_sc > 0.0
        min_floor = settings.min_similarity
        if analysis.is_short_keyword and has_lex_evidence:
            min_floor = min(min_floor, 0.05)

        # Candidate qualifies if dense score meets floor OR if lexical evidence is strong
        if c.score >= min_floor or (has_lex_evidence and c.score >= 0.04):
            # Update display score cleanly
            c.score = round(c.score + (lex_sc * 0.25), 4)
            scored_candidates.append((rrf_score, c))

    # Fallback if strict filtering dropped all but candidates have lexical or decent vector scores
    if not scored_candidates and candidates:
        for c, lex_sc in zip(candidates, lex_scores):
            if lex_sc > 0.0 or c.score >= 0.08:
                scored_candidates.append((c.score + (lex_sc * 0.20), c))

    # 5. De-duplicate near-identical chunk text
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for _, c in sorted(scored_candidates, key=lambda pair: pair[0], reverse=True):
        key = _normalize(c.text)[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    # 6. Section preservation: If section expansion matched, retain all ordered section chunks
    target_sec = analysis.target_section_id
    if not target_sec and deduped and deduped[0].metadata.get("section_id"):
        top_h = deduped[0].metadata.get("section_heading", "").lower()
        if any(p in top_h for p in analysis.phrases) or (query and query.lower()[:30] in top_h):
            target_sec = deduped[0].metadata.get("section_id")

    if target_sec:
        sec_chunks = [c for c in deduped if c.metadata.get("section_id") == target_sec]
        if sec_chunks:
            sec_chunks = sorted(
                sec_chunks,
                key=lambda c: (
                    c.metadata.get("page_number", 0),
                    c.metadata.get("section_chunk_index", 0),
                    c.chunk_id,
                ),
            )
            other_chunks = [c for c in deduped if c.metadata.get("section_id") != target_sec]
            selected = sec_chunks + other_chunks[: max(0, settings.top_k - len(sec_chunks))]
            return selected[:25]

    selected = deduped[: settings.top_k]

    # 7. Neighboring-chunk expansion for cross-chunk answers
    if settings.neighbor_expansion_enabled:
        selected = expand_neighboring_chunks(selected, max_total_chars=settings.max_context_chars)

    logger.debug(
        "RRF selection complete: query='%s', candidates=%d, selected=%d, top_score=%s",
        query, len(candidates), len(selected), selected[0].score if selected else None,
    )
    return selected


def is_sufficient(selected: list[Candidate], query: Optional[str] = None) -> bool:
    """Evaluate whether selected context contains sufficient evidence to answer the query."""
    if not selected:
        return False

    settings = get_settings()
    top_score = selected[0].score

    # 1. Section queries with retrieved section chunks pass immediately
    if query:
        analysis = analyze_query(query)
        if analysis.is_section_query and any(
            c.metadata.get("section_id") == analysis.target_section_id for c in selected
        ):
            return True

    # 2. High vector / hybrid score passes
    if top_score >= settings.sufficiency_threshold:
        return True

    # 3. If query terms are explicitly matched in the text, evaluate lexical evidence
    if query:
        analysis = analyze_query(query)
        lex_sc = _compute_lexical_score(analysis, selected[0].text)
        # For short queries or exact phrase matches, lower threshold avoids false rejections
        if lex_sc >= 0.40 and top_score >= 0.06:
            return True
        if lex_sc > 0.0 and top_score >= 0.08:
            return True

    return False
