"""Tests for heading-aware section retrieval, section hierarchy, and multi-chunk preservation."""
from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    Chunk,
    chunk_pages,
    extract_heading_or_subheading,
    extract_section_header,
)
from app.retrieval.query_understanding import analyze_query, normalize_dashes
from app.retrieval.reranker import filter_and_select, is_sufficient
from app.retrieval.retriever import Candidate


def test_extract_section_header_exact_and_fuzzy():
    """Verify exact and fuzzy heading matching across dashes and case variations."""
    # Em-dash
    sid, head = extract_section_header("SECTION C — 20 Ways to Make Money Using AI (No Skills Needed)")
    assert sid == "section_c"
    assert "SECTION C" in head
    assert "20 Ways to Make Money" in head

    # En-dash
    sid, head = extract_section_header("SECTION A – Why AI Is the Great Equalizer for Beginners")
    assert sid == "section_a"
    assert "SECTION A" in head

    # Standard hyphen & lowercase
    sid, head = extract_section_header("section b - The 7-Day Fast Start Plan")
    assert sid == "section_b"
    assert "SECTION B" in head

    # Chapter / Part / Appendix
    sid, head = extract_section_header("CHAPTER 4: Advanced Fine-Tuning")
    assert sid == "chapter_4"
    assert "CHAPTER 4" in head

    sid, head = extract_section_header("PART 1 - Foundations")
    assert sid == "part_1"

    # Negative cases: TOC dot leaders and normal sentences
    sid, head = extract_section_header("SECTION C ............. 17")
    assert sid is None

    sid, head = extract_section_header("This is a regular sentence explaining how AI works.")
    assert sid is None


def test_extract_heading_or_subheading():
    """Verify method, step, and distinct title detection while rejecting body list items."""
    # Method / Step
    sub = extract_heading_or_subheading("Method 1 — Social Media Caption Packages")
    assert sub is not None
    assert "Method 1" in sub

    sub = extract_heading_or_subheading("Step 3: Setup your Gumroad account")
    assert sub is not None

    # Standalone distinct titles
    sub = extract_heading_or_subheading("The Only 3 Things You Need to Start")
    assert sub is not None
    assert "The Only 3 Things" in sub

    sub = extract_heading_or_subheading("CONCLUSION")
    assert sub == "CONCLUSION"

    # Numbered chapter-style headings
    sub = extract_heading_or_subheading("1.1 System Architecture")
    assert sub is not None

    # Rejection of body list items
    sub = extract_heading_or_subheading("1. A free ChatGPT account Create one at chat.openai.com.")
    assert sub is None

    # Rejection of regular prose
    sub = extract_heading_or_subheading("selling platforms is what makes this accessible to complete beginners right now.")
    assert sub is None


def test_query_analysis_section_intent():
    """Verify query parser identifies section queries and extracts target section ID."""
    # Full section title with em-dash
    q1 = "SECTION C — 20 Ways to Make Money Using AI (No Skills Needed)"
    a1 = analyze_query(q1)
    assert a1.is_section_query is True
    assert a1.target_section_id == "section_c"
    assert "20 ways to make money" in " ".join(a1.phrases)

    # Short section query
    q2 = "section b"
    a2 = analyze_query(q2)
    assert a2.is_section_query is True
    assert a2.target_section_id == "section_b"

    # Standard question without section intent
    q3 = "What are the only 3 things you need to start?"
    a3 = analyze_query(q3)
    assert a3.is_section_query is False
    assert a3.target_section_id is None


def test_chunker_section_propagation():
    """Verify multi-page section hierarchy is propagated across pages until a new section begins."""
    pages = [
        (1, "Table of Contents\n\nSECTION A ... 2\nSECTION B ... 5"),
        (2, "SECTION A — Foundations of AI\n\nAI is transforming software engineering.\n\nKey Concepts\n\nNeural networks and transformers."),
        (3, "Continuing on page 3.\n\nMore details on foundation models."),
        (4, "SECTION B — Practical Workflows\n\nStep 1: Data preparation.\n\nStep 2: Training."),
    ]

    chunks = chunk_pages(pages)
    assert len(chunks) >= 3

    # Page 2 chunk must have section_a
    p2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(p2_chunks) >= 1
    assert p2_chunks[0].section_id == "section_a"
    assert "SECTION A" in (p2_chunks[0].section_heading or "")

    # Page 3 chunk must inherit section_a
    p3_chunks = [c for c in chunks if c.page_number == 3]
    assert len(p3_chunks) >= 1
    assert p3_chunks[0].section_id == "section_a"
    assert "SECTION A" in (p3_chunks[0].section_heading or "")

    # Page 4 chunk must transition to section_b
    p4_chunks = [c for c in chunks if c.page_number == 4]
    assert len(p4_chunks) >= 1
    assert p4_chunks[0].section_id == "section_b"
    assert "SECTION B" in (p4_chunks[0].section_heading or "")


def test_reranker_preserves_multi_chunk_section():
    """Verify filter_and_select preserves all sequential chunks of a matched section beyond top_k."""
    # Create 7 sequential chunks representing a 20-item section
    candidates = []
    for i in range(1, 8):
        candidates.append(
            Candidate(
                chunk_id=f"chunk_{i:04d}",
                document_name="guide.pdf",
                page_number=16 + i,
                text=f"Method {i*3-2} through Method {i*3} details...",
                score=0.95 - (0.001 * i),
                metadata={
                    "section_id": "section_c",
                    "section_heading": "SECTION C — 20 Ways to Make Money Using AI",
                    "section_chunk_index": i - 1,
                    "page_number": 16 + i,
                },
            )
        )

    # Query targeting Section C
    query = "SECTION C — 20 Ways to Make Money Using AI (No Skills Needed)"
    selected = filter_and_select(candidates, query=query)

    # All 7 chunks must be preserved in order even if default top_k is 5
    assert len(selected) == 7
    assert [c.chunk_id for c in selected] == [f"chunk_{i:04d}" for i in range(1, 8)]

    # Must pass sufficiency check
    assert is_sufficient(selected, query=query) is True


def test_non_existent_heading_handling():
    """Verify querying a non-existent section heading correctly reports insufficient context."""
    candidates = [
        Candidate(
            chunk_id="chunk_0001",
            document_name="guide.pdf",
            page_number=2,
            text="Unrelated content about hardware specifications.",
            score=0.04,
            metadata={"section_id": "section_z", "section_heading": "SECTION Z — Hardware"},
        )
    ]
    query = "SECTION X — Quantum Computing Algorithms"
    selected = filter_and_select(candidates, query=query)
    # Insufficient context guard triggers
    assert is_sufficient(selected, query=query) is False
