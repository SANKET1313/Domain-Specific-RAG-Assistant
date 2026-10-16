"""Comprehensive Grounding, Retrieval Diagnostics, and Evidence Contract Test Suite.

Implements the three separate test paths:
1. Test A (Retrieval Path): Pure retrieval tests without calling the LLM. Verifies that
   known ground-truth passages are retrieved in top-k for known questions.
2. Test B (Oracle-Context Generation Path): Provides verified ground-truth context directly
   to the LLM prompt to ensure the generation stage produces grounded answers with citations.
3. Test C (End-to-End RAG Path): Ingests a PDF, indexes chunks, performs retrieval, and verifies
   grounded answer generation with traceable page citations and supporting evidence contract.

Also tests:
- Cross-document isolation and absence of contamination
- Stale vector cleanup upon re-upload
- Exact phrases, numbers, and definitions
- Consecutive page sentence stitching
- Unanswerable out-of-scope questions (honest refusal, zero hallucinated sources)
- Groq Llama provider initialization and generation
- Pre-generation diagnostic report contents
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pymupdf
import pytest

from app.core.config import get_settings
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import ingest_file
from app.ingestion.loader import load_pages_detailed
from app.llm.base import LLMResponse
from app.llm.factory import get_llm
from app.llm.groq_provider import GroqProvider
from app.rag.answer_generator import generate_answer
from app.rag.pipeline import answer_query
from app.rag.validator import validate_answer_evidence
from app.retrieval.context_builder import build_context
from app.retrieval.reranker import filter_and_select, is_sufficient
from app.retrieval.retriever import Candidate, retrieve
from app.vectorstore.factory import get_vector_store


def _make_pdf(path: Path, pages: list[str]) -> str:
    """Helper to generate searchable multi-page PDF."""
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


# ==============================================================================
# 1. TEST PATH A: Pure Retrieval Tests (Zero LLM Calls)
# ==============================================================================
def test_retrieval_isolated_path_exact_phrase_and_heading(tmp_path: Path):
    """Test A.1: Verify pure retrieval surfaces known passage for exact phrases and headings."""
    store = get_vector_store()

    doc_id = "test_doc_alpha_99"
    pdf_path = tmp_path / "alpha_policy.pdf"
    _make_pdf(
        pdf_path,
        [
            "SECTION 4.2: ENTERPRISE ENCRYPTION PROTOCOLS\n"
            "All customer data in transit must utilize TLS 1.3 with AES-256-GCM cipher suites.\n"
            "Key rotation occurs every 90 days automatically through the HSM enclave.",
            "SECTION 4.3: ACCESS CONTROLS\n"
            "Multi-factor authentication is strictly mandatory for all administrative access.",
        ],
    )

    ingest_file(str(pdf_path), document_name="alpha_policy.pdf", document_id=doc_id)

    # Pure retrieval query without LLM
    candidates = retrieve("What cipher suites are required for customer data?", flt={"document_id": doc_id})
    selected = filter_and_select(candidates, query="What cipher suites are required for customer data?")

    assert len(selected) > 0
    top_chunk = selected[0]
    assert "AES-256-GCM" in top_chunk.text
    assert "TLS 1.3" in top_chunk.text
    assert top_chunk.page_number == 1
    assert is_sufficient(selected, query="What cipher suites are required for customer data?") is True

    # Heading-based query
    heading_cands = retrieve("ENTERPRISE ENCRYPTION PROTOCOLS", flt={"document_id": doc_id})
    selected_heading = filter_and_select(heading_cands, query="ENTERPRISE ENCRYPTION PROTOCOLS")
    assert any("Key rotation" in c.text for c in selected_heading)


def test_retrieval_isolated_path_unanswerable_out_of_scope():
    """Test A.2: Verify out-of-scope query without evidence correctly evaluates to insufficient."""
    cand = Candidate(
        chunk_id="chunk_unrelated",
        document_name="cloud_architecture.pdf",
        page_number=1,
        text="Kubernetes ingress controllers route external traffic via Traefik to worker pods.",
        score=0.03,  # Low similarity score below threshold
        metadata={"document_name": "cloud_architecture.pdf"},
    )

    selected = filter_and_select([cand], query="What is the reimbursement limit for meal expenses?")
    assert is_sufficient(selected, query="What is the reimbursement limit for meal expenses?") is False


# ==============================================================================
# 2. TEST PATH B: Oracle-Context Generation Tests (LLM Grounding Direct Context)
# ==============================================================================
def test_oracle_context_generation_path():
    """Test B.1: Provide known ground-truth context and verify answer generation and citation."""
    cand = Candidate(
        chunk_id="oracle_chunk_01",
        document_name="employee_benefits_manual.pdf",
        page_number=14,
        page_start=14,
        page_end=14,
        text=(
            "Section 7.4: Parental Leave Policy.\n"
            "Full-time regular employees with at least 12 months of service are eligible for "
            "16 consecutive weeks of fully paid parental leave."
        ),
        score=0.98,
        metadata={
            "document_name": "employee_benefits_manual.pdf",
            "document_id": "doc_benefits_74",
            "section_heading": "Parental Leave Policy",
        },
    )

    ctx = build_context([cand])

    # Mock LLM returning accurate grounded answer
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        text=(
            "Answer:\n"
            "Full-time regular employees with at least 12 months of continuous service "
            "are entitled to 16 consecutive weeks of fully paid parental leave.\n\n"
            "Sources:\n"
            "- employee_benefits_manual.pdf — Page 14"
        ),
        input_tokens=150,
        output_tokens=45,
        model="mock-llama",
    )

    with patch("app.rag.answer_generator.get_llm", return_value=mock_llm):
        result = generate_answer("How many weeks of parental leave are provided?", ctx)

    validated_answer, sources, val_report = validate_answer_evidence(
        result.answer, result.sources, ctx
    )

    assert "16 consecutive weeks" in validated_answer
    assert len(sources) == 1
    assert sources[0]["document"] == "employee_benefits_manual.pdf"
    assert sources[0]["page"] == 14
    assert val_report["is_supported"] is True
    assert val_report["citations_valid"] is True
    assert val_report["supporting_evidence"] is not None
    assert "16 consecutive weeks" in val_report["supporting_evidence"]


def test_oracle_context_generation_detects_unsupported_or_hallucinated_facts():
    """Test B.2: When answer makes claims absent from context, validator flags or rejects."""
    cand = Candidate(
        chunk_id="oracle_chunk_02",
        document_name="security_policy.pdf",
        page_number=3,
        page_start=3,
        page_end=3,
        text="Password complexity requires a minimum length of 12 characters.",
        score=0.92,
        metadata={"document_name": "security_policy.pdf", "document_id": "doc_sec"},
    )
    ctx = build_context([cand])

    # LLM hallucinating numbers and details not in context
    hallucinated_answer = (
        "Passwords must be changed every 14 days and require 99 special symbols."
    )
    sources = [{"document": "security_policy.pdf", "page": 3, "page_start": 3, "page_end": 3}]

    ans, val_sources, val_report = validate_answer_evidence(hallucinated_answer, sources, ctx)

    assert val_report["is_supported"] is False
    assert val_report["insufficient_evidence"] is True
    assert "retrieved passages do not contain sufficient evidence" in ans.lower()
    assert len(val_sources) == 0


# ==============================================================================
# 3. TEST PATH C: End-to-End RAG Path (PDF Ingest -> Index -> Retrieve -> Answer)
# ==============================================================================
def test_end_to_end_rag_path_with_pdf_ingestion(tmp_path: Path, monkeypatch):
    """Test C: Full workflow from PDF ingestion to grounded answer with evidence contract."""
    pdf_path = tmp_path / "procurement_policy.pdf"
    _make_pdf(
        pdf_path,
        [
            "PROCUREMENT STANDARD OPERATING PROCEDURE\n"
            "Purchases under $5,000 require single-manager approval via the Coupa portal.\n"
            "Purchases between $5,000 and $50,000 require Department Head and VP approval.\n"
            "All contracts exceeding $50,000 require Legal Counsel and CFO signoff.",
        ],
    )

    doc_id = "doc_procure_2026"
    ingest_result = ingest_file(str(pdf_path), document_name="procurement_policy.pdf", document_id=doc_id)
    assert ingest_result["vectors_inserted"] > 0

    # Mock LLM for deterministic generation
    class MockGroqLLM:
        def generate(self, system_prompt, user_prompt, temperature=0.2):
            return LLMResponse(
                text=(
                    "Answer:\n"
                    "Purchases between $5,000 and $50,000 require Department Head and VP approval.\n\n"
                    "Sources:\n"
                    "- procurement_policy.pdf — Page 1"
                ),
                input_tokens=180,
                output_tokens=35,
                model="llama-3.3-70b-versatile",
            )

    monkeypatch.setattr("app.rag.answer_generator.get_llm", lambda: MockGroqLLM())

    response = answer_query(
        query="Who must approve purchases between $5,000 and $50,000?",
        flt={"document_id": doc_id},
        debug=True,
    )

    assert response["sufficient_context"] is True
    assert response["insufficient_evidence"] is False
    assert "Department Head and VP" in response["answer"]
    assert response["supporting_evidence"] is not None
    assert "$5,000 and $50,000" in response["supporting_evidence"]
    assert response["source_document_name"] == "procurement_policy.pdf"
    assert len(response["sources"]) == 1
    assert response["sources"][0]["page"] == 1

    # Verify diagnostic report
    diag = response["diagnostic_report"]
    assert diag is not None
    assert diag["original_user_question"] == "Who must approve purchases between $5,000 and $50,000?"
    assert diag["metadata_filters_applied"] == {"document_id": doc_id}
    assert len(diag["final_ordered_passages"]) > 0
    assert diag["is_sufficient"] is True


# ==============================================================================
# 4. EDGE CASE & REGRESSION TESTS
# ==============================================================================
def test_cross_document_isolation_prevents_contamination(tmp_path: Path):
    """Verify document-scoping strictly prevents cross-document chunk contamination."""
    pdf_a = tmp_path / "secret_project_a.pdf"
    _make_pdf(pdf_a, ["Codename Operation Cobalt: Launch date set for October 24."])
    ingest_file(str(pdf_a), document_name="secret_project_a.pdf", document_id="doc_a")

    pdf_b = tmp_path / "public_guidelines_b.pdf"
    _make_pdf(pdf_b, ["Standard public community forum rules and code of conduct."])
    ingest_file(str(pdf_b), document_name="public_guidelines_b.pdf", document_id="doc_b")

    # Scoped search to Document B must NEVER retrieve chunks from Document A
    results_for_b = retrieve("What is Codename Operation Cobalt launch date?", flt={"document_id": "doc_b"})
    for c in results_for_b:
        assert c.metadata.get("document_id") == "doc_b"
        assert "Cobalt" not in c.text


def test_stale_vectors_deleted_on_reupload(tmp_path: Path):
    """Verify that updating a document purges obsolete chunks from vector store."""
    store = get_vector_store()
    doc_id = "doc_versioned_policy"

    # Version 1
    p1 = tmp_path / "v1_policy.pdf"
    _make_pdf(p1, ["Version 1.0: Legacy system retention window is 30 days."])
    ingest_file(str(p1), document_name="versioned_policy.pdf", document_id=doc_id)

    chunks_v1 = store.get_document_chunks(doc_id)
    assert len(chunks_v1) == 1
    assert "30 days" in chunks_v1[0]["chunk_text"]

    # Version 2 re-upload
    p2 = tmp_path / "v2_policy.pdf"
    _make_pdf(p2, ["Version 2.0: Updated system retention window is now 180 days."])
    ingest_file(str(p2), document_name="versioned_policy.pdf", document_id=doc_id)

    chunks_v2 = store.get_document_chunks(doc_id)
    assert len(chunks_v2) == 1
    assert "180 days" in chunks_v2[0]["chunk_text"]
    assert "30 days" not in chunks_v2[0]["chunk_text"]


def test_consecutive_page_sentence_stitching(tmp_path: Path):
    """Verify sentence split across consecutive pages is stitched without artificial breaks."""
    pdf_path = tmp_path / "split_sentence.pdf"
    _make_pdf(
        pdf_path,
        [
            "In accordance with federal regulation 45 CFR Part 46, all research involving",
            "human subjects must undergo comprehensive institutional review board approval.",
        ],
    )
    pages, summary = load_pages_detailed(str(pdf_path))
    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_stitched", source_filename="split_sentence.pdf")

    combined = " ".join(c.text for c in chunks)
    assert "all research involving human subjects" in combined


def test_groq_provider_initialization_and_generation():
    """Verify Groq provider initializes and formats requests for Llama models."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_test_mock_key"}):
        get_settings.cache_clear()
        get_llm.cache_clear()
        provider = get_llm()
        assert isinstance(provider, GroqProvider)
        assert provider._model == "llama-3.3-70b-versatile"
        assert provider._base_url == "https://api.groq.com/openai/v1"

        # Mock chat completion
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Grounded Llama answer."))]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 10
        provider._client.chat.completions.create = MagicMock(return_value=mock_resp)

        llm_res = provider.generate("System prompt", "User question", temperature=0.2)
        assert llm_res.text == "Grounded Llama answer."
        assert llm_res.input_tokens == 50
        assert llm_res.output_tokens == 10
