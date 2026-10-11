"""Automated Regression Tests for Section 13:
PDF Information Extraction, Ingestion, Retrieval, Grounding, and Generation.

Covers all 10 required test scenarios:
1. Single page answer.
2. Sentence split between two consecutive pages.
3. Definition split across multiple chunks.
4. Table with headers and rows spanning pages (headers preserved on both pages).
5. Scanned page requiring OCR.
6. Information near the end of a document exceeding 100 pages.
7. Multiple passages required to answer one question.
8. No supporting evidence for the question (insufficient context without hallucination).
9. Temperature validation (default 0.2, 0.25 & 0.3 accepted, <0.2 or >0.3 rejected).
10. Configured temperature passed to LLM provider.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pymupdf
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages, _split_table_with_header
from app.ingestion.loader import load_pages_detailed
from app.rag.answer_generator import generate_answer
from app.rag.pipeline import answer_query
from app.rag.prompts import insufficient_context_message
from app.rag.validator import validate_answer_evidence
from app.retrieval.context_builder import build_context
from app.retrieval.reranker import filter_and_select, is_sufficient
from app.retrieval.retriever import Candidate


def _create_searchable_pdf(path: Path, pages: list[str]) -> str:
    """Create a multi-page searchable PDF with explicit text per page."""
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _create_scanned_pdf(path: Path, texts: list[str]) -> str:
    """Create an image-only PDF that requires OCR for text extraction."""
    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page(width=595, height=842)
        img = Image.new("RGB", (750, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), text, fill=(0, 0, 0))

        b = io.BytesIO()
        img.save(b, format="PNG")
        page.insert_image(page.rect, stream=b.getvalue())
    doc.save(str(path))
    doc.close()
    return str(path)


# ------------------------------------------------------------------------------
# 1. Single Page Answer
# ------------------------------------------------------------------------------
def test_regression_1_single_page_answer(tmp_path: Path):
    """Test 1: Verify extraction, retrieval, and grounding for a single page answer."""
    pdf_text = (
        "SECTION 1: REMOTE HARDWARE ALLOWANCE\n\n"
        "All permanent remote employees are entitled to a standard hardware allowance "
        "of $1,500 every two years to purchase ergonomic office equipment."
    )
    pdf_path = tmp_path / "single_page_doc.pdf"
    _create_searchable_pdf(pdf_path, [pdf_text])

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 1
    assert "$1,500" in pages[0].text

    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_p1", source_filename="single_page_doc.pdf")
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert "$1,500" in chunks[0].text

    cand = Candidate(
        chunk_id=chunks[0].chunk_id,
        document_name=chunks[0].source_filename,
        page_number=1,
        page_start=1,
        page_end=1,
        text=chunks[0].text,
        score=0.92,
        metadata={"document_name": chunks[0].source_filename},
    )
    selected = filter_and_select([cand], query="What is the remote hardware allowance?")
    assert len(selected) == 1
    assert is_sufficient(selected, query="What is the remote hardware allowance?") is True

    ctx = build_context(selected)
    assert "Page: 1" in ctx.text
    assert "$1,500" in ctx.text


# ------------------------------------------------------------------------------
# 2. Sentence Split Between Two Consecutive Pages
# ------------------------------------------------------------------------------
def test_regression_2_sentence_split_across_consecutive_pages(tmp_path: Path):
    """Test 2: Reconstruct sentences that continue across consecutive pages."""
    p1 = (
        "SECTION 6: SABBATICAL LEAVE REQUIREMENTS\n\n"
        "Employees eligible for sabbatical leave must have completed at least five consecutive years"
    )
    p2 = "of active full-time service without disciplinary sanctions."
    pdf_path = tmp_path / "sabbatical_policy.pdf"
    _create_searchable_pdf(pdf_path, [p1, p2])

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 2

    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_sab", source_filename="sabbatical_policy.pdf")

    # The sentence split across pages must be stitched together continuously
    target_sentence = "five consecutive years of active full-time service"
    stitched_chunk = next((c for c in chunks if target_sentence in c.text), None)
    assert stitched_chunk is not None, "Sentence broken across pages was not stitched."
    assert stitched_chunk.page_start == 1
    assert stitched_chunk.page_end == 2

    ctx = build_context([
        Candidate(
            chunk_id=stitched_chunk.chunk_id,
            document_name="sabbatical_policy.pdf",
            page_number=1,
            page_start=1,
            page_end=2,
            text=stitched_chunk.text,
            score=0.90,
            metadata={},
        )
    ])
    assert "Page: 1–2" in ctx.text


# ------------------------------------------------------------------------------
# 3. Definition Split Across Multiple Chunks
# ------------------------------------------------------------------------------
def test_regression_3_definition_split_across_multiple_chunks():
    """Test 3: Definition spanning multiple chunks is fully preserved and retrieved."""
    c1 = Candidate(
        chunk_id="chunk_def_01",
        document_name="glossary.pdf",
        page_number=4,
        page_start=4,
        page_end=4,
        chunk_position=1,
        text="Definition: 'Qualifying Material Breach' denotes any security compromise where unauthorized parties access customer PII.",
        score=0.88,
        metadata={"document_id": "doc_gloss", "chunk_id": "chunk_def_01", "chunk_position": 1, "section_id": "sec_def"},
    )
    c2 = Candidate(
        chunk_id="chunk_def_02",
        document_name="glossary.pdf",
        page_number=4,
        page_start=4,
        page_end=4,
        chunk_position=2,
        text="A Qualifying Material Breach additionally requires immediate notification to the Data Protection Officer within 24 hours.",
        score=0.85,
        metadata={"document_id": "doc_gloss", "chunk_id": "chunk_def_02", "chunk_position": 2, "section_id": "sec_def"},
    )

    selected = filter_and_select([c1, c2], query="What is a Qualifying Material Breach and what notification is required?")
    assert len(selected) == 2
    combined = " ".join(c.text for c in selected)
    assert "unauthorized parties access customer PII" in combined
    assert "within 24 hours" in combined


# ------------------------------------------------------------------------------
# 4. Table With Headers and Rows Spanning Pages
# ------------------------------------------------------------------------------
def test_regression_4_table_headers_preserved_across_pages(tmp_path: Path):
    """Test 4: Verify table headers are preserved on both pages / chunks when spanning pages."""
    table_header = "| Severance Tier | Years of Service | Multiplier |\n|---|---|---|\n"
    table_page1 = table_header + "| Tier A | 1 to 3 Years | 1.0x Salary |\n| Tier B | 4 to 7 Years | 1.5x Salary |"
    table_page2 = "| Tier C | 8 to 12 Years | 2.0x Salary |\n| Tier D | 13+ Years | 2.5x Salary |"

    pdf_path = tmp_path / "severance_table.pdf"
    _create_searchable_pdf(pdf_path, [table_page1, table_page2])

    pages, summary = load_pages_detailed(str(pdf_path))
    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_tbl", source_filename="severance_table.pdf")

    # Page 2 table rows must retain table structure and header context
    tier_d_chunk = next((c for c in chunks if "Tier D" in c.text), None)
    assert tier_d_chunk is not None
    assert "| Tier D | 13+ Years | 2.5x Salary |" in tier_d_chunk.text
    # Headers must be present on page 1 chunk as well
    tier_a_chunk = next((c for c in chunks if "Tier A" in c.text), None)
    assert tier_a_chunk is not None
    assert "| Severance Tier | Years of Service |" in tier_a_chunk.text

    # Also test _split_table_with_header directly to ensure header replication
    huge_table = table_header + "\n".join(f"| Tier-{i} | {i} Years | {i*0.5}x |" for i in range(1, 35))
    splits = _split_table_with_header(huge_table, max_chunk_tokens=50)
    assert len(splits) > 1
    for s in splits:
        assert "| Severance Tier | Years of Service | Multiplier |" in s
        assert "|---|---|---|" in s


# ------------------------------------------------------------------------------
# 5. Scanned Page Requiring OCR
# ------------------------------------------------------------------------------
def test_regression_5_scanned_page_requiring_ocr(tmp_path: Path):
    """Test 5: Process scanned image-only PDF pages using OCR without silently skipping."""
    target_secret = "SECURITY CLASSIFICATION CODE: DELTA-9817"
    pdf_path = tmp_path / "scanned_doc.pdf"
    _create_scanned_pdf(pdf_path, [target_secret])

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 1
    assert summary.document_type == "scanned"
    assert "DELTA-9817" in pages[0].text or "SECURITY" in pages[0].text or "CLASSIFICATION" in pages[0].text


# ------------------------------------------------------------------------------
# 6. Information Near End of Document Exceeding 100 Pages
# ------------------------------------------------------------------------------
def test_regression_6_information_near_end_of_100plus_pages(tmp_path: Path):
    """Test 6: Ingest and retrieve information located near the end of a document > 100 pages."""
    num_pages = 106
    pages_text = [f"Page {i}: General corporate standard operating guidelines." for i in range(1, num_pages + 1)]
    # Target clause on page 105
    target_clause = "Clause 99.8: The decommissioning escrow account balance is $45,000,000 at Apex Trust."
    pages_text[104] = target_clause

    pdf_path = tmp_path / "large_operations_manual.pdf"
    doc = pymupdf.open()
    for text in pages_text:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 106

    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_large_106", source_filename="large_operations_manual.pdf")

    # Locate the chunk corresponding to page 105
    target_chunk = next((c for c in chunks if "$45,000,000" in c.text), None)
    assert target_chunk is not None, "Page 105 clause was not found in chunked document."
    assert target_chunk.page_start == 105
    assert "Apex Trust" in target_chunk.text

    # Verify context formatting for page > 100
    ctx = build_context([
        Candidate(
            chunk_id=target_chunk.chunk_id,
            document_name="large_operations_manual.pdf",
            page_number=105,
            page_start=105,
            page_end=105,
            text=target_chunk.text,
            score=0.91,
            metadata={},
        )
    ])
    assert "Page: 105" in ctx.text
    assert "$45,000,000" in ctx.text


# ------------------------------------------------------------------------------
# 7. Multiple Passages Required to Answer One Question
# ------------------------------------------------------------------------------
def test_regression_7_multiple_passages_required():
    """Test 7: Retrieve and combine multiple distinct passages required for a full answer."""
    c1 = Candidate(
        chunk_id="chunk_w1",
        document_name="warranty.pdf",
        page_number=2,
        page_start=2,
        page_end=2,
        text="Section 2.1: The baseline factory warranty covers equipment for 12 months from delivery.",
        score=0.86,
        metadata={"document_name": "warranty.pdf", "page_number": 2},
    )
    c2 = Candidate(
        chunk_id="chunk_w2",
        document_name="warranty.pdf",
        page_number=8,
        page_start=8,
        page_end=8,
        text="Section 8.4: An extended enterprise warranty adds 24 months of coverage upon registration.",
        score=0.82,
        metadata={"document_name": "warranty.pdf", "page_number": 8},
    )

    query = "What is the total warranty period combining the baseline factory warranty and extended enterprise warranty?"
    selected = filter_and_select([c1, c2], query=query)
    assert len(selected) == 2
    ctx = build_context(selected)

    assert "12 months" in ctx.text
    assert "24 months" in ctx.text
    assert "Page: 2" in ctx.text
    assert "Page: 8" in ctx.text


# ------------------------------------------------------------------------------
# 8. No Supporting Evidence (Insufficient Context Without Hallucination)
# ------------------------------------------------------------------------------
def test_regression_8_no_supporting_evidence_insufficient_context():
    """Test 8: Ensure unsupported questions return insufficient context without hallucinations."""
    unrelated_cand = Candidate(
        chunk_id="chunk_cafe",
        document_name="office_cafeteria.pdf",
        page_number=1,
        page_start=1,
        page_end=1,
        text="Cafeteria lunch hours are 11:30 AM to 2:00 PM. Vegetarian options are available daily.",
        score=0.04,
        metadata={"document_name": "office_cafeteria.pdf"},
    )
    query = "What is the quantum telemetry subsystem protocol for deep-space orbiters?"
    selected = filter_and_select([unrelated_cand], query=query)
    sufficient = is_sufficient(selected, query=query)
    assert sufficient is False

    msg = insufficient_context_message()
    assert "could not find sufficient information" in msg.lower()


# ------------------------------------------------------------------------------
# 9. Verify Temperature Validation
# ------------------------------------------------------------------------------
def test_regression_9_temperature_validation():
    """Test 9: Verify temperature validation (default 0.2, 0.2-0.3 valid, <0.2 or >0.3 raises)."""
    # 1. Default must be 0.2
    s_default = Settings()
    assert s_default.llm_temperature == 0.2

    # 2. 0.25 and 0.3 must be valid
    s_mid = Settings(llm_temperature=0.25)
    assert s_mid.llm_temperature == 0.25

    s_max = Settings(llm_temperature=0.3)
    assert s_max.llm_temperature == 0.3

    s_min = Settings(llm_temperature=0.2)
    assert s_min.llm_temperature == 0.2

    # 3. Values < 0.2 must raise ValidationError
    with pytest.raises(ValidationError) as exc_low:
        Settings(llm_temperature=0.19)
    assert "LLM_TEMPERATURE must be between 0.2 and 0.3" in str(exc_low.value)

    with pytest.raises(ValidationError) as exc_zero:
        Settings(llm_temperature=0.0)
    assert "LLM_TEMPERATURE must be between 0.2 and 0.3" in str(exc_zero.value)

    # 4. Values > 0.3 must raise ValidationError
    with pytest.raises(ValidationError) as exc_high:
        Settings(llm_temperature=0.31)
    assert "LLM_TEMPERATURE must be between 0.2 and 0.3" in str(exc_high.value)

    with pytest.raises(ValidationError) as exc_large:
        Settings(llm_temperature=0.7)
    assert "LLM_TEMPERATURE must be between 0.2 and 0.3" in str(exc_large.value)


# ------------------------------------------------------------------------------
# 10. Confirm Configured Temperature Is Passed to LLM Provider
# ------------------------------------------------------------------------------
def test_regression_10_temperature_passed_to_llm_provider(monkeypatch):
    """Test 10: Verify the configured temperature is passed to the LLM generator."""
    recorded_temps: list[float] = []

    class MockLLM:
        def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2):
            from app.llm.base import LLMResponse
            recorded_temps.append(temperature)
            return LLMResponse(
                text="The remote allowance is $1,500.\n\nSources:\n- policy.pdf — Page 1",
                input_tokens=15,
                output_tokens=10,
                model="mock-llm",
            )

    import app.rag.answer_generator as ag
    monkeypatch.setattr(ag, "get_llm", lambda: MockLLM())

    # Set custom temperature within the allowed 0.2 - 0.3 range
    test_settings = Settings(llm_temperature=0.28)
    monkeypatch.setattr(ag, "get_settings", lambda: test_settings)

    cand = Candidate(
        chunk_id="chunk_test",
        document_name="policy.pdf",
        page_number=1,
        page_start=1,
        page_end=1,
        text="The remote allowance is $1,500.",
        score=0.95,
        metadata={},
    )
    ctx = build_context([cand])

    result = generate_answer("What is the allowance?", ctx)
    assert len(recorded_temps) == 1
    assert recorded_temps[0] == 0.28
    assert "1,500" in result.answer


# ------------------------------------------------------------------------------
# Additional Section 13 Feature Tests: Secondary Retrieval & Answer Validation
# ------------------------------------------------------------------------------
def test_secondary_retrieval_and_validation(monkeypatch):
    """Verify secondary retrieval activates when initial search is insufficient."""
    queries_retrieved: list[str] = []

    # Mock retrieve to simulate a miss on query but a hit on reformulated keywords
    def fake_retrieve(q: str, flt=None):
        queries_retrieved.append(q)
        if q == "Can you explain the quantum satellite telemetry bandwidth?":
            # Initial complex query returns very low score candidate
            return [
                Candidate(
                    chunk_id="c_low",
                    document_name="specs.pdf",
                    page_number=1,
                    page_start=1,
                    page_end=1,
                    text="Introduction to unrelated laboratory equipment.",
                    score=0.03,
                    metadata={"document_name": "specs.pdf"},
                )
            ]
        else:
            # Secondary keyword query returns high score candidate
            return [
                Candidate(
                    chunk_id="c_high",
                    document_name="specs.pdf",
                    page_number=3,
                    page_start=3,
                    page_end=3,
                    text="The satellite telemetry bandwidth is calibrated at 500 Mbps.",
                    score=0.91,
                    metadata={"document_name": "specs.pdf", "page_number": 3},
                )
            ]
        return []

    import app.rag.pipeline as pipe
    monkeypatch.setattr(pipe, "retrieve", fake_retrieve)

    class MockLLM:
        def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2):
            from app.llm.base import LLMResponse
            return LLMResponse(
                text="The satellite telemetry bandwidth is calibrated at 500 Mbps.",
                input_tokens=20,
                output_tokens=15,
                model="mock-llm",
            )

    monkeypatch.setattr("app.rag.answer_generator.get_llm", lambda: MockLLM())

    resp = pipe.answer_query("Can you explain the quantum satellite telemetry bandwidth?", debug=True)
    assert resp["sufficient_context"] is True
    assert "500 Mbps" in resp["answer"]
    assert resp["debug"]["secondary_retrieval_attempted"] is True
    assert "telemetry" in resp["debug"]["secondary_query"]


def test_answer_evidence_validation_flags_hallucinated_citations():
    """Verify validate_answer_evidence flags citations to nonexistent pages/documents."""
    cand = Candidate(
        chunk_id="c1",
        document_name="actual_policy.pdf",
        page_number=2,
        page_start=2,
        page_end=2,
        text="Eligible staff receive 15 days vacation.",
        score=0.9,
        metadata={"document_name": "actual_policy.pdf"},
    )
    ctx = build_context([cand])

    # Hallucinated answer citing a fake document and fake page
    answer_with_hallucination = (
        "According to secret_guide.pdf on Page 99, employees receive 15 days vacation."
    )
    sources = [
        {"document": "actual_policy.pdf", "page": 2, "page_start": 2, "page_end": 2},
        {"document": "fake_document.pdf", "page": 99, "page_start": 99, "page_end": 99},
    ]

    _, val_sources, report = validate_answer_evidence(answer_with_hallucination, sources, ctx)
    assert report["citations_valid"] is False
    assert any("fake_document.pdf" in h for h in report["hallucinated_citations"])
    assert any("Page 99" in h for h in report["hallucinated_citations"])
    # Fake source removed from validated sources
    assert len(val_sources) == 1
    assert val_sources[0]["document"] == "actual_policy.pdf"
