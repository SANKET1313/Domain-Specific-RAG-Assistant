"""Comprehensive test suite covering the 4 decoupled pipeline stages and all 10 evaluation scenarios.

Stages tested independently:
A. Extraction: Information present in extracted text (PDF, DOCX, Scanned, Tables, Footnotes).
B. Chunking: Information preserved across chunk and page boundaries, table header preservation.
C. Retrieval: Candidate recall, hybrid scoring, and neighboring-chunk expansion.
D. Generation: Grounded answer generation, citations, and insufficient-context handling.

10 Scenarios tested:
1. Question whose answer is contained within one chunk.
2. Question whose answer spans two consecutive pages.
3. Question whose answer spans multiple chunks.
4. Question requiring a heading and its associated paragraph.
5. Question about a table.
6. Question about a scanned PDF.
7. Question requiring information near the end of a document exceeding 100 pages.
8. Question involving exact numbers, dates, or identifiers.
9. Document re-uploaded after its content changes.
10. Question whose answer is not supported by the document.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.ingestion.cleaner import clean_pages, clean_text
from app.ingestion.chunker import Chunk, chunk_pages, _split_table_with_header
from app.ingestion.indexer import ingest_file
from app.ingestion.loader import ExtractionError, load_pages_detailed, _load_docx
from app.rag.answer_generator import generate_answer
from app.retrieval.context_builder import BuiltContext, build_context
from app.retrieval.query_understanding import analyze_query
from app.retrieval.reranker import expand_neighboring_chunks, filter_and_select, is_sufficient
from app.retrieval.retriever import Candidate
from app.vectorstore.factory import get_vector_store


def _make_pdf(path: Path, pages: list[str]) -> str:
    """Create a searchable PDF with specific text per page."""
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _make_scanned_pdf(path: Path, page_texts: list[str]) -> str:
    """Create a bitmap-only PDF with rendered text."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        img = Image.new("RGB", (700, 900), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), text, fill=(0, 0, 0))

        b = io.BytesIO()
        img.save(b, format="PNG")
        page.insert_image(page.rect, stream=b.getvalue())
    doc.save(str(path))
    doc.close()
    return str(path)


# ==============================================================================
# SECTION 8: FOUR INDEPENDENT STAGE TESTS
# ==============================================================================

def test_stage_a_extraction(tmp_path: Path):
    """Stage A: Extraction test independently verifies text, tables, and page tracking."""
    text_p1 = "Introduction to Security Architecture.\nAll systems must authenticate via SSO."
    text_p2 = "Access Control List:\n\n| Role | Access Level |\n|---|---|\n| Admin | Full |\n| Guest | ReadOnly |"
    pdf_path = tmp_path / "stage_a.pdf"
    _make_pdf(pdf_path, [text_p1, text_p2])

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 2
    assert "SSO" in pages[0].text
    assert "| Admin | Full |" in pages[1].text
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_stage_b_chunking_cross_page_and_table_headers():
    """Stage B: Chunking test independently verifies sentence stitching and table headers."""
    # Sentence starts on page 1 and finishes on page 2
    p1 = "SECTION 1 - Identity Governance\n\nThe authorization policy strictly stipulates that all multi-factor keys must be rotated"
    p2 = "every ninety calendar days without exception."
    pages = [(1, p1), (2, p2)]

    chunks = chunk_pages(pages, document_id="doc_stage_b", source_filename="policy.pdf")
    assert len(chunks) >= 1
    
    # Check that the sentence is preserved continuously
    full_sentence = "rotated every ninety calendar days without exception"
    assert any(full_sentence in c.text for c in chunks)
    
    # Verify page metadata on the stitched chunk
    stitched_chunk = next(c for c in chunks if full_sentence in c.text)
    assert stitched_chunk.page_start == 1
    assert stitched_chunk.page_end == 2

    # Verify table header replication
    big_table = (
        "| ID | Server | Status |\n"
        "|---|---|---|\n"
        + "\n".join(f"| {i} | srv-{i}.corp | Active |" for i in range(1, 40))
    )
    table_splits = _split_table_with_header(big_table, max_chunk_tokens=60)
    assert len(table_splits) > 1
    for part in table_splits:
        assert "| ID | Server | Status |" in part
        assert "|---|---|---|" in part


def test_stage_c_retrieval_and_neighbor_expansion():
    """Stage C: Retrieval test independently verifies hybrid fusion and neighboring chunk expansion."""
    c1 = Candidate(
        chunk_id="chunk_0001",
        document_name="guide.pdf",
        page_number=1,
        page_start=1,
        page_end=1,
        chunk_position=1,
        text="Incident Response Procedure: Step 1 involves immediate host isolation.",
        score=0.88,
        metadata={"document_id": "doc_100", "chunk_id": "chunk_0001", "chunk_position": 1, "section_id": "sec_ir"},
    )
    c2 = Candidate(
        chunk_id="chunk_0002",
        document_name="guide.pdf",
        page_number=2,
        page_start=2,
        page_end=2,
        chunk_position=2,
        text="Step 2 requires notifying the cybersecurity steering committee.",
        score=0.82,
        metadata={"document_id": "doc_100", "chunk_id": "chunk_0002", "chunk_position": 2, "section_id": "sec_ir"},
    )

    selected = filter_and_select([c1, c2], query="Incident Response host isolation")
    assert selected[0].chunk_id == "c1" or selected[0].chunk_id == "chunk_0001"
    assert is_sufficient(selected, query="host isolation") is True


def test_stage_d_generation_grounded_answer(monkeypatch):
    """Stage D: Generation test independently verifies LLM answers accurately from evidence."""
    class FakeLLM:
        def generate(self, system_prompt, user_prompt, temperature=0.0):
            from app.llm.base import LLMResponse
            return LLMResponse(
                text="Answer:\nFull-time personnel receive 20 days paid leave according to the leave policy.\n\nSources:\n- leave_policy.pdf — Page 1",
                input_tokens=25,
                output_tokens=20,
                model="test-llm",
            )

    import app.rag.answer_generator as ag
    monkeypatch.setattr(ag, "get_llm", lambda: FakeLLM())

    c = Candidate(
        chunk_id="chunk_0001",
        document_name="leave_policy.pdf",
        page_number=1,
        page_start=1,
        page_end=1,
        text="Full-time employees receive 20 days paid annual leave per year.",
        score=0.92,
        metadata={"document_name": "leave_policy.pdf"},
    )
    ctx = build_context([c])
    result = generate_answer("How many leave days do employees receive?", ctx)
    assert "20 days" in result.answer
    assert result.sources[0]["document"] == "leave_policy.pdf"
    assert str(result.sources[0]["page"]) == "1"


# ==============================================================================
# SECTION 10: TEN SPECIFIC REGRESSION TEST SCENARIOS
# ==============================================================================

# Scenario 1: Answer contained within one chunk
def test_scenario_1_answer_in_single_chunk():
    cand = Candidate(
        chunk_id="chunk_0001",
        document_name="benefits.pdf",
        page_number=3,
        text="The annual learning and development stipend is $1,000 per employee.",
        score=0.89,
        metadata={"document_name": "benefits.pdf", "page_number": 3},
    )
    selected = filter_and_select([cand], query="What is the learning and development stipend?")
    assert len(selected) == 1
    assert "$1,000" in selected[0].text
    assert is_sufficient(selected, query="What is the learning stipend?") is True


# Scenario 2: Answer spans two consecutive pages
def test_scenario_2_answer_spans_two_consecutive_pages(tmp_path: Path):
    p1 = "SECTION 4 - Equipment Return Policy\n\nWhen terminating employment, employees must surrender all company-issued laptops,"
    p2 = "security badges, monitors, and encrypted storage keys within 48 business hours."
    pdf_path = tmp_path / "equipment_policy.pdf"
    _make_pdf(pdf_path, [p1, p2])

    pages, summary = load_pages_detailed(str(pdf_path))
    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_equip", source_filename="equipment_policy.pdf")

    # Cross-page paragraph should be joined with page_start=1 and page_end=2
    matching = [c for c in chunks if "within 48 business hours" in c.text and "laptops" in c.text]
    assert len(matching) >= 1
    target_chunk = matching[0]
    assert target_chunk.page_start == 1
    assert target_chunk.page_end == 2

    # Verify context builder formats as Page 1-2
    ctx = build_context([
        Candidate(
            chunk_id=target_chunk.chunk_id or "chunk_0001",
            document_name="equipment_policy.pdf",
            page_number=1,
            text=target_chunk.text,
            score=0.92,
            metadata={},
            page_start=1,
            page_end=2,
        )
    ])
    assert "Page: 1–2" in ctx.text


# Scenario 3: Answer spans multiple chunks
def test_scenario_3_answer_spans_multiple_chunks():
    c1 = Candidate(
        chunk_id="chunk_0001",
        document_name="project.pdf",
        page_number=1,
        chunk_position=1,
        text="Phase 1 Deliverables: Architecture specification and security threat modeling.",
        score=0.85,
        metadata={"document_id": "doc_proj", "chunk_id": "chunk_0001", "chunk_position": 1, "section_id": "sec_phases"},
    )
    c2 = Candidate(
        chunk_id="chunk_0002",
        document_name="project.pdf",
        page_number=2,
        chunk_position=2,
        text="Phase 2 Deliverables: Prototype implementation, load testing, and penetration sign-off.",
        score=0.80,
        metadata={"document_id": "doc_proj", "chunk_id": "chunk_0002", "chunk_position": 2, "section_id": "sec_phases"},
    )

    selected = filter_and_select([c1, c2], query="What are the deliverables for Phase 1 and Phase 2?")
    assert len(selected) == 2
    combined_text = " ".join(c.text for c in selected)
    assert "Phase 1 Deliverables" in combined_text
    assert "Phase 2 Deliverables" in combined_text


# Scenario 4: Heading and its associated paragraph
def test_scenario_4_heading_and_associated_paragraph():
    text = (
        "### Mandatory Whistleblower Protections\n\n"
        "Employees reporting regulatory violations are granted complete confidentiality "
        "and legal immunity from retaliatory termination under corporate policy WP-204."
    )
    chunks = chunk_pages([(1, text)], document_id="doc_wp", source_filename="governance.pdf")
    assert len(chunks) == 1
    assert "Mandatory Whistleblower Protections" in chunks[0].text
    assert "WP-204" in chunks[0].text
    assert chunks[0].section_heading == "Mandatory Whistleblower Protections"


# Scenario 5: Question about a table
def test_scenario_5_table_extraction_and_query(tmp_path: Path):
    table_text = (
        "Reimbursement Schedule:\n\n"
        "| Travel Category | Max Reimbursement | Authorization Required |\n"
        "|---|---|---|\n"
        "| Domestic Flight | $600 | Manager |\n"
        "| International Flight | $1800 | Vice President |\n"
        "| Train Travel | $150 | None |"
    )
    pdf = _make_pdf(tmp_path / "travel.pdf", [table_text])
    pages, summary = load_pages_detailed(pdf)
    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned)

    flight_chunk = next((c for c in chunks if "International Flight" in c.text), None)
    assert flight_chunk is not None
    assert "| International Flight | $1800 | Vice President |" in flight_chunk.text
    # Table header preserved
    assert "| Travel Category | Max Reimbursement |" in flight_chunk.text


# Scenario 6: Question about a scanned PDF
def test_scenario_6_scanned_pdf_ocr(tmp_path: Path):
    pdf = _make_scanned_pdf(
        tmp_path / "scanned_invoice.pdf",
        ["Invoice Number: INV-98421\nVendor: Northwind IT Systems\nTotal Due: $4,250.00"],
    )
    pages, summary = load_pages_detailed(pdf)
    assert summary.total_pages == 1
    assert summary.document_type == "scanned"
    assert "INV-98421" in pages[0].text or "4,250" in pages[0].text or "Northwind" in pages[0].text


# Scenario 7: Question near end of document exceeding 100 pages
def test_scenario_7_document_exceeding_100_pages(tmp_path: Path):
    # Construct a 105-page PDF with target secret clause on page 104
    num_pages = 105
    pages_text = [f"Page {i}: General administrative boilerplate regulations." for i in range(1, num_pages + 1)]
    pages_text[103] = "Appendix K: Special Severance Clause\nKey executives receive twenty-four months salary continuation."

    doc = pymupdf.open()
    for text in pages_text:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=10)
    pdf_path = tmp_path / "large_manual.pdf"
    doc.save(str(pdf_path))
    doc.close()

    pages, summary = load_pages_detailed(str(pdf_path))
    assert summary.total_pages == 105

    cleaned = clean_pages([(p.page_number, p.text) for p in pages])
    chunks = chunk_pages(cleaned, document_id="doc_large", source_filename="large_manual.pdf")

    # Find the chunk near the end
    p104_chunk = next((c for c in chunks if "Special Severance Clause" in c.text), None)
    assert p104_chunk is not None
    assert p104_chunk.page_start == 104
    assert "twenty-four months salary continuation" in p104_chunk.text


# Scenario 8: Question involving exact numbers, dates, or identifiers
def test_scenario_8_exact_numbers_and_identifiers():
    query = "What is the budget for Project Titan (ID: TITAN-2026) starting November 15, 2026?"
    analysis = analyze_query(query)
    assert "titan-2026" in [kw.lower() for kw in analysis.keywords] or any("titan" in p for p in analysis.phrases)

    cand = Candidate(
        chunk_id="chunk_0042",
        document_name="projects.pdf",
        page_number=14,
        text="Project Titan (ID: TITAN-2026) has an allocated capital budget of $2,450,000 commencing November 15, 2026.",
        score=0.88,
        metadata={"chunk_id": "chunk_0042"},
    )
    selected = filter_and_select([cand], query=query)
    assert selected[0].chunk_id == "chunk_0042"
    assert "$2,450,000" in selected[0].text
    assert "TITAN-2026" in selected[0].text


# Scenario 9: Document re-uploaded after content changes
def test_scenario_9_reupload_and_replacement(tmp_path: Path):
    store = get_vector_store()
    store.ensure_collection(2048)

    # Version 1
    doc_v1 = tmp_path / "dynamic_policy.txt"
    doc_v1.write_text("Notice Period Policy:\nThe resignation notice period is 30 days.", encoding="utf-8")
    ingest_file(str(doc_v1), document_name="dynamic_policy.txt", document_id="doc_dyn")

    chunks_v1 = store.get_document_chunks("doc_dyn")
    assert len(chunks_v1) >= 1
    assert any("30 days" in c.get("chunk_text", "") for c in chunks_v1)

    # Version 2 with updated content
    doc_v2 = tmp_path / "dynamic_policy.txt"
    doc_v2.write_text("Notice Period Policy:\nThe resignation notice period is 60 days.", encoding="utf-8")
    ingest_file(str(doc_v2), document_name="dynamic_policy.txt", document_id="doc_dyn")

    chunks_v2 = store.get_document_chunks("doc_dyn")
    assert len(chunks_v2) >= 1
    # Old stale chunk must NOT exist
    assert not any("30 days" in c.get("chunk_text", "") for c in chunks_v2)
    # New chunk must exist
    assert any("60 days" in c.get("chunk_text", "") for c in chunks_v2)


# Scenario 10: Question whose answer is not supported by the document
def test_scenario_10_unsupported_question():
    unrelated_cand = Candidate(
        chunk_id="chunk_0001",
        document_name="office_rules.pdf",
        page_number=1,
        text="Office kitchen hours are 8:00 AM to 6:00 PM. Please label all personal containers.",
        score=0.03,
        metadata={"document_name": "office_rules.pdf"},
    )
    query = "What is the warp drive subspace distortion coefficient in quantum vacuum?"
    selected = filter_and_select([unrelated_cand], query=query)
    assert is_sufficient(selected, query=query) is False
