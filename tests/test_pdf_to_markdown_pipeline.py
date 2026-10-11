"""Regression tests for the automatic PDF-to-Markdown preprocessing pipeline.

Covers:
1. PDF-to-Markdown generation and preservation of original PDF.
2. Scanned PDF processing and Markdown generation via OCR.
3. Cross-page sentence and paragraph reconstruction across page markers.
4. Markdown table preservation with header repetition.
5. Section and subsection heading hierarchy detection.
6. Processing documents exceeding 100 pages and querying final pages.
7. Extraction completeness validation (rejection of empty/unreadable PDFs).
8. Re-uploading / reprocessing replacing obsolete chunks safely.
9. Questions with no supporting evidence (insufficient context handling).
10. Development inspection mode (original file, markdown content, chunk boundaries).
"""
from __future__ import annotations

import io
from pathlib import Path
import pymupdf
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.ingestion.chunker import chunk_markdown
from app.ingestion.indexer import ingest_file
from app.ingestion.loader import ExtractionError
from app.ingestion.pdf_to_markdown import convert_document_to_markdown
from app.main import app
from app.rag.pipeline import answer_query
from app.vectorstore.factory import get_vector_store

client = TestClient(app)


def _make_pdf(path: Path, pages: list[str]) -> str:
    """Create a searchable PDF with specific text per page."""
    doc = pymupdf.open()
    for text in pages:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def _make_scanned_pdf(path: Path, texts: list[str]) -> str:
    """Create a bitmap-only PDF with rendered text."""
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
# 1. PDF-to-Markdown generation and preservation of original PDF
# ------------------------------------------------------------------------------
def test_pdf_to_markdown_preserves_pdf_and_generates_md(tmp_path: Path):
    pdf_path = tmp_path / "sample_policy.pdf"
    content = "SECTION 1 - Remote Work\n\nAll employees may work remotely up to two days per week."
    _make_pdf(pdf_path, [content])

    res = convert_document_to_markdown(
        pdf_path,
        document_name="sample_policy.pdf",
        document_id="doc_policy",
        output_dir=tmp_path,
    )

    # Original PDF must be preserved
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0

    # Corresponding .md file must exist
    assert res.markdown_path.exists()
    assert res.markdown_path.suffix == ".md"
    assert "sample_policy.md" in str(res.markdown_path)

    # Markdown content validation
    md = res.markdown_text
    assert "document_id: doc_policy" in md
    assert "<!-- PAGE: 1 -->" in md
    assert "All employees may work remotely" in md
    assert res.validation.is_valid is True


# ------------------------------------------------------------------------------
# 2. Scanned PDF processing and Markdown generation via OCR
# ------------------------------------------------------------------------------
def test_scanned_pdf_markdown_generation(tmp_path: Path):
    pdf_path = tmp_path / "scanned_receipt.pdf"
    _make_scanned_pdf(pdf_path, ["INVOICE ID: INV-7721\nTOTAL: $350.00"])

    res = convert_document_to_markdown(
        pdf_path,
        document_name="scanned_receipt.pdf",
        document_id="doc_receipt",
        output_dir=tmp_path,
    )

    assert res.markdown_path.exists()
    assert res.summary.document_type == "scanned"
    assert res.validation.pages_ocr_local >= 1 or res.validation.pages_ocr_vision >= 1
    assert "INV-7721" in res.markdown_text or "TOTAL" in res.markdown_text or "350" in res.markdown_text


# ------------------------------------------------------------------------------
# 3. Cross-page sentence and paragraph reconstruction
# ------------------------------------------------------------------------------
def test_cross_page_sentence_reconstruction(tmp_path: Path):
    p1 = "SECTION 5 - Security Compliance\n\nAll production cryptographic secrets must be rotated"
    p2 = "every ninety calendar days without exception by the security team."
    pdf_path = tmp_path / "cross_page.pdf"
    _make_pdf(pdf_path, [p1, p2])

    res = convert_document_to_markdown(
        pdf_path,
        document_name="cross_page.pdf",
        document_id="doc_cross",
        output_dir=tmp_path,
    )

    full_sentence = "rotated every ninety calendar days without exception"
    assert full_sentence in res.markdown_text

    # Chunks parsed from this Markdown must retain both pages and stitched sentence
    chunks = chunk_markdown(res.markdown_text, document_id="doc_cross", source_filename="cross_page.pdf")
    target_chunk = next((c for c in chunks if full_sentence in c.text), None)
    assert target_chunk is not None
    assert target_chunk.page_start == 1
    assert target_chunk.page_end == 2


# ------------------------------------------------------------------------------
# 4. Markdown table preservation with header repetition
# ------------------------------------------------------------------------------
def test_markdown_table_preservation_and_split(tmp_path: Path):
    table_header = "| Plan | Monthly Price | Cloud Storage |\n|---|---|---|\n"
    table_body = "\n".join(f"| Tier-{i} | ${i * 10} | {i * 50} GB |" for i in range(1, 35))
    full_table = f"## Pricing Architecture\n\n{table_header}{table_body}"

    pdf_path = tmp_path / "pricing_plans.pdf"
    _make_pdf(pdf_path, [full_table])

    res = convert_document_to_markdown(pdf_path, output_dir=tmp_path)
    assert "| Plan | Monthly Price | Cloud Storage |" in res.markdown_text

    # Chunk the markdown and ensure table headers repeat on splits
    chunks = chunk_markdown(res.markdown_text, document_id="doc_tbl", source_filename="pricing_plans.pdf")
    assert len(chunks) >= 1
    for c in chunks:
        if "| Tier-" in c.text:
            assert "| Plan | Monthly Price | Cloud Storage |" in c.text


# ------------------------------------------------------------------------------
# 5. Section and subsection heading hierarchy detection
# ------------------------------------------------------------------------------
def test_markdown_heading_hierarchy_detection():
    md_content = (
        "---\ndocument_id: doc_demo\n---\n\n"
        "<!-- PAGE: 1 -->\n\n"
        "# Architecture Overview\n\n"
        "The system employs a decentralized microservice architecture.\n\n"
        "<!-- PAGE: 2 -->\n\n"
        "## Method 1 — Asynchronous Event Streaming\n\n"
        "Events are published to Kafka topics with strict partition keys."
    )
    chunks = chunk_markdown(md_content, document_id="doc_demo", source_filename="demo.md")
    assert len(chunks) >= 2
    c1 = chunks[0]
    c2 = chunks[1]

    assert c1.section_heading == "Architecture Overview"
    assert c2.subsection_heading == "Method 1 — Asynchronous Event Streaming" or "Method 1" in (c2.section_heading or "")


# ------------------------------------------------------------------------------
# 6. Document exceeding 100 pages to Markdown & querying final pages
# ------------------------------------------------------------------------------
def test_document_exceeding_100_pages_to_markdown(tmp_path: Path):
    pages_text = [f"Page {i}: Boilerplate operations manual chapter." for i in range(1, 106)]
    # Place critical secret on page 104
    secret = "Special Severance Clause 104.9: Executive termination payout is $7,500,000."
    pages_text[103] = secret

    pdf_path = tmp_path / "large_doc_105.pdf"
    doc = pymupdf.open()
    for text in pages_text:
        p = doc.new_page(width=595, height=842)
        p.insert_text((50, 72), text, fontsize=10)
    doc.save(str(pdf_path))
    doc.close()

    res = convert_document_to_markdown(pdf_path, document_id="doc_large105", output_dir=tmp_path)
    assert res.validation.total_pages == 105
    assert res.validation.pages_processed == 105
    assert "$7,500,000" in res.markdown_text
    assert "<!-- PAGE: 104 -->" in res.markdown_text

    # Verify chunking preserves the 104 page number
    chunks = chunk_markdown(res.markdown_text, document_id="doc_large105", source_filename="large_doc_105.pdf")
    p104_chunk = next((c for c in chunks if "$7,500,000" in c.text), None)
    assert p104_chunk is not None
    assert p104_chunk.page_start == 104


# ------------------------------------------------------------------------------
# 7. Extraction completeness validation
# ------------------------------------------------------------------------------
def test_extraction_completeness_rejects_empty_pdf(tmp_path: Path):
    pdf_path = tmp_path / "completely_blank.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.save(str(pdf_path))
    doc.close()

    with pytest.raises(ExtractionError):
        convert_document_to_markdown(pdf_path, output_dir=tmp_path)


# ------------------------------------------------------------------------------
# 8. Re-uploading / reprocessing replacing obsolete chunks safely
# ------------------------------------------------------------------------------
def test_reuploading_replaces_obsolete_chunks(tmp_path: Path):
    store = get_vector_store()
    store.ensure_collection(2048)

    # Ingest Version 1
    doc_v1 = tmp_path / "dynamic_doc.txt"
    doc_v1.write_text("Notice Policy: The resignation notice period is 30 days.", encoding="utf-8")
    ingest_file(str(doc_v1), document_name="dynamic_doc.txt", document_id="doc_dyn_md")

    chunks_v1 = store.get_document_chunks("doc_dyn_md")
    assert any("30 days" in c.get("chunk_text", "") for c in chunks_v1)

    # Ingest Version 2 with modified policy
    doc_v2 = tmp_path / "dynamic_doc.txt"
    doc_v2.write_text("Notice Policy: The resignation notice period is 90 days.", encoding="utf-8")
    ingest_file(str(doc_v2), document_name="dynamic_doc.txt", document_id="doc_dyn_md")

    chunks_v2 = store.get_document_chunks("doc_dyn_md")
    assert not any("30 days" in c.get("chunk_text", "") for c in chunks_v2)
    assert any("90 days" in c.get("chunk_text", "") for c in chunks_v2)


# ------------------------------------------------------------------------------
# 9. Question with no supporting evidence
# ------------------------------------------------------------------------------
def test_unsupported_question_insufficient_context():
    query = "What is the dark energy warp propulsion metric in asteroid mining?"
    res = answer_query(query)
    assert res["sufficient_context"] is False
    assert "could not find sufficient information" in res["answer"].lower()
    assert len(res["sources"]) == 0


# ------------------------------------------------------------------------------
# 10. Development inspection mode
# ------------------------------------------------------------------------------
def test_development_inspection_endpoint(tmp_path: Path):
    pdf_path = tmp_path / "inspectable.pdf"
    _make_pdf(pdf_path, ["Section 1: System Health\nAll systems operational."])
    ingest_res = ingest_file(str(pdf_path), document_name="inspectable.pdf", document_id="doc_inspect")

    r = client.get("/documents/doc_inspect/inspect")
    assert r.status_code == 200
    data = r.json()
    assert data["document_id"] == "doc_inspect"
    assert data["chunk_count"] >= 1
    assert "chunks" in data
    assert data["chunks"][0]["page_start"] == 1
    # Check that markdown path or content is populated
    assert data.get("markdown_path") is not None or data.get("markdown_content") is not None
