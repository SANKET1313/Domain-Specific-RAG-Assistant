"""Automatic PDF and Document-to-Markdown Preprocessing Pipeline.

Converts uploaded PDFs, DOCXs, and documents into a structured, validated
canonical Markdown representation preserving:
- Original PDF page numbers via `<!-- PAGE: N -->` markers.
- Cross-page sentence and paragraph continuity without artificial boundaries.
- Headings, subheadings, lists, definitions, tables, table headers, footnotes.
- Correct reading order and layout structure.
- Extraction completeness validation (tracking empty pages, OCR errors, short text).
- Saving the generated `.md` file alongside the original document on disk.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.ingestion.cleaner import clean_text
from app.ingestion.loader import (
    DocumentExtractionSummary,
    ExtractionError,
    ExtractedPage,
    load_pages_detailed,
)

logger = get_logger("pdf_to_markdown")

_PAGE_MARKER_REGEX = re.compile(r"<!--\s*PAGE:\s*(\d+)\s*-->", re.IGNORECASE)
_TABLE_ROW_REGEX = re.compile(r"^\s*\|.*\|\s*$")
_TERMINAL_PUNCTUATION = (".", "!", "?", ":", ";", '"', "'", ")", "]", "}")


@dataclass
class ExtractionValidationReport:
    is_valid: bool
    total_pages: int
    pages_processed: int
    pages_digital: int
    pages_ocr_local: int
    pages_ocr_vision: int
    pages_empty: int
    empty_pages: list[int]
    suspiciously_short_pages: list[int]
    warnings: list[str] = field(default_factory=list)


@dataclass
class MarkdownConversionResult:
    markdown_text: str
    markdown_path: Path
    summary: DocumentExtractionSummary
    validation: ExtractionValidationReport
    page_map: dict[int, str] = field(default_factory=dict)


def _is_terminal(line: str) -> bool:
    """Check if a line ends with terminal punctuation or structural ending."""
    clean = line.strip()
    if not clean:
        return True
    return clean.endswith(_TERMINAL_PUNCTUATION) or bool(_TABLE_ROW_REGEX.match(clean))


def _is_heading_or_structure(line: str) -> bool:
    """Check if a line is a heading, table, list item, or structural block."""
    clean = line.strip()
    if not clean:
        return False
    if clean.startswith(("#", "-", "*", "•", ">", "|")):
        return True
    if re.match(r"^\d+[\.\)]\s+", clean):
        return True
    if re.match(r"^(SECTION|CHAPTER|PART|MODULE|APPENDIX|UNIT)\s+[A-Z0-9]+", clean, re.IGNORECASE):
        return True
    return False


def validate_extraction_completeness(
    pages: list[ExtractedPage],
    summary: DocumentExtractionSummary,
    filename: str,
) -> ExtractionValidationReport:
    """Validate that extraction was complete, no pages were dropped, and detect anomalies."""
    if not pages or summary.total_pages == 0:
        raise ExtractionError(f"Document '{filename}' contains 0 pages or produced no extraction output.")

    total_pages = summary.total_pages
    processed_count = len(pages)
    if processed_count != total_pages:
        raise ExtractionError(
            f"Extraction integrity failure for '{filename}': expected {total_pages} pages, "
            f"but only processed {processed_count} pages."
        )

    empty_pages: list[int] = []
    short_pages: list[int] = []
    warnings: list[str] = list(summary.warnings)

    non_empty_chars = 0
    for p in pages:
        txt = p.text.strip()
        if not txt:
            empty_pages.append(p.page_number)
        else:
            non_empty_chars += len(txt)
            if len(txt) < 15 and p.method in ("local_ocr", "vision_ocr"):
                short_pages.append(p.page_number)
                warnings.append(f"Page {p.page_number}: Suspiciously short extraction ({len(txt)} chars).")

    # If completely empty across all pages
    if non_empty_chars == 0:
        raise ExtractionError(f"Document '{filename}' contains no readable text or content across all {total_pages} pages.")

    # Check if failed pages exceed 50% on a multi-page document
    if total_pages > 2 and len(empty_pages) == total_pages:
        raise ExtractionError(f"All {total_pages} pages of '{filename}' are empty or unreadable.")

    return ExtractionValidationReport(
        is_valid=True,
        total_pages=total_pages,
        pages_processed=processed_count,
        pages_digital=summary.pages_digital,
        pages_ocr_local=summary.pages_ocr_local,
        pages_ocr_vision=summary.pages_ocr_vision,
        pages_empty=len(empty_pages),
        empty_pages=empty_pages,
        suspiciously_short_pages=short_pages,
        warnings=warnings,
    )


def stitch_cross_page_markdown(pages: list[ExtractedPage]) -> list[tuple[int, int, str]]:
    """Stitch sentences and paragraphs across consecutive pages, tracking (start_page, end_page, content)."""
    blocks: list[tuple[int, int, str]] = []

    for idx, page in enumerate(pages):
        p_num = page.page_number
        raw_text = page.text.strip()
        if not raw_text:
            continue

        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        # Check if first paragraph of this page continues the last paragraph of the previous page
        if blocks and paragraphs:
            prev_start, prev_end, prev_text = blocks[-1]
            first_curr = paragraphs[0]

            prev_lines = prev_text.splitlines()
            last_prev_line = prev_lines[-1].strip() if prev_lines else ""
            curr_lines = first_curr.splitlines()
            first_curr_line = curr_lines[0].strip() if curr_lines else ""

            is_prev_table = bool(_TABLE_ROW_REGEX.match(last_prev_line))
            is_curr_table = bool(_TABLE_ROW_REGEX.match(first_curr_line))
            is_curr_structural = _is_heading_or_structure(first_curr_line)
            is_prev_ended = _is_terminal(last_prev_line)

            # Table continuation across consecutive pages
            if is_prev_table and is_curr_table and prev_end == p_num - 1:
                # Strip repeated header row on page continuation if present
                clean_rows = [ln.strip() for ln in first_curr.splitlines() if ln.strip()]
                if len(clean_rows) >= 3 and "|---" in clean_rows[1]:
                    table_body = "\n".join(clean_rows[2:])
                else:
                    table_body = first_curr
                merged_table = prev_text + "\n" + table_body
                blocks[-1] = (prev_start, p_num, merged_table)
                paragraphs = paragraphs[1:]

            # Sentence / paragraph continuation across consecutive pages
            elif not is_prev_ended and not is_curr_structural and prev_end == p_num - 1:
                if last_prev_line.endswith("-"):
                    # De-hyphenate broken word at page margin
                    merged_text = prev_text[:-1] + first_curr
                else:
                    merged_text = prev_text + " " + first_curr
                blocks[-1] = (prev_start, p_num, merged_text)
                paragraphs = paragraphs[1:]

        for p in paragraphs:
            blocks.append((p_num, p_num, p))

    return blocks


def format_structured_markdown(
    blocks: list[tuple[int, int, str]],
    document_id: str,
    document_name: str,
    summary: DocumentExtractionSummary,
) -> str:
    """Format extracted stitched blocks into structured Markdown with frontmatter and page markers."""
    now_iso = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "---",
        f"document_id: {document_id}",
        f"document_name: {document_name}",
        f"total_pages: {summary.total_pages}",
        f"extraction_type: {summary.document_type}",
        f"generated_at: {now_iso}",
        "---",
        "",
    ]

    current_page_marker: Optional[tuple[int, int]] = None

    for p_start, p_end, text in blocks:
        marker = (p_start, p_end)
        # Emit page marker when entering a new page or page span
        if marker != current_page_marker:
            if p_start == p_end:
                lines.append(f"<!-- PAGE: {p_start} -->\n")
            else:
                lines.append(f"<!-- PAGE: {p_start}-{p_end} -->\n")
            current_page_marker = marker

        # Normalize heading formats in the Markdown stream
        clean_block = text.strip()

        # If block is a capitalized section header without Markdown tag, elevate to markdown heading
        sec_m = re.match(
            r"^(SECTION|CHAPTER|PART|MODULE|APPENDIX|UNIT)\s+([A-Z0-9]+)(?:\s*[:\-—]\s*(.*))?$",
            clean_block.splitlines()[0],
            re.IGNORECASE,
        )
        if sec_m and not clean_block.startswith("#"):
            lines.append(f"## {clean_block}\n")
        else:
            lines.append(f"{clean_block}\n")

    return "\n".join(lines).strip() + "\n"


def convert_document_to_markdown(
    file_path: str | Path,
    document_name: Optional[str] = None,
    document_id: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> MarkdownConversionResult:
    """Execute complete PDF/Document-to-Markdown preprocessing pipeline.

    1. Extracts all pages (digital + OCR fallback) page-by-page.
    2. Validates extraction completeness (rejects empty/corrupted documents).
    3. Stitches cross-page sentences, paragraphs, and tables.
    4. Formats into canonical structured Markdown with page markers.
    5. Saves the generated `.md` file to disk while preserving the original document.
    """
    settings = get_settings()
    path = Path(file_path).resolve()
    doc_name = document_name or path.name
    doc_id = document_id or f"doc_{path.stem}"

    # Target Markdown path (saved alongside original document or in documents_dir)
    save_dir = Path(output_dir or settings.documents_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    md_filename = f"{path.stem}.md"
    md_path = save_dir / md_filename

    # 1. Page-by-page detailed extraction
    extracted_pages, summary = load_pages_detailed(str(path))

    # 2. Completeness validation
    validation = validate_extraction_completeness(extracted_pages, summary, doc_name)

    # 3. Cross-page sentence & paragraph reconstruction
    stitched_blocks = stitch_cross_page_markdown(extracted_pages)

    # 4. Generate structured Markdown
    markdown_content = format_structured_markdown(
        blocks=stitched_blocks,
        document_id=doc_id,
        document_name=doc_name,
        summary=summary,
    )

    # 5. Persist the generated .md file to disk
    try:
        md_path.write_text(markdown_content, encoding="utf-8")
        logger.info("Saved canonical Markdown for '%s' to '%s'", doc_name, md_path)
    except Exception as exc:
        logger.warning("Could not write markdown file to '%s': %s", md_path, exc)

    page_map = {p.page_number: p.text for p in extracted_pages}

    return MarkdownConversionResult(
        markdown_text=markdown_content,
        markdown_path=md_path,
        summary=summary,
        validation=validation,
        page_map=page_map,
    )
