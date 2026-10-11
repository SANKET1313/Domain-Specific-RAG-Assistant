"""Token-based, structure-aware and section-hierarchical chunking pipeline.

Preserves:
- Document section hierarchy (section_id, section_heading, subsection_heading).
- Cross-page boundary paragraph and sentence continuity (page_start, page_end).
- Section headings attached to their corresponding body text.
- Table integrity and header repetition across multi-row splits.
- Numbered procedures, lists, and definitions without arbitrary slicing.
- Page start/end, section metadata, chunk position, and deterministic sequence indices.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from app.core.config import get_settings

_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
_TERMINAL_PUNCTUATION = (".", "!", "?", ":", ";", '"', "'", ")", "]", "}")


@dataclass
class Chunk:
    text: str
    page_number: int  # Primary / start page for backwards compatibility
    index: int  # Sequential order within document
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    source_filename: Optional[str] = None
    page_start: int = 1
    page_end: int = 1
    chunk_position: int = 0  # Same as index
    section_id: Optional[str] = None  # e.g. "section_c"
    section_heading: Optional[str] = None  # e.g. "SECTION C — 20 Ways to Make Money Using AI"
    subsection_heading: Optional[str] = None  # e.g. "Method 1 — Social Media Caption Packages"
    section_chunk_index: int = 0  # Order within section
    token_count: int = 0


def _normalize_dash(text: str) -> str:
    """Normalize em-dash, en-dash, figure-dash, and minus signs to standard dash with surrounding space."""
    t = (
        text.replace("\u2014", " - ")
        .replace("\u2013", " - ")
        .replace("\u2012", " - ")
        .replace("\u2015", " - ")
        .replace("\u2212", " - ")
    )
    return re.sub(r"\s+", " ", t).strip()


def extract_section_header(line: str) -> tuple[Optional[str], Optional[str]]:
    """Detect major section headers such as 'SECTION C — ...', 'CHAPTER 1: ...', etc.
    Returns (section_id, full_heading).
    """
    clean = _normalize_dash(line.strip())
    # Skip table of contents dot leaders (e.g. 'SECTION C ... 17')
    if re.search(r"\.{3,}\s*\d+$", clean):
        return None, None

    clean_no_hash = clean.lstrip("#").strip()
    m = re.match(
        r"^(SECTION|CHAPTER|PART|MODULE|APPENDIX|UNIT)\s+([A-Z0-9]+)(?:\s*[:\-—]\s*(.*))?$",
        clean_no_hash,
        re.IGNORECASE,
    )
    if m:
        sec_type = m.group(1).upper()
        sec_num = m.group(2).upper()
        sec_title = m.group(3).strip() if m.group(3) else ""
        sec_id = f"{sec_type.lower()}_{sec_num.lower()}"
        full_title = f"{sec_type} {sec_num}" + (f" — {sec_title}" if sec_title else "")
        return sec_id, full_title

    # Top-level Markdown header (e.g. '# Architecture Overview')
    if clean.startswith("# ") and not clean.startswith("## ") and 3 <= len(clean_no_hash) <= 100:
        sec_slug = re.sub(r"[^a-z0-9]+", "_", clean_no_hash.lower()).strip("_")
        return f"sec_{sec_slug[:30]}", clean_no_hash

    return None, None


def extract_heading_or_subheading(line: str) -> Optional[str]:
    """Detect subsection headings, numbered procedures, methods, steps, or bold headings."""
    clean = line.strip().strip("#* :")
    if not clean or len(clean) < 3 or len(clean) > 120:
        return None

    # Markdown heading #, ##, etc.
    if line.strip().startswith(("#", "**")):
        return clean

    # Methods / Steps / Days / Categories
    if re.match(r"^(?:Method|Step|Day|Category|Phase|Part|Lesson)\s+\d+[\s:—–\-]", clean, re.IGNORECASE):
        return clean

    # Numbered chapter-style headings (e.g. 1.1 Architecture or short 1. Introduction)
    if re.match(r"^\d+(\.\d+)+\s+[A-Z]", clean) and len(clean.split()) <= 8:
        return clean
    if re.match(r"^\d+\.\s+[A-Z][a-zA-Z0-9\s]{2,40}$", clean) and len(clean.split()) <= 4:
        return clean

    # Distinct standalone titles without trailing punctuation (e.g., "The Only 3 Things You Need to Start", "Conclusion")
    if not clean.endswith((".", ",", ";", ":")) and not re.match(r"^\d+\.", clean) and len(clean.split()) <= 12:
        words = [w for w in clean.split() if w]
        if words:
            if clean.isupper() and len(clean) >= 4 and not any(c in clean for c in "{}[]<>"):
                return clean
            capitalized = sum(
                1 for w in words
                if w[0].isupper() or w.lower() in {"to", "in", "on", "of", "and", "the", "a", "an", "for", "with", "by", "no", "is", "your"}
            )
            if capitalized >= len(words) * 0.75:
                return clean

    return None


def _is_table_block(block: str) -> bool:
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    return len(lines) >= 2 and all(_TABLE_ROW_PATTERN.match(ln) for ln in lines)


def _split_table_with_header(table_text: str, max_chunk_tokens: int) -> list[str]:
    """Split a large table row-by-row, replicating the table header and separator on each chunk."""
    lines = [ln.strip() for ln in table_text.split("\n") if ln.strip()]
    if len(lines) <= 3:
        return [table_text]

    header_lines = lines[:2]  # Header row and |---|---| separator
    header_block = "\n".join(header_lines)
    header_tokens = max(1, len(header_block.split()))

    chunks: list[str] = []
    current_rows: list[str] = []
    current_tokens = header_tokens

    for row in lines[2:]:
        row_tokens = max(1, len(row.split()))
        if current_rows and (current_tokens + row_tokens > max_chunk_tokens):
            chunk_content = header_block + "\n" + "\n".join(current_rows)
            chunks.append(chunk_content)
            current_rows = [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunks.append(header_block + "\n" + "\n".join(current_rows))

    return chunks if chunks else [table_text]


def chunk_pages(
    pages: list[tuple[int, str]],
    document_id: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> list[Chunk]:
    """Split pages into structured, token-bounded chunks keeping section hierarchy, table headers,
    and cross-page continuations intact.
    """
    from llama_index.core.node_parser import SentenceSplitter

    settings = get_settings()
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # 1. First pass: Stitch paragraphs that continue across consecutive pages
    # Structure of page_blocks: list of (block_text, page_start, page_end)
    stitched_blocks: list[tuple[str, int, int]] = []
    
    for i, item in enumerate(pages):
        if len(item) == 3:
            p_start, p_end, text = item
        else:
            p_start, text = item
            p_end = p_start

        if not text or not text.strip():
            continue

        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not raw_paragraphs:
            continue

        # Check if the first paragraph of this page is a continuation of the last paragraph of the previous page
        if stitched_blocks and raw_paragraphs:
            prev_text, prev_start, prev_end = stitched_blocks[-1]
            first_curr = raw_paragraphs[0]
            
            # Continuation heuristic: previous paragraph does not end with terminal punctuation
            # and is not a heading or table, and current paragraph does not start with a heading or table
            prev_lines = prev_text.splitlines()
            last_prev_line = prev_lines[-1].strip() if prev_lines else ""
            curr_lines = first_curr.splitlines()
            first_curr_line = curr_lines[0].strip() if curr_lines else ""

            is_prev_complete = (
                not last_prev_line
                or last_prev_line.endswith(_TERMINAL_PUNCTUATION)
                or _TABLE_ROW_PATTERN.match(last_prev_line)
                or extract_section_header(last_prev_line)[0] is not None
                or extract_heading_or_subheading(last_prev_line) is not None
            )

            is_curr_heading = (
                _TABLE_ROW_PATTERN.match(first_curr_line)
                or extract_section_header(first_curr_line)[0] is not None
                or extract_heading_or_subheading(first_curr_line) is not None
                or first_curr_line.startswith(("#", "-", "*", "•"))
                or re.match(r"^\d+\.", first_curr_line)
            )

            is_prev_table = _TABLE_ROW_PATTERN.match(last_prev_line) is not None
            is_curr_table = _TABLE_ROW_PATTERN.match(first_curr_line) is not None

            if is_prev_table and is_curr_table and prev_end == p_start - 1:
                # Table continues across consecutive pages
                curr_lines_clean = [ln.strip() for ln in first_curr.splitlines() if ln.strip()]
                if len(curr_lines_clean) >= 3 and "|---" in curr_lines_clean[1]:
                    table_continuation = "\n".join(curr_lines_clean[2:])
                else:
                    table_continuation = first_curr
                merged_table = prev_text + "\n" + table_continuation
                stitched_blocks[-1] = (merged_table, prev_start, p_end)
                raw_paragraphs = raw_paragraphs[1:]
            elif not is_prev_complete and not is_curr_heading and prev_end == p_start - 1:
                # Text paragraph continuation across page boundary
                # If ended with a hyphen, join directly
                if last_prev_line.endswith("-"):
                    merged_first = prev_text[:-1] + first_curr
                else:
                    merged_first = prev_text + " " + first_curr
                stitched_blocks[-1] = (merged_first, prev_start, p_end)
                raw_paragraphs = raw_paragraphs[1:]

        idx = 0
        while idx < len(raw_paragraphs):
            p = raw_paragraphs[idx]
            # Check if this paragraph is purely a heading without body content
            is_heading_only = (
                len(p.splitlines()) <= 2
                and len(p.split()) <= 12
                and (
                    extract_section_header(p)[0] is not None
                    or extract_heading_or_subheading(p) is not None
                    or p.startswith(("#", "**"))
                )
            )
            if is_heading_only and idx + 1 < len(raw_paragraphs):
                next_p = raw_paragraphs[idx + 1]
                stitched_blocks.append((p + "\n\n" + next_p, p_start, p_end))
                idx += 2
            else:
                stitched_blocks.append((p, p_start, p_end))
                idx += 1

    # 2. Second pass: Track section headers and group blocks into structured chunks
    chunks: list[Chunk] = []
    running_index = 0

    active_section_id: Optional[str] = None
    active_section_heading: Optional[str] = None
    active_subsection: Optional[str] = None
    section_chunk_counts: dict[str, int] = {}

    for block_text, p_start, p_end in stitched_blocks:
        if not block_text.strip():
            continue

        first_line = block_text.split("\n")[0].strip()

        # Check for section header
        sec_id, sec_title = extract_section_header(first_line)
        is_toc = p_start <= 6 and ("contents" in block_text.lower()[:80] or "table of contents" in block_text.lower()[:80])
        
        if sec_id and not is_toc:
            if sec_id != active_section_id:
                active_section_id = sec_id
                active_section_heading = sec_title
                active_subsection = None
            elif sec_title and len(sec_title) > len(active_section_heading or "") and not sec_title.endswith("..."):
                active_section_heading = sec_title
        else:
            sub = extract_heading_or_subheading(first_line)
            if sub and not extract_section_header(first_line)[0]:
                active_subsection = sub

        current_sec_id = active_section_id
        current_sec_heading = active_section_heading

        # Handle tables with row-header preservation
        if _is_table_block(block_text):
            approx_tokens = max(1, int(len(block_text) / 3.8))
            if approx_tokens > settings.chunk_size:
                sub_pieces = _split_table_with_header(block_text, settings.chunk_size)
            else:
                sub_pieces = [block_text]
        else:
            # Use SentenceSplitter for regular text
            sub_pieces = splitter.split_text(block_text)

        for piece in sub_pieces:
            piece = piece.strip()
            if len(piece) < 15:
                continue

            # Context enrichment: Prepend section and subsection context if not already prominent
            chunk_text = piece
            header_tags = []
            if current_sec_heading and current_sec_heading.lower()[:25] not in piece.lower()[:120]:
                header_tags.append(current_sec_heading)
            if active_subsection and active_subsection.lower() not in piece.lower()[:120] and active_subsection != current_sec_heading:
                header_tags.append(active_subsection)

            if header_tags:
                prefix = f"[{' — '.join(header_tags)}]\n"
                chunk_text = f"{prefix}{piece}"

            approx_tokens = max(1, int(len(chunk_text) / 3.8))
            sec_key = current_sec_id or "general"
            sec_idx = section_chunk_counts.get(sec_key, 0)
            section_chunk_counts[sec_key] = sec_idx + 1

            effective_heading = current_sec_heading or active_subsection
            cid = f"chunk_{running_index:04d}"

            chunks.append(
                Chunk(
                    text=chunk_text,
                    page_number=p_start,
                    index=running_index,
                    document_id=document_id,
                    chunk_id=cid,
                    source_filename=source_filename,
                    page_start=p_start,
                    page_end=p_end,
                    chunk_position=running_index,
                    section_id=current_sec_id,
                    section_heading=effective_heading,
                    subsection_heading=active_subsection,
                    section_chunk_index=sec_idx,
                    token_count=approx_tokens,
                )
            )
            running_index += 1

    return chunks


def chunk_markdown(
    markdown_text: str,
    document_id: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> list[Chunk]:
    """Parse canonical Markdown document with <!-- PAGE: N --> markers into structure-aware chunks.

    Preserves page boundaries, headings, subheadings, tables, and sentence continuity.
    """
    if not markdown_text or not markdown_text.strip():
        return []

    content = markdown_text.strip()
    # Strip YAML frontmatter if present
    if content.startswith("---"):
        end_idx = content.find("---", 3)
        if end_idx != -1:
            content = content[end_idx + 3:].strip()

    # Split on page markers: <!-- PAGE: 1 --> or <!-- PAGE: 1-2 -->
    marker_pattern = re.compile(r"<!--\s*PAGE:\s*(\d+)(?:\s*[\-–]\s*(\d+))?\s*-->", re.IGNORECASE)
    parts = marker_pattern.split(content)

    pages: list[tuple[int, int, str]] = []

    if len(parts) == 1:
        # No page markers found
        pages.append((1, 1, content))
    else:
        # parts[0] is preamble before first marker
        if parts[0].strip():
            pages.append((1, 1, parts[0].strip()))

        i = 1
        while i < len(parts):
            p_start_str = parts[i]
            p_end_str = parts[i + 1] if i + 1 < len(parts) else None
            seg_text = parts[i + 2] if i + 2 < len(parts) else ""

            p_start = int(p_start_str) if p_start_str and p_start_str.isdigit() else 1
            p_end = int(p_end_str) if p_end_str and p_end_str.isdigit() else p_start
            if seg_text.strip():
                pages.append((p_start, p_end, seg_text.strip()))

            i += 3

    if not pages:
        pages = [(1, 1, content)]

    return chunk_pages(pages, document_id=document_id, source_filename=source_filename)

