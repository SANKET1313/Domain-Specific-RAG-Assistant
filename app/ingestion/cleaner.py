"""Text cleaning and normalization.

Preserves:
- Markdown tables, structure, and column relationships.
- Headings, lists, numbered procedures, and bullet points.
- Dates, numbers, acronyms, and important domain terminology.

Removes:
- ASCII control characters and non-printable noise.
- Hyphenated line breaks.
- Repeated recurring header/footer artifacts across multi-page documents.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_HYPHEN_WRAP = re.compile(r"(\w)[-\xad\u2010\u2011]\n(\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")
_PAGE_NUM_PATTERNS = re.compile(r"^(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|\d+)$", re.IGNORECASE)
_SECTION_PREFIX = re.compile(
    r"^(?:SECTION|CHAPTER|PART|MODULE|APPENDIX|UNIT)\s+[A-Z0-9]+(?:\s*[—–\-].*)?$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize whitespace, join hyphenated words, and strip control characters while preserving layout."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Clean horizontal whitespace line-by-line to preserve indentation and tables
    raw_lines: list[str] = [_MULTISPACE.sub(" ", line).strip() for line in text.split("\n")]

    # Join split section headings across consecutive lines (skipping empty lines)
    joined_lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if line and _SECTION_PREFIX.match(line):
            j = i + 1
            while j < len(raw_lines) and not raw_lines[j]:
                j += 1
            if j < len(raw_lines):
                next_line = raw_lines[j]
                # Next line must be a legitimate title continuation (starts with capital or '(', short, no lowercase start)
                is_title_continuation = (
                    bool(next_line)
                    and (next_line[0].isupper() or next_line.startswith("("))
                    and not next_line.endswith((".", ":", ";"))
                    and len(next_line.split()) <= 8
                    and not _SECTION_PREFIX.match(next_line)
                    and not re.match(r"^(?:Method|Step|Day|Category|Part)\s+\d+", next_line, re.IGNORECASE)
                    and not re.match(r"^(?:Each|This|In|For|When|The|And|With|To)\s+[a-z]", next_line)
                )
                if is_title_continuation:
                    joined_lines.append(f"{line} {next_line}")
                    i = j + 1
                    continue
        joined_lines.append(line)
        i += 1

    text = "\n".join(joined_lines)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Clean text across pages, detecting and stripping recurring headers and footers as well as isolated page numbers."""
    if not pages:
        return []

    # First pass: basic clean for each page
    prelim = [(pn, clean_text(t)) for pn, t in pages if t and t.strip()]
    if not prelim:
        return []

    # Check for repeating headers (first line) and footers (last line)
    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()

    for _, t in prelim:
        lines = [line.strip() for line in t.split("\n") if line.strip()]
        if lines:
            header_counts[lines[0]] += 1
            if len(lines) > 1:
                footer_counts[lines[-1]] += 1

    threshold = max(2, int(len(prelim) * 0.70))
    recurring_headers = {
        h for h, count in header_counts.items()
        if count >= threshold and (len(h) < 100 or _PAGE_NUM_PATTERNS.match(h))
    }
    recurring_footers = {
        f for f, count in footer_counts.items()
        if count >= threshold and (len(f) < 100 or _PAGE_NUM_PATTERNS.match(f))
    }

    cleaned_pages_list: list[tuple[int, str]] = []
    for pn, t in prelim:
        lines = [l for l in t.split("\n")]
        # Strip recurring header or isolated page number from top
        while lines and (
            lines[0].strip() in recurring_headers or _PAGE_NUM_PATTERNS.match(lines[0].strip())
        ):
            lines.pop(0)
        # Strip recurring footer or isolated page number from bottom
        while lines and (
            lines[-1].strip() in recurring_footers or _PAGE_NUM_PATTERNS.match(lines[-1].strip())
        ):
            lines.pop()

        page_str = "\n".join(lines).strip()
        if page_str:
            cleaned_pages_list.append((pn, clean_text(page_str)))

    return cleaned_pages_list
