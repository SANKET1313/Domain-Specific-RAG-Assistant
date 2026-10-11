"""Query understanding and preprocessing.

Provides:
- Whitespace and punctuation normalization.
- Query type classification (short keyword, phrase, question).
- Preservation of acronyms, technical terms, and numbers.
- Extraction of search tokens and exact phrases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "what", "how", "why", "when",
    "where", "who", "which", "this", "that", "these", "those", "it", "its",
    "do", "does", "did", "can", "could", "should", "would", "about", "tell",
    "me", "give", "show", "please", "i", "you", "we", "they", "there",
}


@dataclass
class QueryAnalysis:
    original_query: str
    cleaned_query: str
    is_short_keyword: bool
    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    query_type: str = "question"  # "short_keyword" | "phrase" | "question" | "section"
    is_section_query: bool = False
    target_section_id: Optional[str] = None
    target_section_heading: Optional[str] = None


def normalize_dashes(text: str) -> str:
    """Normalize em-dash, en-dash, figure-dash, and minus signs to standard dash with surrounding space."""
    t = text.replace("\u2014", " - ").replace("\u2013", " - ").replace("\u2012", " - ").replace("\u2015", " - ").replace("\u2212", " - ")
    return re.sub(r"\s+", " ", t).strip()


def analyze_query(query: str) -> QueryAnalysis:
    """Analyze and structure user query for hybrid retrieval and section expansion."""
    if not query:
        return QueryAnalysis(
            original_query="",
            cleaned_query="",
            is_short_keyword=True,
            keywords=[],
            phrases=[],
            query_type="short_keyword",
        )

    # Clean whitespace and normalize Unicode dashes
    normalized = normalize_dashes(query.strip())
    cleaned = " ".join(normalized.split())
    tokens = [t for t in re.findall(r"[A-Za-z0-9_\-\.]+", cleaned.lower())]

    # Filter stop words to find salient technical keywords
    keywords = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]

    # Detect phrases enclosed in quotes or 2-4 word sequences
    quoted_phrases = re.findall(r'"([^"]+)"', query)
    phrases = [p.strip().lower() for p in quoted_phrases if p.strip()]
    if not phrases and 1 < len(tokens) <= 6:
        phrases.append(cleaned.lower())

    # Detect explicit section intent (e.g. SECTION C — 20 Ways to Make Money..., Chapter 1, Part A)
    sec_m = re.search(
        r"\b(SECTION|CHAPTER|PART|MODULE|APPENDIX|UNIT)\s+([A-Z0-9]+)(?:\s*[:\-]\s*(.*))?",
        cleaned,
        re.IGNORECASE,
    )
    is_sec = False
    target_sec_id = None
    target_sec_heading = None

    if sec_m:
        is_sec = True
        sec_type = sec_m.group(1).lower()
        sec_num = sec_m.group(2).lower()
        sec_rest = sec_m.group(3).strip() if sec_m.group(3) else ""
        target_sec_id = f"{sec_type}_{sec_num}"
        target_sec_heading = f"{sec_m.group(1).upper()} {sec_m.group(2).upper()}" + (f" - {sec_rest}" if sec_rest else "")
        if sec_rest and len(sec_rest) >= 4:
            phrases.append(sec_rest.lower())

    # Classify query type
    is_short = len(tokens) <= 3
    if is_sec:
        q_type = "section"
    elif is_short:
        q_type = "short_keyword"
    elif any(cleaned.lower().startswith(w) for w in ("what", "how", "why", "when", "where", "who", "can", "is", "are")):
        q_type = "question"
    else:
        q_type = "phrase"

    return QueryAnalysis(
        original_query=query,
        cleaned_query=cleaned,
        is_short_keyword=is_short,
        keywords=keywords,
        phrases=phrases,
        query_type=q_type,
        is_section_query=is_sec,
        target_section_id=target_sec_id,
        target_section_heading=target_sec_heading,
    )
