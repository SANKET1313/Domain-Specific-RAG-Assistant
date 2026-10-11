"""Unit tests for cleaning and chunking (no external services required)."""
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_text


def test_clean_text_collapses_whitespace_and_hyphenation():
    raw = "This  is   a  hyphen-\nated   word.\r\n\r\n\r\nExtra lines."
    cleaned = clean_text(raw)
    assert "hyphenated" in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned


def test_chunker_preserves_page_numbers_and_order():
    pages = [(1, "Sentence one. " * 80), (2, "Sentence two. " * 80)]
    chunks = chunk_pages(pages)
    assert chunks, "expected at least one chunk"
    assert {c.page_number for c in chunks} <= {1, 2}
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunker_skips_empty_pages():
    chunks = chunk_pages([(1, "   "), (2, "Real content here. " * 50)])
    assert all(c.page_number == 2 for c in chunks)
