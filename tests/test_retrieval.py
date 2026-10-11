"""Unit tests for retrieval filtering, selection, and context assembly."""
from app.retrieval.context_builder import build_context
from app.retrieval.reranker import filter_and_select, is_sufficient
from app.retrieval.retriever import Candidate


def _c(score, text, cid):
    return Candidate(
        chunk_id=cid, document_name="doc.pdf", page_number=1,
        text=text, score=score, metadata={"source": "doc.pdf"},
    )


def test_filter_drops_low_scores_and_dedups():
    cands = [
        _c(0.9, "alpha beta", "c1"),
        _c(0.85, "alpha beta", "c2"),   # duplicate text
        _c(0.05, "irrelevant", "c3"),   # below floor
        _c(0.6, "gamma delta", "c4"),
    ]
    selected = filter_and_select(cands)
    texts = [c.text for c in selected]
    assert "irrelevant" not in texts
    assert texts.count("alpha beta") == 1


def test_is_sufficient_detects_weak_retrieval():
    assert is_sufficient([_c(0.9, "x", "c1")]) is True
    assert is_sufficient([_c(0.1, "x", "c1")]) is False
    assert is_sufficient([]) is False


def test_context_format_is_structured_and_traceable():
    ctx = build_context([_c(0.9, "The policy requires X.", "chunk_0001")])
    assert "SOURCE 1" in ctx.text
    assert "Document: doc.pdf" in ctx.text
    assert "Page: 1" in ctx.text
    assert "chunk_0001" in ctx.text
