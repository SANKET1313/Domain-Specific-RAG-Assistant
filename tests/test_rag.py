"""Tests for prompt construction and insufficient-context handling (mocked LLM)."""
from app.rag import answer_generator
from app.rag.prompts import build_user_prompt, system_prompt
from app.retrieval.context_builder import build_context
from app.retrieval.retriever import Candidate


def test_system_prompt_contains_domain_and_rules():
    sp = system_prompt()
    assert "only" in sp.lower()
    assert "sources" in sp.lower()


def test_user_prompt_embeds_context_and_question():
    prompt = build_user_prompt("SOURCE 1\nDocument: a.pdf\ntext", "What is X?")
    assert "SOURCE 1" in prompt
    assert "What is X?" in prompt


def test_generate_answer_uses_llm_and_returns_sources(monkeypatch):
    class FakeLLM:
        def generate(self, system_prompt, user_prompt, temperature=0.0):
            from app.llm.base import LLMResponse
            return LLMResponse(text="Answer: X based on a.pdf.\nSources:\n- a.pdf — Page 1",
                               input_tokens=10, output_tokens=5, model="fake")

    monkeypatch.setattr(answer_generator, "get_llm", lambda: FakeLLM())
    cand = Candidate(chunk_id="chunk_0001", document_name="a.pdf", page_number=1,
                     text="X is defined here.", score=0.9, metadata={"source": "a.pdf"})
    ctx = build_context([cand])
    result = answer_generator.generate_answer("What is X?", ctx)
    assert "a.pdf" in result.answer
    assert result.sources[0]["document"] == "a.pdf"
    assert result.sources[0]["page"] == 1
