"""Modal cloud LLM provider backed by Hugging Face Inference API."""
from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.config import get_settings
from app.llm.base import BaseLLM, LLMResponse

logger = logging.getLogger(__name__)


class ModalProvider(BaseLLM):
    """LLM Provider that delegates inference to a Modal endpoint running Hugging Face models."""

    def __init__(self) -> None:
        settings = get_settings()
        self._url = (settings.modal_llm_url or "").rstrip("/")
        self._model = settings.hf_model or settings.llm_model or "openai/gpt-oss-20b"
        self._app_name = settings.modal_app_name or "domain-specific-rag-llm"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> LLMResponse:
        """Generate a response by dispatching the RAG query to the Modal LLM service."""
        # 1. If HTTP endpoint URL is configured, call the Modal web endpoint
        if self._url:
            endpoint_url = f"{self._url}/v1/rag/generate"
            payload = {
                "question": user_prompt,
                "context": system_prompt,
                "history": [],
                "temperature": temperature,
            }
            try:
                resp = requests.post(endpoint_url, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "")
                    usage = data.get("usage", {})
                    return LLMResponse(
                        text=answer,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        model=data.get("model", self._model),
                    )
                elif resp.status_code in (401, 403):
                    raise RuntimeError("LLM authentication failed. Check the Hugging Face token configured in the Modal secret.")
                elif resp.status_code == 404:
                    raise RuntimeError("The configured Hugging Face model is unavailable. Check HF_MODEL and the available Hugging Face Inference Provider.")
                elif resp.status_code == 429:
                    raise RuntimeError("Rate limit exceeded on Hugging Face Inference Provider. Please retry shortly.")
                elif resp.status_code == 400:
                    raise RuntimeError(f"Bad Request to Modal LLM endpoint: {resp.text}")
                else:
                    raise RuntimeError(f"Modal LLM endpoint returned status {resp.status_code}: {resp.text}")
            except requests.RequestException as exc:
                logger.error("Failed to connect to Modal LLM endpoint at %s: %s", endpoint_url, exc)
                raise RuntimeError("Modal LLM endpoint unavailable or network error occurred. Please check MODAL_LLM_URL.") from exc

        # 2. Direct Modal SDK invocation fallback if endpoint URL is not provided AND Modal credentials exist
        import os
        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            try:
                import modal
                func = modal.Function.from_name(self._app_name, "generate_answer")
                result: dict[str, Any] = func.remote(context=system_prompt, question=user_prompt)
                answer = result.get("answer", "")
                usage = result.get("usage", {})
                return LLMResponse(
                    text=answer,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    model=result.get("model", self._model),
                )
            except Exception as exc:
                logger.warning("Modal remote invocation failed: %s", exc)

        # 3. Check if NVIDIA NIM is available as generative LLM fallback
        settings = get_settings()
        if settings.nvidia_api_key and settings.nvidia_api_key not in ("YOUR_NEW_NVIDIA_API_KEY", "your_nvidia_api_key_here"):
            try:
                from app.llm.nvidia_provider import NvidiaLLMProvider
                return NvidiaLLMProvider().generate(system_prompt, user_prompt, temperature=temperature)
            except Exception as exc:
                logger.warning("NVIDIA NIM LLM fallback failed, falling back to extractive synthesis: %s", exc)

        # 4. Grounded context synthesis
        context_block = ""
        if "=== CONTEXT START ===" in user_prompt and "=== CONTEXT END ===" in user_prompt:
            context_block = user_prompt.split("=== CONTEXT START ===")[1].split("=== CONTEXT END ===")[0].strip()

        if context_block:
            answer_text = _synthesize_grounded_answer(user_prompt, context_block)
            return LLMResponse(
                text=answer_text,
                input_tokens=len(user_prompt.split()),
                output_tokens=len(answer_text.split()),
                model="grounded-extractive-synthesis",
            )

        raise RuntimeError(
            "MODAL_LLM_URL is not set in .env and retrieved context was empty."
        )


def _synthesize_grounded_answer(user_prompt: str, context_block: str) -> str:
    """Extract and synthesize clean, readable grounded answers from retrieved context."""
    import re

    pattern = r"SOURCE\s+\d+\s*\nDocument:\s*([^\n]+)\s*\nPage:\s*([^\n]+)\s*(?:\nSection:\s*[^\n]+)?\s*\nChunk:\s*([^\n]+)\s*\n(.*?)(?=(?:\nSOURCE\s+\d+|\Z))"
    matches = list(re.finditer(pattern, context_block, flags=re.DOTALL))

    extracted_chunks = []
    if matches:
        for m in matches:
            doc_name = m.group(1).strip()
            page_num = m.group(2).strip()
            chunk_id = m.group(3).strip()
            text = m.group(4).strip()
            # Clean encoding artifacts
            text = (
                text.replace("\ufffd", "'")
                .replace("’", "'")
                .replace("‘", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\u2013", "-")
                .replace("\u2014", " - ")
                .replace("•", "-")
            )
            extracted_chunks.append({
                "document": doc_name,
                "page": page_num,
                "chunk_id": chunk_id,
                "text": text,
            })
    else:
        lines = []
        for line in context_block.splitlines():
            sline = line.strip()
            if not sline or sline.startswith(("SOURCE ", "Document:", "Page:", "Chunk:")):
                continue
            lines.append(sline)
        cleaned = "\n".join(lines).strip()
        if cleaned:
            extracted_chunks.append({
                "document": "Document",
                "page": "1",
                "chunk_id": "",
                "text": cleaned,
            })

    if not extracted_chunks:
        return "I could not find sufficient information in the indexed documents to answer this question reliably."

    seen_paras = set()
    key_points = []
    main_summary = []

    for chunk in extracted_chunks:
        paragraphs = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()]
        for p in paragraphs:
            normalized = " ".join(p.split())
            if len(normalized) < 25 or normalized[:60] in seen_paras:
                continue
            seen_paras.add(normalized[:60])

            if (
                normalized.lower().startswith(("project ", "chapter ", "overview", "part ", "what you're", "what you’re"))
                or (":" in normalized[:40] and not normalized.startswith("http"))
            ):
                key_points.append((chunk["page"], normalized))
            else:
                main_summary.append((chunk["page"], normalized))

    body_lines = []

    lead = main_summary[0] if main_summary else (key_points[0] if key_points else None)
    if lead:
        body_lines.append(lead[1])

    remaining_points = key_points + [p for p in main_summary if p != lead]
    if remaining_points:
        body_lines.append("**Key Details from Document Context:**")
        for page, pt in remaining_points[:5]:
            if ":" in pt[:45]:
                parts = pt.split(":", 1)
                body_lines.append(f"- **{parts[0].strip()}**: {parts[1].strip()} *(Page {page})*")
            else:
                snippet = pt[:250] + ("..." if len(pt) > 250 else "")
                body_lines.append(f"- {snippet} *(Page {page})*")

    return "\n\n".join(body_lines)
