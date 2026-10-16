"""Document loading, page-by-page text extraction, and hybrid OCR.

Supports:
- Digital PDFs with table and layout preservation.
- Fully scanned PDFs via NVIDIA Vision OCR or local RapidOCR fallback.
- Mixed PDFs (processing page-by-page, avoiding unnecessary Vision API calls for digital pages).
- Memory-efficient streaming rendering (never loading all page images simultaneously).
- Word, TXT, and Markdown files.
- Detailed extraction diagnostics and summary telemetry.
"""
from __future__ import annotations

import base64
import gc
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Lazily initialized RapidOCR singleton
_RAPID_OCR_INSTANCE = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _sanitize_ocr_text(text: str) -> str:
    """Sanitize OCR output to valid unicode printable characters without broken encodings."""
    if not text:
        return ""
    # Filter out null characters and unprintable control characters, keep newlines and tabs
    cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t ")
    return cleaned.strip()


class ExtractionError(RuntimeError):
    """Raised when text cannot be extracted from a document or extraction critically fails."""


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    method: str  # "digital" | "vision_ocr" | "local_ocr" | "mixed_ocr" | "blank" | "unrecoverable"
    char_count: int
    word_count: int
    dpi_used: int = 0
    warning: Optional[str] = None
    status: str = "digital"  # "digital" | "recovered" | "partial" | "blank" | "unrecoverable"
    confidence: float = 1.0
    recovery_reason: Optional[str] = None


@dataclass
class DocumentExtractionSummary:
    path: str
    total_pages: int = 0
    pages_digital: int = 0
    pages_recovered: int = 0
    pages_blank: int = 0
    pages_unrecoverable: int = 0
    pages_ocr_vision: int = 0
    pages_ocr_local: int = 0
    pages_failed: int = 0
    total_chars: int = 0
    total_words: int = 0
    document_type: str = "digital"  # "digital" | "scanned" | "mixed" | "empty"
    pages_requiring_attention: list[int] = field(default_factory=list)
    recovery_log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _get_local_ocr_engine():
    """Lazily load and cache RapidOCR instance."""
    global _RAPID_OCR_INSTANCE
    if _RAPID_OCR_INSTANCE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID_OCR_INSTANCE = RapidOCR()
            logger.info("Initialized local RapidOCR engine successfully.")
        except Exception as exc:
            logger.warning("RapidOCR could not be initialized: %s", exc)
            return None
    return _RAPID_OCR_INSTANCE


def _ocr_page_with_local(pixmap_bytes: bytes) -> str:
    """Run local OCR using RapidOCR (or pytesseract if available) with layout reconstruction."""
    engine = _get_local_ocr_engine()
    if engine is not None:
        try:
            result, _ = engine(pixmap_bytes)
            if result:
                # result is a list of [box, text, confidence]
                # Reconstruct lines sorted by vertical (y) then horizontal (x) coordinate
                sorted_boxes = sorted(result, key=lambda b: (round(b[0][0][1] / 15) * 15, b[0][0][0]))
                lines: list[str] = []
                current_y_group = None
                current_line: list[str] = []

                for box in sorted_boxes:
                    y_top = round(box[0][0][1] / 15) * 15
                    text = box[1].strip()
                    if not text:
                        continue
                    if current_y_group is None or current_y_group == y_top:
                        current_line.append(text)
                        current_y_group = y_top
                    else:
                        lines.append(" ".join(current_line))
                        current_line = [text]
                        current_y_group = y_top
                if current_line:
                    lines.append(" ".join(current_line))

                return _sanitize_ocr_text("\n".join(lines))
        except Exception as exc:
            logger.warning("RapidOCR inference failed: %s", exc)

    # Secondary fallback: pytesseract if installed
    try:
        import io
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(pixmap_bytes))
        text = pytesseract.image_to_string(img)
        if text and text.strip():
            return _sanitize_ocr_text(text)
    except Exception:
        pass

    return ""


VISION_PRIMARY_PROMPT = (
    "You are an expert document extraction engine. Transcribe and extract ALL text, headings, "
    "bullet points, numbered lists, tables, labels, and metadata from this document page image "
    "into clean Markdown format.\n"
    "Guidelines:\n"
    "- Maintain the exact original reading order.\n"
    "- Transcribe complete tables using standard Markdown table syntax (| Header 1 | Header 2 |) with all rows intact.\n"
    "- Preserve headings and subheadings using Markdown headers (#, ##, ###).\n"
    "- Preserve bulleted lists (-) and numbered lists (1., 2.).\n"
    "- Extract everything verbatim. Do NOT summarize, shorten, or omit content.\n"
    "- Do NOT invent, assume, or hallucinate missing text. Only transcribe visible content."
)

VISION_RECOVERY_PROMPT = (
    "Transcribe every piece of readable text, title, heading, table, bullet point, or label "
    "from this document page image verbatim into Markdown.\n"
    "Carefully extract text from low-contrast areas, stylized fonts, banners, covers, or degraded scans. "
    "Preserve structure, tables, and line breaks. Do not summarize or guess missing characters."
)


def is_page_blank(page, pixmap=None) -> bool:
    """Determine whether a PDF page is genuinely blank.
    
    A page is considered genuinely blank if:
    1. It has no digital text.
    2. It has no embedded images.
    3. It has no visible vector drawings.
    4. Its rendered pixmap is uniform (standard deviation of pixel values < 2.0).
    """
    # 1. Digital text check
    if hasattr(page, "get_text") and page.get_text().strip():
        return False

    # 2. Embedded images check
    if hasattr(page, "get_images") and len(page.get_images()) > 0:
        return False

    # 3. Vector drawings check
    if hasattr(page, "get_drawings"):
        drawings = page.get_drawings()
        for d in drawings:
            fill = d.get("fill")
            color = d.get("color")
            if fill and fill != (1.0, 1.0, 1.0) and fill != (1, 1, 1):
                return False
            if color and color != (1.0, 1.0, 1.0) and color != (1, 1, 1):
                return False

    # 4. Visual pixel variance check on rendered pixmap
    if pixmap is not None:
        try:
            import numpy as np
            samples = pixmap.samples
            arr = np.frombuffer(samples, dtype=np.uint8)
            if float(arr.std()) < 2.0:
                return True
            return False
        except Exception:
            try:
                samples = pixmap.samples
                if len(samples) > 200:
                    step = max(1, len(samples) // 500)
                    subset = samples[::step]
                    if max(subset) - min(subset) < 5:
                        return True
            except Exception:
                pass

    return False


def _preprocess_image_for_ocr(img_bytes: bytes) -> bytes:
    """Enhance contrast and sharpness for difficult-to-read, low-contrast, or degraded scans."""
    try:
        import io
        from PIL import Image, ImageEnhance

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.6)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.4)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception as exc:
        logger.debug("Image preprocessing skipped: %s", exc)
        return img_bytes


def _ocr_page_with_vision(
    pixmap_bytes: bytes,
    prompt: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """Send page image to configured vision LLM (NVIDIA NIM or OpenAI)."""
    settings = get_settings()
    img_b64 = base64.b64encode(pixmap_bytes).decode("utf-8")
    actual_prompt = prompt or VISION_PRIMARY_PROMPT
    req_timeout = timeout or getattr(settings, "ocr_timeout", 60)

    # 1. NVIDIA NIM Vision model
    if settings.nvidia_api_key and settings.nvidia_api_key not in (
        "YOUR_NEW_NVIDIA_API_KEY",
        "your_nvidia_api_key_here",
        "test-nvidia-key",
    ):
        base_url = (settings.nvidia_base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }
        vision_model = (
            settings.llm_model
            if (settings.llm_model and "vision" in settings.llm_model.lower())
            else "meta/llama-3.2-11b-vision-instruct"
        )
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": actual_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": getattr(settings, "llm_max_tokens", 2500),
            "temperature": 0.1,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if resp.status_code == 200:
                data = resp.json()
                return _sanitize_ocr_text(data["choices"][0]["message"]["content"])
            safe_err = resp.text[:200].replace(settings.nvidia_api_key, "[REDACTED]")
            logger.warning("NVIDIA Vision API returned HTTP %s: %s", resp.status_code, safe_err)
        except Exception as exc:
            safe_exc = str(exc).replace(settings.nvidia_api_key, "[REDACTED]")
            logger.warning("NVIDIA Vision API network error: %s", safe_exc)

    # 2. OpenAI Vision fallback
    if settings.openai_api_key and not settings.openai_api_key.startswith("YOUR_"):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": actual_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 2500,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=req_timeout)
            if resp.status_code == 200:
                data = resp.json()
                return _sanitize_ocr_text(data["choices"][0]["message"]["content"])
            safe_err = resp.text[:200].replace(settings.openai_api_key, "[REDACTED]")
            logger.warning("OpenAI Vision API returned HTTP %s: %s", resp.status_code, safe_err)
        except Exception as exc:
            safe_exc = str(exc).replace(settings.openai_api_key, "[REDACTED]")
            logger.warning("OpenAI Vision API error: %s", safe_exc)

    raise RuntimeError("No configured vision service succeeded (timeout, rate limit, or invalid credentials).")


def _recover_page_content(
    pixmap_bytes: bytes,
    high_res_bytes: Optional[bytes] = None,
    engine: Optional[str] = None,
    prefer_local: bool = False,
) -> tuple[str, str, str, float, str]:
    """Execute multi-tier extraction & recovery pipeline.
    
    Returns:
        tuple of (extracted_text, method_used, status, confidence, note)
        status: "recovered" | "partial" | "unrecoverable"
    """
    settings = get_settings()
    active_engine = (engine or settings.ocr_engine).lower()

    if active_engine == "none":
        return "", "none", "unrecoverable", 0.0, "OCR engine set to none"

    # Fast path for embedded images when prefer_local is requested
    if prefer_local or active_engine == "local":
        try:
            loc_text = _ocr_page_with_local(pixmap_bytes)
            if len(loc_text.split()) < 15 and high_res_bytes is not None:
                high_text = _ocr_page_with_local(high_res_bytes)
                if len(high_text.split()) > len(loc_text.split()):
                    loc_text = high_text
            if loc_text and len(loc_text.strip()) >= 15:
                return loc_text.strip(), "local_ocr", "recovered", 0.85, "Recovered via local OCR engine"
        except Exception:
            pass

    # Tier 1: Try Vision Model extraction if enabled
    if active_engine in ("auto", "vision"):
        try:
            v_text = _ocr_page_with_vision(pixmap_bytes, prompt=VISION_PRIMARY_PROMPT)
            if v_text and len(v_text.strip()) >= 15:
                return v_text.strip(), "vision_ocr", "recovered", 0.95, "Successfully recovered via AI Vision extraction"
        except Exception as v_err:
            logger.info("Vision primary attempt failed: %s. Retrying with enhanced preprocessing...", v_err)

        # Tier 1 Retry: Preprocess image (contrast/sharpness) and retry with secondary prompt
        try:
            target_bytes = high_res_bytes or pixmap_bytes
            enhanced_bytes = _preprocess_image_for_ocr(target_bytes)
            v_text_retry = _ocr_page_with_vision(enhanced_bytes, prompt=VISION_RECOVERY_PROMPT)
            if v_text_retry and len(v_text_retry.strip()) >= 15:
                return v_text_retry.strip(), "vision_ocr", "recovered", 0.90, "Recovered via AI Vision retry with enhanced contrast"
        except Exception as retry_err:
            logger.info("Vision retry attempt failed: %s. Falling back to local OCR engine...", retry_err)

    # Tier 2: Local OCR fallback (RapidOCR / pytesseract)
    if active_engine in ("auto", "local"):
        try:
            loc_text = _ocr_page_with_local(pixmap_bytes)
            # If text is sparse and high_res is available, try high_res
            if len(loc_text.split()) < 15 and high_res_bytes is not None:
                high_text = _ocr_page_with_local(high_res_bytes)
                if len(high_text.split()) > len(loc_text.split()):
                    loc_text = high_text

            if loc_text and len(loc_text.strip()) >= 15:
                return loc_text.strip(), "local_ocr", "recovered", 0.85, "Recovered via local OCR engine"
            elif loc_text and len(loc_text.strip()) > 0:
                return loc_text.strip(), "local_ocr", "partial", 0.60, f"Partially recovered via local OCR ({len(loc_text.strip())} chars)"
        except Exception as loc_err:
            logger.warning("Local OCR fallback failed: %s", loc_err)

    return "", "failed", "unrecoverable", 0.0, "Document scan contained no decipherable text across AI Vision and OCR passes"


def _ocr_page(
    pixmap_bytes: bytes,
    high_res_bytes: Optional[bytes] = None,
    prefer_local: bool = False,
) -> tuple[str, str]:
    """Execute OCR according to configured strategy (auto, vision, local).
    
    Maintains backward compatibility with older callers and tests.
    """
    settings = get_settings()
    engine = "local" if prefer_local and settings.ocr_engine != "vision" else settings.ocr_engine
    text, method, _, _, _ = _recover_page_content(
        pixmap_bytes=pixmap_bytes,
        high_res_bytes=high_res_bytes,
        engine=engine,
    )
    return text, method


def _extract_page_digital(page) -> str:
    """Extract digital text from a PyMuPDF page, preserving reading order and table structure without duplicating text."""
    # 1. Detect structured tables and their bounding boxes
    tables_found: list[tuple[tuple[float, float, float, float], str]] = []
    try:
        tabs = page.find_tables()
        if tabs and tabs.tables:
            for tab in tabs.tables:
                try:
                    md_table = tab.to_markdown()
                    if md_table and md_table.strip():
                        tables_found.append((tuple(tab.bbox), md_table.strip()))
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Extract layout blocks: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    page_height = page.rect.height if hasattr(page, "rect") else 842.0

    # Elements to sort by vertical reading order: (y0, x0, element_type, text)
    ordered_elements: list[tuple[float, float, str, str]] = []

    for b in blocks:
        if len(b) >= 5 and b[4]:
            t = b[4].strip()
            if not t:
                continue

            bx0, by0, bx1, by1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])

            # Check if this block's center falls inside any detected table
            mid_x = (bx0 + bx1) / 2
            mid_y = (by0 + by1) / 2
            in_table = False
            for (tx0, ty0, tx1, ty1), _ in tables_found:
                if (tx0 - 2 <= mid_x <= tx1 + 2) and (ty0 - 2 <= mid_y <= ty1 + 2):
                    in_table = True
                    break

            if not in_table:
                # Detect footnotes near bottom of page
                is_footnote = by0 >= (page_height * 0.88) and bool(
                    re.match(r"^(?:[\*\d†‡]\s+|\[\d+\]\s+|footnote\b)", t, re.IGNORECASE)
                )
                el_type = "footnote" if is_footnote else "text"
                ordered_elements.append((by0, bx0, el_type, t))

    # Add the extracted tables at their respective vertical position
    for (tx0, ty0, tx1, ty1), md_table in tables_found:
        ordered_elements.append((ty0, tx0, "table", md_table))

    # Sort elements by top-to-bottom (y0), then left-to-right (x0)
    ordered_elements.sort(key=lambda item: (item[0], item[1]))

    regular_parts: list[str] = []
    footnote_parts: list[str] = []

    for _, _, el_type, content in ordered_elements:
        if el_type == "footnote":
            footnote_parts.append(f"[Footnote: {content}]")
        else:
            regular_parts.append(content)

    full_pieces = list(regular_parts)
    if footnote_parts:
        full_pieces.append("\n".join(footnote_parts))

    return "\n\n".join(full_pieces).strip()


def load_pages_detailed(path: str) -> tuple[list[ExtractedPage], DocumentExtractionSummary]:
    """Extract text from a document page-by-page, returning detailed diagnostic metrics."""
    ext = Path(path).suffix.lower()
    summary = DocumentExtractionSummary(path=path)
    settings = get_settings()

    # 1. Standalone image files (.png, .jpg, .jpeg, .webp, .bmp, .tiff)
    if ext in IMAGE_EXTENSIONS:
        try:
            image_bytes = Path(path).read_bytes()
            rec_txt, method, status, conf, note = _recover_page_content(
                image_bytes, engine=settings.ocr_engine
            )
            if not rec_txt.strip():
                raise ExtractionError(f"Image '{Path(path).name}' contains no readable text or recovery failed.")

            p = ExtractedPage(
                page_number=1,
                text=rec_txt.strip(),
                method=method,
                char_count=len(rec_txt.strip()),
                word_count=len(rec_txt.strip().split()),
                dpi_used=0,
                status="recovered",
                confidence=conf,
                recovery_reason=note,
            )
            summary.total_pages = 1
            summary.pages_recovered = 1
            if method == "vision_ocr":
                summary.pages_ocr_vision = 1
            elif method == "local_ocr":
                summary.pages_ocr_local = 1
            else:
                summary.pages_failed = 1
            summary.total_chars = p.char_count
            summary.total_words = p.word_count
            summary.document_type = "scanned"
            summary.recovery_log.append(f"Page 1: {note} ({p.char_count} chars).")
            return [p], summary
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Failed to process image '{Path(path).name}': {exc}") from exc

    # 2. Non-PDF text/docx documents
    if ext != ".pdf":
        raw_pages = load_pages(path)
        pages = [
            ExtractedPage(
                page_number=pn,
                text=t,
                method="digital",
                char_count=len(t),
                word_count=len(t.split()),
                status="digital",
                confidence=1.0,
            )
            for pn, t in raw_pages
        ]
        summary.total_pages = len(pages)
        summary.pages_digital = len(pages)
        summary.total_chars = sum(p.char_count for p in pages)
        summary.total_words = sum(p.word_count for p in pages)
        summary.document_type = "digital"
        return pages, summary

    # 3. PDF documents (digital, scanned, and mixed with embedded images/tables)
    import pymupdf

    settings = get_settings()
    extracted_pages: list[ExtractedPage] = []

    try:
        with pymupdf.open(path) as doc:
            summary.total_pages = len(doc)
            if summary.total_pages == 0:
                raise ExtractionError(f"PDF '{Path(path).name}' contains 0 pages.")

            for i, page in enumerate(doc, start=1):
                digital_text = _extract_page_digital(page)

                # Check for significant embedded images (tables, diagrams, charts, infographics)
                has_significant_images = False
                embedded_ocr_texts: list[str] = []
                page_vision_used = False
                page_local_used = False

                img_list = page.get_images(full=True)
                # Only check for embedded infographics/tables if page has sufficient digital text (mixed page).
                # Scanned or image-only pages go directly to high-DPI full page recovery in Scenario D.
                if img_list and len(digital_text) >= settings.ocr_trigger_char_threshold:
                    page_area = page.rect.width * page.rect.height
                    for img_item in img_list:
                        xref = img_item[0]
                        rects = page.get_image_rects(xref)
                        if rects and page_area > 0:
                            img_area = rects[0].width * rects[0].height
                            # Significant image: occupies >= 10% of page area or >= 80,000 pt²
                            if (img_area / page_area) >= 0.10 or img_area >= 80000:
                                has_significant_images = True
                                try:
                                    img_data = doc.extract_image(xref)
                                    if img_data and img_data.get("image"):
                                        img_bytes = img_data["image"]
                                        rec_res, method, _, _, _ = _recover_page_content(
                                            img_bytes, engine=settings.ocr_engine, prefer_local=True
                                        )
                                        if rec_res and len(rec_res.strip()) > 10:
                                            embedded_ocr_texts.append(rec_res.strip())
                                            if method == "vision_ocr":
                                                page_vision_used = True
                                            elif method == "local_ocr":
                                                page_local_used = True
                                except Exception as img_exc:
                                    logger.warning(
                                        "Failed to OCR embedded image xref %d on page %d: %s",
                                        xref, i, img_exc,
                                    )

                # Scenario A: Mixed page with digital text AND embedded image OCR text
                if len(digital_text) >= settings.ocr_trigger_char_threshold and embedded_ocr_texts:
                    combined_text = digital_text
                    for img_txt in embedded_ocr_texts:
                        combined_text = f"{combined_text}\n\n[Extracted Table / Image Content]\n{img_txt}".strip()
                    summary.pages_digital += 1
                    summary.pages_recovered += 1
                    if page_vision_used:
                        summary.pages_ocr_vision += 1
                    elif page_local_used:
                        summary.pages_ocr_local += 1

                    summary.recovery_log.append(
                        f"Page {i}: Recovered embedded table/image content ({sum(len(t) for t in embedded_ocr_texts)} chars)."
                    )
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=combined_text,
                            method="mixed_ocr",
                            char_count=len(combined_text),
                            word_count=len(combined_text.split()),
                            dpi_used=0,
                            status="recovered",
                            confidence=0.95,
                            recovery_reason="Extracted digital text + embedded table/image content",
                        )
                    )
                    continue

                # Scenario B: Pure digital text page (sufficient text and no significant image content)
                if len(digital_text) >= settings.ocr_trigger_char_threshold and not has_significant_images:
                    summary.pages_digital += 1
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=digital_text,
                            method="digital",
                            char_count=len(digital_text),
                            word_count=len(digital_text.split()),
                            status="digital",
                            confidence=1.0,
                        )
                    )
                    continue

                # Scenario C: Scanned page with already extracted embedded image text
                if embedded_ocr_texts:
                    ocr_text = "\n\n".join(embedded_ocr_texts)
                    final_text = ocr_text or digital_text
                    final_method = "vision_ocr" if page_vision_used else ("local_ocr" if page_local_used else "digital")
                    summary.pages_recovered += 1
                    if final_method == "vision_ocr":
                        summary.pages_ocr_vision += 1
                    elif final_method == "local_ocr":
                        summary.pages_ocr_local += 1
                    else:
                        summary.pages_digital += 1

                    summary.recovery_log.append(f"Page {i}: Recovered content from embedded images ({len(final_text)} chars).")
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=final_text,
                            method=final_method,
                            char_count=len(final_text),
                            word_count=len(final_text.split()),
                            dpi_used=0,
                            status="recovered",
                            confidence=0.95,
                            recovery_reason="Extracted content from embedded images",
                        )
                    )
                    continue

                # Scenario D: Minimal/empty digital text — inspect for blank vs image/scan content
                logger.info(
                    "Page %d of '%s' has %d chars (below threshold %d). Inspecting for content...",
                    i, Path(path).name, len(digital_text), settings.ocr_trigger_char_threshold,
                )

                pix = None
                pix_high = None
                try:
                    pix = page.get_pixmap(dpi=settings.ocr_default_dpi)

                    # 1. Distinguish genuinely blank pages
                    if is_page_blank(page, pix):
                        logger.info("Page %d of '%s' identified as genuinely blank.", i, Path(path).name)
                        summary.pages_blank += 1
                        summary.recovery_log.append(f"Page {i}: Verified genuinely blank page.")
                        extracted_pages.append(
                            ExtractedPage(
                                page_number=i,
                                text="",
                                method="blank",
                                char_count=0,
                                word_count=0,
                                dpi_used=settings.ocr_default_dpi,
                                status="blank",
                                confidence=1.0,
                                recovery_reason="Genuinely blank page (no text, drawings, or visual content)",
                            )
                        )
                        continue

                    # 2. Non-blank page — run recovery pipeline
                    pix_bytes = pix.tobytes("png")
                    high_bytes = None
                    if settings.ocr_high_dpi > settings.ocr_default_dpi:
                        try:
                            pix_high = page.get_pixmap(dpi=settings.ocr_high_dpi)
                            high_bytes = pix_high.tobytes("png")
                        except Exception:
                            high_bytes = None

                    rec_txt, rec_method, rec_status, rec_conf, rec_note = _recover_page_content(
                        pix_bytes,
                        high_res_bytes=high_bytes,
                        engine=settings.ocr_engine,
                    )

                    final_text = rec_txt or digital_text
                    if final_text.strip():
                        if rec_status == "recovered":
                            summary.pages_recovered += 1
                            summary.recovery_log.append(f"Page {i}: {rec_note} ({len(final_text)} chars).")
                        else:
                            summary.pages_recovered += 1
                            summary.pages_requiring_attention.append(i)
                            summary.warnings.append(f"Page {i}: Partial recovery ({len(final_text)} chars) - {rec_note}.")

                        if rec_method == "vision_ocr":
                            summary.pages_ocr_vision += 1
                        elif rec_method == "local_ocr":
                            summary.pages_ocr_local += 1
                        else:
                            summary.pages_digital += 1

                        extracted_pages.append(
                            ExtractedPage(
                                page_number=i,
                                text=final_text,
                                method=rec_method if rec_txt else "digital",
                                char_count=len(final_text),
                                word_count=len(final_text.split()),
                                dpi_used=settings.ocr_default_dpi,
                                status=rec_status if rec_txt else "digital",
                                confidence=rec_conf if rec_txt else 1.0,
                                recovery_reason=rec_note,
                            )
                        )
                    else:
                        # Non-blank page where all recovery methods failed
                        summary.pages_unrecoverable += 1
                        summary.pages_failed += 1
                        summary.pages_requiring_attention.append(i)
                        warn_msg = f"Page {i}: Unrecoverable - {rec_note}."
                        summary.warnings.append(warn_msg)
                        extracted_pages.append(
                            ExtractedPage(
                                page_number=i,
                                text="",
                                method="unrecoverable",
                                char_count=0,
                                word_count=0,
                                dpi_used=settings.ocr_default_dpi,
                                warning=warn_msg,
                                status="unrecoverable",
                                confidence=0.0,
                                recovery_reason=rec_note,
                            )
                        )
                except Exception as exc:
                    summary.pages_unrecoverable += 1
                    summary.pages_failed += 1
                    summary.pages_requiring_attention.append(i)
                    warn_msg = f"Page {i}: Extraction error - {exc}"
                    summary.warnings.append(warn_msg)
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=i,
                            text=digital_text,
                            method="unrecoverable",
                            char_count=len(digital_text),
                            word_count=len(digital_text.split()),
                            warning=str(exc),
                            status="unrecoverable",
                            confidence=0.0,
                            recovery_reason=f"Processing exception: {exc}",
                        )
                    )
                finally:
                    # Clean up native image memory immediately for each page
                    del pix
                    del pix_high
                    gc.collect()

    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Failed to process PDF '{Path(path).name}': {exc}") from exc

    # Compute summary totals and document classification
    summary.total_chars = sum(p.char_count for p in extracted_pages)
    summary.total_words = sum(p.word_count for p in extracted_pages)

    if summary.pages_digital == summary.total_pages:
        summary.document_type = "digital"
    elif summary.pages_recovered > 0 and summary.pages_digital > 0:
        summary.document_type = "mixed"
    elif summary.pages_recovered > 0 and summary.pages_digital == 0:
        summary.document_type = "scanned"
    elif summary.pages_blank == summary.total_pages:
        summary.document_type = "empty"
    else:
        summary.document_type = "digital" if summary.pages_digital > 0 else "scanned"

    # Reject document only if 100% of pages produced zero text
    if summary.total_chars == 0:
        err_msg = "; ".join(summary.warnings) if summary.warnings else "No readable text found."
        raise ExtractionError(
            f"PDF '{Path(path).name}' contains no extractable text. {err_msg}"
        )

    return extracted_pages, summary


def load_pages(path: str) -> list[tuple[int, str]]:
    """Extract text page-by-page. page_number is 1-based; 1 for non-paged files."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".pdf":
            pages, _ = load_pages_detailed(path)
            return [(p.page_number, p.text) for p in pages]
        if ext in IMAGE_EXTENSIONS:
            pages, _ = load_pages_detailed(path)
            return [(p.page_number, p.text) for p in pages]
        if ext == ".docx":
            return _load_docx(path)
        if ext in {".txt", ".md"}:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            return [(1, text)]
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Failed to extract text from {path}: {exc}") from exc
    raise ExtractionError(f"Unsupported extension for extraction: {ext}")


def _load_docx(path: str) -> list[tuple[int, str]]:
    """Extract text and tables from DOCX in document order, preserving headings, lists, tables, and pages."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    pages: list[list[str]] = [[]]
    current_page = pages[0]
    page_words = 0

    # Iterate through body elements in document order
    for element in document.element.body:
        if element.tag.endswith("p"):
            p = Paragraph(element, document)
            raw_t = p.text.strip()
            if not raw_t:
                continue

            # Detect manual or rendered page breaks
            has_break = False
            for run in p.runs:
                if "w:br" in run._r.xml and 'type="page"' in run._r.xml:
                    has_break = True
                    break
            if "w:lastRenderedPageBreak" in element.xml:
                has_break = True

            style = getattr(p.style, "name", "").lower()
            if style.startswith("heading 1"):
                line = f"# {raw_t}"
            elif style.startswith("heading 2"):
                line = f"## {raw_t}"
            elif style.startswith("heading 3"):
                line = f"### {raw_t}"
            elif "list" in style or "bullet" in style:
                line = f"- {raw_t}" if not raw_t.startswith(("-", "*", "•")) else raw_t
            else:
                line = raw_t

            if has_break and current_page:
                pages.append([])
                current_page = pages[-1]
                page_words = 0

            current_page.append(line)
            page_words += len(line.split())
            # Logical pagination if no explicit breaks: ~450 words per page
            if page_words >= 450:
                pages.append([])
                current_page = pages[-1]
                page_words = 0

        elif element.tag.endswith("tbl"):
            t = Table(element, document)
            md_lines: list[str] = []
            rows = t.rows
            if rows:
                header_cells = [cell.text.strip().replace("\n", " ") for cell in rows[0].cells]
                md_lines.append("| " + " | ".join(header_cells) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
                for row in rows[1:]:
                    row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    md_lines.append("| " + " | ".join(row_cells) + " |")

            if md_lines:
                tbl_text = "\n".join(md_lines)
                current_page.append(tbl_text)
                page_words += len(tbl_text.split())
                if page_words >= 450:
                    pages.append([])
                    current_page = pages[-1]
                    page_words = 0

    results: list[tuple[int, str]] = []
    for pn, lines_list in enumerate(pages, start=1):
        content = "\n\n".join(lines_list).strip()
        if content:
            results.append((pn, content))

    if not results:
        raise ExtractionError("DOCX contained no extractable text.")
    return results
