"""Tests for direct image ingestion and embedded PDF image extraction."""
from __future__ import annotations

import io
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from app.ingestion.loader import load_pages, load_pages_detailed, IMAGE_EXTENSIONS
from app.core.security import validate_upload


def test_allowed_image_extensions():
    """Verify image extensions are validated and allowed."""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        assert ext in IMAGE_EXTENSIONS
        norm = validate_upload(f"test_image{ext}", 1024)
        assert norm == ext


def test_direct_image_extraction_local(tmp_path: Path):
    """Create a synthetic PNG image and verify load_pages extracts text via local OCR."""
    img_path = tmp_path / "sample_chart.png"
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Tool: Claude AI", fill=(0, 0, 0))
    draw.text((20, 60), "Purpose: Long-form writing and document editing", fill=(0, 0, 0))
    draw.text((20, 100), "Free Tier: Yes - claude.ai", fill=(0, 0, 0))
    img.save(str(img_path))

    pages, summary = load_pages_detailed(str(img_path))
    assert len(pages) == 1
    assert summary.total_pages == 1
    assert summary.pages_ocr_local >= 1 or summary.pages_ocr_vision >= 1
    assert pages[0].page_number == 1
    assert "Claude" in pages[0].text or "writing" in pages[0].text.lower()


def test_load_pages_supports_images(tmp_path: Path):
    """Verify load_pages returns tuples of (1, text) for image files."""
    img_path = tmp_path / "diagram.jpg"
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "Sales Funnel Architecture Overview", fill=(0, 0, 0))
    img.save(str(img_path))

    tuples = load_pages(str(img_path))
    assert len(tuples) == 1
    assert tuples[0][0] == 1
    assert len(tuples[0][1]) > 0
