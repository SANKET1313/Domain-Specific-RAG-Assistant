"""Lightweight security helpers: file validation and safe temp handling."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from app.core.config import get_settings


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_upload(filename: str, size_bytes: int) -> str:
    """Validate extension and size; return the normalized lower-case extension."""
    settings = get_settings()
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_ext_set:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_ext_set)}"
        )
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes <= 0:
        raise FileValidationError("Uploaded file is empty.")
    if size_bytes > max_bytes:
        raise FileValidationError(
            f"File too large ({size_bytes} bytes). Limit is {settings.max_upload_mb} MB."
        )
    return ext


def safe_filename(filename: str) -> str:
    """Strip path components and unsafe characters to prevent path traversal."""
    base = os.path.basename(filename)
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)


def write_temp_file(content: bytes, suffix: str) -> str:
    """Write bytes to a secure temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content)
    return path
