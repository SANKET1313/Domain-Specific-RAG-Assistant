"""POST /upload — validate, persist, and ingest a document."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import UploadResponse
from app.core.config import get_settings
from app.core.security import FileValidationError, safe_filename, validate_upload
from app.ingestion.indexer import ingest_file
from app.ingestion.loader import ExtractionError

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    content = await file.read()
    try:
        ext = validate_upload(file.filename or "", len(content))
    except FileValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    os.makedirs(settings.documents_dir, exist_ok=True)
    name = safe_filename(file.filename or f"upload{ext}")
    dest = Path(settings.documents_dir) / name
    dest.write_bytes(content)

    try:
        summary = ingest_file(str(dest), document_name=name)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}") from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"Ingestion configuration error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(**summary)
