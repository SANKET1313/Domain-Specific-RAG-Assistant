"""GET /documents, DELETE /documents/{id}, batch deletion, reindexing, and local indexing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    BatchDeleteRequest,
    BatchDeleteResponse,
    DocumentModel,
    IndexLocalRequest,
    LocalFileModel,
    UploadResponse,
)
from app.core.config import get_settings
from app.core.security import safe_filename
from app.ingestion.indexer import ingest_file
from app.ingestion.loader import ExtractionError
from app.vectorstore.factory import get_vector_store

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=list[DocumentModel])
def list_documents() -> list[DocumentModel]:
    store = get_vector_store()
    return [DocumentModel(**d) for d in store.list_documents()]


@router.delete("/documents", response_model=dict[str, str])
def clear_all_documents() -> dict[str, str]:
    """Delete all documents from the vector store index."""
    store = get_vector_store()
    try:
        store.delete_all()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {exc}") from exc
    return {"status": "cleared", "message": "All documents removed from vector index."}


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    delete_file: bool = Query(False, description="Also remove file from disk if present"),
) -> dict[str, str]:
    store = get_vector_store()
    settings = get_settings()
    
    doc_name = None
    if delete_file:
        for d in store.list_documents():
            if d.get("document_id") == document_id:
                doc_name = d.get("document_name")
                break

    try:
        store.delete_by_document(document_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

    if delete_file and doc_name:
        try:
            target = Path(settings.documents_dir) / safe_filename(doc_name)
            if target.exists() and target.is_file():
                target.unlink(missing_ok=True)
        except Exception:
            pass

    return {"status": "deleted", "document_id": document_id}


@router.post("/documents/batch-delete", response_model=BatchDeleteResponse)
def batch_delete_documents(req: BatchDeleteRequest) -> BatchDeleteResponse:
    """Delete multiple documents in one request."""
    store = get_vector_store()
    deleted: list[str] = []
    failed: list[dict[str, str]] = []

    for doc_id in req.document_ids:
        try:
            store.delete_by_document(doc_id)
            deleted.append(doc_id)
        except Exception as exc:  # noqa: BLE001
            failed.append({"document_id": doc_id, "error": str(exc)})

    return BatchDeleteResponse(deleted=deleted, failed=failed, status="success" if not failed else "partial")


@router.get("/documents/local-files", response_model=list[LocalFileModel])
def list_local_files() -> list[LocalFileModel]:
    """List documents stored on local server disk with indexing status."""
    settings = get_settings()
    store = get_vector_store()
    indexed_docs = store.list_documents()
    indexed_names = {d.get("document_name") for d in indexed_docs if d.get("document_name")}

    files: list[LocalFileModel] = []
    seen: set[str] = set()

    for directory in [Path(settings.documents_dir), Path("data/sample_documents")]:
        if not directory.exists():
            continue
        for p in directory.glob("*"):
            if p.is_file() and not p.name.startswith(".") and p.name not in seen:
                seen.add(p.name)
                files.append(
                    LocalFileModel(
                        name=p.name,
                        size_bytes=p.stat().st_size,
                        is_indexed=p.name in indexed_names,
                        source_folder=directory.name,
                    )
                )

    return sorted(files, key=lambda f: f.name.lower())


@router.post("/documents/index-local", response_model=UploadResponse)
def index_local_document(req: IndexLocalRequest) -> UploadResponse:
    """Index an existing file from the server's documents directory."""
    settings = get_settings()
    filename = safe_filename(req.filename)

    doc_path = Path(settings.documents_dir) / filename
    if not doc_path.exists():
        sample_path = Path("data/sample_documents") / filename
        if sample_path.exists():
            doc_path = sample_path
        else:
            raise HTTPException(status_code=404, detail=f"File '{req.filename}' not found on server disk.")

    try:
        summary = ingest_file(str(doc_path), document_name=filename)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return UploadResponse(**summary)


@router.post("/documents/{document_id}/reindex", response_model=UploadResponse)
def reindex_document(document_id: str) -> UploadResponse:
    """Reindex an existing document by re-chunking and re-embedding its source file."""
    store = get_vector_store()
    settings = get_settings()

    docs = store.list_documents()
    target = next((d for d in docs if d.get("document_id") == document_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found in index.")

    doc_name = target.get("document_name")
    if not doc_name:
        raise HTTPException(status_code=400, detail="Document metadata is missing document_name.")

    doc_path = Path(settings.documents_dir) / safe_filename(doc_name)
    if not doc_path.exists():
        sample_path = Path("data/sample_documents") / safe_filename(doc_name)
        if sample_path.exists():
            doc_path = sample_path
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Source file '{doc_name}' is no longer present on server disk.",
            )

    # Delete previous chunk vectors
    try:
        store.delete_by_document(document_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to clear old index entries: {exc}") from exc

    # Ingest afresh
    try:
        summary = ingest_file(str(doc_path), document_name=doc_name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {exc}") from exc

    return UploadResponse(**summary)


@router.get("/documents/{document_id}/inspect")
def inspect_document(document_id: str):
    """Inspect all chunks, page numbers, sections, and metadata stored for a document."""
    store = get_vector_store()
    docs = store.list_documents()
    target = next((d for d in docs if d.get("document_id") == document_id), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found in index.")

    chunk_payloads = store.get_document_chunks(document_id)
    settings = get_settings()
    doc_name = target.get("document_name", "unknown")
    original_path = Path(settings.documents_dir) / safe_filename(doc_name)
    if not original_path.exists():
        sample_p = Path("data/sample_documents") / safe_filename(doc_name)
        if sample_p.exists():
            original_path = sample_p

    # Locate generated Markdown representation
    md_file = Path(settings.documents_dir) / f"{original_path.stem}.md"
    md_content = None
    if md_file.exists():
        try:
            md_content = md_file.read_text(encoding="utf-8")
        except Exception:
            pass
    elif chunk_payloads and chunk_payloads[0].get("markdown_path"):
        stored_md_p = Path(chunk_payloads[0]["markdown_path"])
        if stored_md_p.exists():
            try:
                md_content = stored_md_p.read_text(encoding="utf-8")
                md_file = stored_md_p
            except Exception:
                pass

    chunks = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "chunk_text": c.get("chunk_text", ""),
            "page_number": c.get("page_number"),
            "page_start": c.get("page_start", c.get("page_number")),
            "page_end": c.get("page_end", c.get("page_start", c.get("page_number"))),
            "chunk_position": c.get("chunk_position", 0),
            "section_id": c.get("section_id", ""),
            "section_heading": c.get("section_heading", ""),
            "subsection_heading": c.get("subsection_heading", ""),
            "section_chunk_index": c.get("section_chunk_index", 0),
            "token_count": c.get("token_count", 0),
        }
        for c in chunk_payloads
    ]

    # Group detected sections
    sections: dict[str, dict[str, Any]] = {}
    for c in chunks:
        sid = c.get("section_id")
        sh = c.get("section_heading")
        if sid or sh:
            key = sid or sh
            if key not in sections:
                sections[key] = {
                    "section_id": sid or "",
                    "section_heading": sh or "",
                    "first_page": c.get("page_start", c.get("page_number")),
                    "last_page": c.get("page_end", c.get("page_number")),
                    "chunk_count": 0,
                }
            sections[key]["chunk_count"] += 1
            p_val = c.get("page_end") or c.get("page_number")
            if p_val is not None:
                sections[key]["last_page"] = max(sections[key]["last_page"] or 0, p_val)

    return {
        "document_id": document_id,
        "document_name": doc_name,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "sections": list(sections.values()),
        "markdown_path": str(md_file) if md_file.exists() else None,
        "markdown_content": md_content,
        "original_file_path": str(original_path) if original_path.exists() else None,
        "metadata": target,
    }
