"""Ingestion orchestrator: file -> pages -> clean -> chunk -> embed -> store.

Produces complete chunk metadata (document_id, document_name, chunk_id, page_number,
page_start, page_end, chunk_position, section_heading, source, domain, token_count)
with stable identifiers, deduplication, batching, and diagnostic telemetry.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.embeddings.factory import get_embedder
from app.ingestion.chunker import chunk_markdown, chunk_pages
from app.ingestion.pdf_to_markdown import convert_document_to_markdown
from app.vectorstore.base import VectorRecord
from app.vectorstore.factory import get_vector_store

logger = get_logger("ingestion")


def ingest_file(
    path: str,
    document_name: str | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Ingest one document, replace stale chunks if updating, and index into vector store."""
    settings = get_settings()
    t_start = time.perf_counter()
    document_name = document_name or Path(path).name

    # Stable deterministic document identifier based on document name if not provided
    if not document_id:
        doc_hash = hashlib.sha256(document_name.encode("utf-8")).hexdigest()[:12]
        document_id = f"doc_{doc_hash}"

    upload_time = datetime.now(timezone.utc).isoformat()

    # 1. Convert Document to Canonical Structured Markdown & Validate Completeness
    conversion_res = convert_document_to_markdown(
        path,
        document_name=document_name,
        document_id=document_id,
    )
    md_text = conversion_res.markdown_text
    md_path = conversion_res.markdown_path
    extraction_summary = conversion_res.summary
    validation = conversion_res.validation

    # 2. Chunk using the canonical Markdown representation
    chunks = chunk_markdown(
        md_text,
        document_id=document_id,
        source_filename=document_name,
    )
    if not chunks:
        raise ValueError(f"No chunks produced for '{document_name}'; content appears empty.")

    # Chunk token statistics
    token_counts = [c.token_count for c in chunks]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else 0.0

    # 4. Prepare embedder & store
    embedder = get_embedder()
    store = get_vector_store()
    store.ensure_collection(embedder.dim)

    # 5. Purge any previous/stale chunks for this document before indexing new ones
    try:
        store.delete_by_document(document_id)
    except Exception as exc:
        logger.warning("Attempted to clear prior chunks for %s: %s", document_id, exc)

    # 6. Generate embeddings with safe batching
    texts = [c.text for c in chunks]
    batch_size = 32
    all_vectors: list[list[float]] = []
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = embedder.embed_texts(batch)
        if len(res.vectors) != len(batch):
            raise RuntimeError(
                f"Embedding vector count mismatch: expected {len(batch)} vectors, got {len(res.vectors)}"
            )
        all_vectors.extend(res.vectors)
        total_tokens += res.token_count

    # 7. Create deterministic vector records with complete metadata
    records: list[VectorRecord] = []
    for chunk, vector in zip(chunks, all_vectors):
        if len(vector) != embedder.dim:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match embedder dimension {embedder.dim}"
            )

        chunk_id = f"chunk_{chunk.index:04d}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{chunk_id}"))

        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "chunk_id": chunk_id,
            "chunk_text": chunk.text,
            "page_number": chunk.page_start,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_position": chunk.chunk_position,
            "section_id": chunk.section_id or "",
            "section_heading": chunk.section_heading or "",
            "subsection_heading": chunk.subsection_heading or "",
            "section_chunk_index": chunk.section_chunk_index,
            "source": document_name,
            "domain": settings.domain_name,
            "upload_time": upload_time,
            "token_count": chunk.token_count,
            "markdown_path": str(md_path),
        }
        records.append(VectorRecord(id=point_id, vector=vector, metadata=metadata))

    # 8. Upsert vectors in batches to avoid network or memory spikes
    upsert_batch_size = 64
    for i in range(0, len(records), upsert_batch_size):
        batch_records = records[i : i + upsert_batch_size]
        store.upsert(batch_records)

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        "document_ingested",
        extra={"extra": {
            "document_id": document_id,
            "document_name": document_name,
            "total_pages": extraction_summary.total_pages,
            "pages_digital": extraction_summary.pages_digital,
            "pages_ocr": extraction_summary.pages_ocr_vision + extraction_summary.pages_ocr_local,
            "pages_failed": extraction_summary.pages_failed,
            "chunk_count": len(records),
            "chunk_tokens_min": min_tokens,
            "chunk_tokens_max": max_tokens,
            "chunk_tokens_avg": avg_tokens,
            "embedding_tokens": total_tokens,
            "vectors_inserted": len(records),
            "document_type": extraction_summary.document_type,
            "ingestion_time_ms": total_time_ms,
            "status": "success",
            "markdown_path": str(md_path),
        }},
    )

    return {
        "document_id": document_id,
        "document_name": document_name,
        "chunk_count": len(records),
        "status": "indexed",
        "upload_time": upload_time,
        "embedding_tokens": total_tokens,
        "total_pages": extraction_summary.total_pages,
        "pages_digital": extraction_summary.pages_digital,
        "pages_ocr": extraction_summary.pages_ocr_vision + extraction_summary.pages_ocr_local,
        "document_type": extraction_summary.document_type,
        "warnings": extraction_summary.warnings,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "avg_tokens": avg_tokens,
        "ingestion_time_ms": total_time_ms,
        "markdown_path": str(md_path),
    }
