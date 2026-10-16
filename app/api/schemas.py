"""Pydantic request/response models for the API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None  # optional metadata filter
    debug: Optional[bool] = False  # optional diagnostic debug flag


class SourceModel(BaseModel):
    document: str
    page: Any = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_id: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    sufficient_context: bool
    metrics: dict[str, Any]
    contexts: list[str] = []
    supporting_evidence: Optional[str] = None
    source_document_id: Optional[str] = None
    source_document_name: Optional[str] = None
    source_pages: Optional[list[Any]] = None
    insufficient_evidence: bool = False
    diagnostic_report: Optional[dict[str, Any]] = None
    debug: Optional[dict[str, Any]] = None


class DocumentModel(BaseModel):
    document_id: str
    document_name: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    upload_time: Optional[str] = None
    chunk_count: int = 0
    status: str = "indexed"


class UploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    status: str
    upload_time: str
    embedding_tokens: int = 0
    total_pages: int = 1
    pages_digital: int = 1
    pages_ocr: int = 0
    pages_recovered: int = 0
    pages_blank: int = 0
    pages_unrecoverable: int = 0
    pages_requiring_attention: list[int] = []
    document_type: str = "digital"
    warnings: list[str] = []
    recovery_log: list[str] = []
    ingestion_summary: Optional[dict[str, Any]] = None
    markdown_path: Optional[str] = None


class DocumentInspectionChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    page_number: Any = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_id: Optional[str] = None
    section_heading: Optional[str] = None
    subsection_heading: Optional[str] = None
    token_count: Optional[int] = None


class DocumentInspectionResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    chunks: list[DocumentInspectionChunk]
    markdown_path: Optional[str] = None
    markdown_content: Optional[str] = None
    original_file_path: Optional[str] = None
    sections: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}


class BatchDeleteRequest(BaseModel):
    document_ids: list[str]


class BatchDeleteResponse(BaseModel):
    deleted: list[str]
    failed: list[dict[str, str]] = []
    status: str = "success"


class IndexLocalRequest(BaseModel):
    filename: str


class LocalFileModel(BaseModel):
    name: str
    size_bytes: int
    is_indexed: bool
    source_folder: str
