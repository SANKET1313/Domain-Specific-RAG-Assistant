"""POST /query — run the RAG pipeline and return an answer + sources."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import QueryRequest, QueryResponse
from app.rag.pipeline import answer_query

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    flt = {"document_id": req.document_id} if req.document_id else None
    try:
        if req.debug:
            result = answer_query(req.query, flt=flt, debug=True)
        else:
            result = answer_query(req.query, flt=flt)
    except TypeError:
        result = answer_query(req.query, flt=flt)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    return QueryResponse(**result)
