"""End-to-end integration tests for document ingestion, listing, and RAG querying."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify both /health endpoints return 200 OK and expected status."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_root_docs_endpoint():
    """Verify / root returns service info and docs link."""
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()
    assert r.json().get("docs") == "/docs"


def test_upload_and_list_documents(monkeypatch, tmp_path):
    """Integration test: upload a text document, verify ingestion response and document listing."""
    import app.api.routes_upload as ru
    import app.api.routes_documents as rd

    # Mock ingest_file to avoid external network/GPU embedding during unit tests
    def fake_ingest(path, document_name=None, document_id=None):
        return {
            "document_id": "doc_test_123",
            "document_name": document_name or "remote_policy.txt",
            "chunk_count": 3,
            "status": "indexed",
            "upload_time": "2026-10-11T00:00:00Z",
            "document_type": "digital",
            "total_pages": 1,
            "pages_digital": 1,
            "pages_ocr": 0,
            "warnings": [],
        }

    monkeypatch.setattr(ru, "ingest_file", fake_ingest)

    doc_content = b"The quarterly remote work stipend is 500 dollars."
    files = {"file": ("remote_policy.txt", doc_content, "text/plain")}

    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "doc_test_123"
    assert data["chunk_count"] == 3
    assert data["document_name"] == "remote_policy.txt"


def test_query_grounded_response_with_sources(monkeypatch):
    """Integration test: query endpoint returns answer grounded in retrieved document chunks."""
    import app.api.routes_query as rq

    def fake_answer_query(query, flt=None):
        return {
            "answer": "The quarterly remote work stipend is $500 according to company policy.",
            "sources": [
                {
                    "document": "remote_policy.txt",
                    "page": 1,
                    "chunk_id": "chunk_0001",
                    "score": 0.95,
                }
            ],
            "sufficient_context": True,
            "metrics": {
                "retrieval_latency_ms": 12.5,
                "llm_latency_ms": 180.2,
                "total_latency_ms": 192.7,
            },
        }

    monkeypatch.setattr(rq, "answer_query", fake_answer_query)

    query_payload = {
        "query": "How much is the remote work stipend?",
        "document_id": "doc_test_123",
    }
    resp = client.post("/query", json=query_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "$500" in body["answer"]
    assert body["sufficient_context"] is True
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document"] == "remote_policy.txt"
    assert body["sources"][0]["page"] == 1
