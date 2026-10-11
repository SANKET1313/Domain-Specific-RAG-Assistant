"""API endpoint tests using FastAPI's TestClient (RAG pipeline mocked)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_shape():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "config" in body


def test_query_rejects_empty():
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400


def test_query_end_to_end_mocked(monkeypatch):
    import app.api.routes_query as rq

    def fake_answer(query, flt=None):
        return {
            "answer": "Mocked answer.",
            "sources": [{"document": "a.pdf", "page": 2, "chunk_id": "chunk_0001"}],
            "sufficient_context": True,
            "metrics": {"retrieval_latency_ms": 10, "llm_latency_ms": 20, "total_latency_ms": 30},
        }

    monkeypatch.setattr(rq, "answer_query", fake_answer)
    r = client.post("/query", json={"query": "What is the leave policy?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Mocked answer."
    assert r.json()["sources"][0]["document"] == "a.pdf"


def test_documents_endpoints(monkeypatch):
    import app.api.routes_documents as rd

    class DummyStore:
        def __init__(self):
            self.docs = [{"document_id": "doc_123", "document_name": "test.txt", "chunk_count": 5}]

        def list_documents(self):
            return self.docs

        def delete_by_document(self, doc_id):
            self.docs = [d for d in self.docs if d["document_id"] != doc_id]

        def delete_all(self):
            self.docs = []

    dummy = DummyStore()
    monkeypatch.setattr(rd, "get_vector_store", lambda: dummy)

    # 1. List
    r = client.get("/documents")
    assert r.status_code == 200
    assert len(r.json()) == 1

    # 2. Batch delete
    r = client.post("/documents/batch-delete", json={"document_ids": ["doc_123"]})
    assert r.status_code == 200
    assert "doc_123" in r.json()["deleted"]

    # 3. Clear all
    r = client.delete("/documents")
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"

    # 4. Local files
    r = client.get("/documents/local-files")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_upload_rejects_empty_file():
    files = {"file": ("empty.txt", b"", "text/plain")}
    r = client.post("/upload", files=files)
    assert r.status_code == 400
    assert "empty" in r.text.lower()


def test_upload_rejects_unsupported_extension():
    files = {"file": ("malicious.exe", b"malicious binary content", "application/octet-stream")}
    r = client.post("/upload", files=files)
    assert r.status_code == 400
    assert "unsupported file type" in r.text.lower()
