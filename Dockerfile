# Backend image (FastAPI + RAG pipeline).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for PyMuPDF etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY prompts ./prompts
COPY frontend ./frontend
RUN mkdir -p data/documents data/sample_documents data/evaluation qdrant_storage

ENV API_URL="http://127.0.0.1:8001"

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 127.0.0.1 --port 8001 & streamlit run frontend/app.py --server.port ${PORT:-8000} --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"]
