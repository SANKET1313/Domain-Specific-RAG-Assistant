# Container image for Domain-Specific RAG Application.
# Runs the FastAPI backend internally and serves the Streamlit UI on $PORT.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for PyMuPDF, OCR etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY prompts ./prompts
COPY frontend ./frontend
COPY .streamlit ./.streamlit
COPY run.py .
RUN mkdir -p data/documents data/sample_documents data/evaluation qdrant_storage

EXPOSE 8501 8000
CMD ["python", "run.py"]
