FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir faiss-cpu==1.12.0

COPY scripts/ ./scripts/
COPY data/ ./data/
COPY knowledge_base/ ./knowledge_base/

EXPOSE 8000 8001 8002 8003
