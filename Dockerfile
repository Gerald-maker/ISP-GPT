# ----------------------------------------
# Career GPT RAG API - Hugging Face Space (Docker)
# ----------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  HF_HOME=/data/.huggingface \
  RAG_DB_DIR=/data/chroma_db \
  RAG_CORPUS_DIR=/data/corpus \
  RAG_DATASET_ID=internationalscholarsprogram/DOC \
  RAG_DATASET_REVISION=main \
  RAG_PORT=7860 \
  PORT=7860 \
  PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# System deps (no wget, no git)
RUN apt-get update && apt-get install -y --no-install-recommends \
  tini curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Persistent dirs & permissions
RUN mkdir -p /data/chroma_db /data/.huggingface /data/corpus \
  && chown -R appuser:appuser /data /app

# Optional: start script permissions
RUN if [ -f "bootstrap.sh" ]; then chmod +x bootstrap.sh; fi

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini","--"]
# CMD ["bash","bootstrap.sh"]
CMD ["python","app.py"]
