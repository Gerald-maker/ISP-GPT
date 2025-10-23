# ----------------------------------------
# Career GPT RAG API - Hugging Face Space (Docker)
# ----------------------------------------
FROM python:3.11-slim-bookworm

# --- Environment settings ---
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_ROOT_USER_ACTION=ignore \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  HF_HOME=/data/.huggingface \
  RAG_DB_DIR=/data/chroma_db \
  RAG_CORPUS_DIR=/data/corpus \
  RAG_DATASET_ID=internationalscholarsprogram/DOC \
  RAG_DATASET_REVISION=main \
  RAG_PORT=7860 \
  PORT=7860 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_DISABLE_TELEMETRY=1 \
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  ORT_LOG_SEVERITY_LEVEL=3

# NOTE:
# - Removed legacy Chroma envs (CHROMA_DB_IMPL, CHROMADB_TELEMETRY, ANONYMIZED_TELEMETRY)
#   since the new PersistentClient doesn’t use them.

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
  tini wget curl ca-certificates tar git \
  && rm -rf /var/lib/apt/lists/*

# --- (Optional) Non-root user (kept for reference) ---
RUN useradd -m -u 1000 appuser || true

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# --- Project files ---
COPY . .

# --- Persistent / writable directories ---
RUN mkdir -p /data/chroma_db /data/.huggingface /data/corpus /tmp/chroma_db \
  && chmod -R 777 /data /app /tmp

# Do NOT switch user; keep root so /data and /tmp are writable in Spaces
# USER appuser

EXPOSE 7860

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# --- App start command ---
# CMD ["bash", "bootstrap.sh"]
CMD ["python", "app.py"]
