# ----------------------------------------
# ISP Retrieval (RAG) API - Hugging Face Space (Docker)
# ----------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_ROOT_USER_ACTION=ignore \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  HF_HOME=/data/.huggingface \
  RAG_DB_DIR=/tmp/chroma_db \
  RAG_CORPUS_DIR=/data/corpus \
  RAG_DATASET_ID=internationalscholarsprogram/DOC \
  RAG_DATASET_REVISION=main \
  PORT=7860 \
  TOKENIZERS_PARALLELISM=false \
  HF_HUB_DISABLE_TELEMETRY=1 \
  CUDA_VISIBLE_DEVICES="" \
  OMP_NUM_THREADS=1 \
  ORT_LOG_SEVERITY_LEVEL=3 \
  ORT_FORCE_CPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  tini curl ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /tmp/chroma_db /data/.huggingface /data/corpus \
  && chmod -R 777 /tmp /data /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash","-lc","python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
