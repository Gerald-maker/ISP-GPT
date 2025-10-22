# ----------------------------------------
# Career GPT RAG API - Hugging Face Space (Docker)
# ----------------------------------------
FROM python:3.11-slim-bookworm

# Core env
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  HF_HOME=/data/.huggingface \
  RAG_DB_DIR=/data/chroma_db \
  RAG_PORT=7860 \
  PORT=7860

# System deps (add more if your loaders need them)
# - tini: clean signal handling for FastAPI/uvicorn
# - git, curl: handy for debugging and HF pulls
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  curl \
  tini \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install -r requirements.txt

# Project files
COPY . .

# Make and open writable data dirs (Space persistent storage should mount /data)
RUN mkdir -p /data/chroma_db /data/.huggingface && chmod -R 777 /data

# ✅ Make bootstrap.sh executable inside the container
RUN chmod +x bootstrap.sh

# The Space routes traffic to $PORT; exposing is optional but harmless
EXPOSE 7860

# Healthcheck (optional but useful)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Use tini as PID 1 so SIGTERM/SIGINT cleanly stop uvicorn
ENTRYPOINT ["/usr/bin/tini", "--"]

# Entrypoint script
CMD ["bash", "bootstrap.sh"]
