# ----------------------------------------
# Career GPT RAG API - Hugging Face Space (Docker)
# ----------------------------------------
FROM python:3.11-slim-bookworm

# (Optional) bump this to force a fresh rebuild on HF when needed
LABEL build.cachebuster="2025-10-22-01"

# ---- Core env (persistent paths + defaults) ----
ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  HF_HOME=/data/.huggingface \
  RAG_DB_DIR=/data/chroma_db \
  RAG_CORPUS_DIR=/data/corpus \
  RAG_DATASET_ID=internationalscholarsprogram/DOC \
  RAG_DATASET_REVISION=main \
  RAG_PORT=7860 \
  PORT=7860

# ---- System deps ----
# Add wget and tar to avoid "command not found" errors in scripts/layers
RUN apt-get update && apt-get install -y --no-install-recommends \
  tini curl wget tar ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ---- Create a non-root user (safer) ----
RUN useradd -m -u 1000 appuser

# ---- Workdir ----
WORKDIR /app

# ---- Python deps (copy first to leverage cache) ----
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# ---- Project files ----
COPY . .

# ---- Persistent dirs & permissions (/data is mounted by Spaces) ----
RUN mkdir -p /data/chroma_db /data/.huggingface /data/corpus \
  && chown -r appuser:appuser /data /app || chown -R appuser:appuser /data /app

# If you use a start script, ensure it's executable (optional)
RUN if [ -f "bootstrap.sh" ]; then chmod +x bootstrap.sh; fi

# ---- Drop privileges ----
USER appuser

# ---- Networking ----
EXPOSE 7860

# ---- Healthcheck (hits /health) ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# ---- PID 1 = tini for clean shutdowns ----
ENTRYPOINT ["/usr/bin/tini", "--"]

# ---- Start command ----
# If you’re using bootstrap.sh, uncomment the next line and comment out the python line.
# CMD ["bash", "bootstrap.sh"]
CMD ["python", "app.py"]
