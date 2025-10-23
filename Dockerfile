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
  PORT=7860

# --- System dependencies ---
# Added wget + tar to prevent 'wget: not found' error
RUN apt-get update && apt-get install -y --no-install-recommends \
  tini wget curl ca-certificates tar \
  && rm -rf /var/lib/apt/lists/*

# --- Non-root user ---
RUN useradd -m -u 1000 appuser

WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# --- Project files ---
COPY . .

# --- Persistent directories & permissions ---
RUN mkdir -p /data/chroma_db /data/.huggingface /data/corpus \
  && chown -R appuser:appuser /data /app

# --- Optional: bootstrap script permissions ---
RUN if [ -f "bootstrap.sh" ]; then chmod +x bootstrap.sh; fi

# --- (Optional) OpenVSCode Server setup ---
# Comment out if not needed, but fixed version included
ARG OPENVSCODE_VERSION=v1.101.2
RUN wget -O /tmp/openvscode-server.tar.gz \
  "https://github.com/gitpod-io/openvscode-server/releases/download/${OPENVSCODE_VERSION}/openvscode-server-${OPENVSCODE_VERSION#v}-linux-x64.tar.gz" \
  && tar -xzf /tmp/openvscode-server.tar.gz -C /opt \
  && rm /tmp/openvscode-server.tar.gz \
  && mv /opt/openvscode-server-* /opt/openvscode-server \
  && chown -R 1000:1000 /opt/openvscode-server

USER appuser
EXPOSE 7860

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

# --- App start command ---
# CMD ["bash", "bootstrap.sh"]
CMD ["python", "app.py"]
