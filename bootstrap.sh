#!/usr/bin/env bash
set -euo pipefail

# Respect Spaces’ $PORT if provided
: "${PORT:=7860}"
: "${RAG_PORT:=${PORT}}"
: "${RAG_DB_DIR:=/data/chroma_db}"

echo "[bootstrap] Using PORT=${PORT}  RAG_DB_DIR=${RAG_DB_DIR}"

# Optional: pre-warm embeddings or index docs on first run
if [ -d "docs" ]; then
  echo "[bootstrap] Ingesting docs -> ${RAG_DB_DIR}"
  # light + SciPy-free providers recommended (bge or fastembed)
  python ingest.py --docs docs --db "${RAG_DB_DIR}" --embed-provider bge --device cpu || true
fi

echo "[bootstrap] Starting API server..."
# uvicorn picks up HOST/PORT from app.py; still pass here for clarity
python app.py
