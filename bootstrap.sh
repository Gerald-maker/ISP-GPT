#!/usr/bin/env bash
set -euo pipefail

# Respect Spaces’ injected PORT; default to 7860 locally
: "${PORT:=7860}"
: "${RAG_PORT:=${PORT}}"
: "${RAG_DB_DIR:=/data/chroma_db}"
: "${RAG_CORPUS_DIR:=/data/corpus}"
: "${RAG_FORCE_REFRESH:=0}"   # set to 1 to force reindex on startup

echo "[bootstrap] PORT=${PORT}  RAG_DB_DIR=${RAG_DB_DIR}  RAG_CORPUS_DIR=${RAG_CORPUS_DIR}  FORCE_REFRESH=${RAG_FORCE_REFRESH}"

# Ensure persistent storage paths exist (Spaces mounts /data)
mkdir -p "${RAG_DB_DIR}" "${RAG_CORPUS_DIR}" /data/.huggingface || true
chmod -R 777 /data || true

# Optional: warn if HF token is missing (private dataset or Inference needs it)
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  echo "[bootstrap] WARNING: HUGGINGFACEHUB_API_TOKEN is not set. Private datasets or HF Inference will fail."
fi

# Start the API (app.py does dataset sync + reindex automatically on startup)
# If you want to *force* a rebuild each boot, export RAG_FORCE_REFRESH=1 and we'll hit /refresh once it's up.
echo "[bootstrap] Starting API server..."
python app.py &

APP_PID=$!

# Optionally trigger a forced refresh once the server is listening
if [ "${RAG_FORCE_REFRESH}" = "1" ]; then
  echo "[bootstrap] Waiting for API to come up to trigger /refresh ..."
  for i in {1..30}; do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "[bootstrap] API is up. Forcing reindex via /refresh"
      # If you protect endpoints with RAG_API_KEY, add: -H "Authorization: Bearer ${RAG_API_KEY}"
      curl -fsS -X POST "http://127.0.0.1:${PORT}/refresh" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
fi

# Bring python to the foreground so container signals are handled
wait "${APP_PID}"
