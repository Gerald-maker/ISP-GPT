#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, json, time, re
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download, HfApi
from langchain_chroma import Chroma
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import httpx
import urllib.parse
from contextlib import suppress

warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

log = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Env ---
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/tmp/chroma_db")
COLLECTION_NAME = ENV("RAG_COLLECTION", "career_gpt")
DATASET_ID = ENV("RAG_DATASET_ID", "internationalscholarsprogram/DOC")
DATA_REV = ENV("RAG_DATASET_REVISION", "main")
CORPUS_DIR = ENV("RAG_CORPUS_DIR", "/data/corpus")
STATE_FILE = ENV("RAG_STATE_FILE", "/data/.state.json")
PORT = int(ENV("PORT", "7860"))
HOST = ENV("HOST", "0.0.0.0")

# --- Secret hygiene helpers (prevents 'Illegal header value' 502s) ---
_WS = re.compile(r"\s+")  # any whitespace

def _clean_secret(val: Optional[str], name: str) -> str:
    """
    Strip all whitespace (including CR/LF/tabs/non-breaking) from ends AND middle.
    If any whitespace remains (shouldn't), reject. Ensures tokens are one contiguous string.
    """
    v = (val or "")
    # remove CR/LF specifically then trim
    v = v.replace("\r", "").replace("\n", "").strip()
    # reject if any whitespace still inside
    if _WS.search(v):
        raise RuntimeError(f"{name} contains whitespace; fix your secret (paste as a single line with no spaces)")
    # reject obvious quotes
    if v.startswith(("'", '"')) or v.endswith(("'", '"')):
        raise RuntimeError(f"{name} appears to include quotes; remove wrapping quotes")
    return v

def _auth_header(token: str) -> Dict[str, str]:
    if not token:
        raise RuntimeError("Missing API token for upstream call")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Managed endpoints + gen params
HF_PRIMARY_ENDPOINT = (ENV("HF_PRIMARY_ENDPOINT", "") or "").rstrip("/")
HF_FALLBACK_ENDPOINT = (ENV("HF_FALLBACK_ENDPOINT", "") or "").rstrip("/")
HF_API_TOKEN = _clean_secret(ENV("HF_API_TOKEN", ""), "HF_API_TOKEN") if ENV("HF_API_TOKEN") else ""
FRIENDLI_API_KEY = _clean_secret(ENV("FRIENDLI_API_KEY", ""), "FRIENDLI_API_KEY") if ENV("FRIENDLI_API_KEY") else ""
REQUEST_TIMEOUT = int(ENV("REQUEST_TIMEOUT_MS", "60000")) / 1000.0
GEN_MAX_TOKENS = int(ENV("GEN_MAX_TOKENS", "512"))
GEN_TEMPERATURE = float(ENV("GEN_TEMPERATURE", "0.4"))
MAX_RETRIES = int(ENV("UPSTREAM_MAX_RETRIES", "3"))
BACKOFF_SECS = float(ENV("UPSTREAM_BACKOFF_SECS", "0.5"))

def _normalize_url(u: str) -> str:
    # ensure scheme present
    if u and not urllib.parse.urlparse(u).scheme:
        u = "https://" + u
    return u.rstrip("/")

HF_PRIMARY_ENDPOINT = _normalize_url(HF_PRIMARY_ENDPOINT)
HF_FALLBACK_ENDPOINT = _normalize_url(HF_FALLBACK_ENDPOINT)

# Quick redacted preview for debugging (safe)
def _prefix(s: str) -> str:
    return (s[:6] + "***") if s else "<none>"

log.info("HF endpoint set: %s", bool(HF_PRIMARY_ENDPOINT))
log.info("HF token prefix: %s", _prefix(HF_API_TOKEN))
log.info("Fallback endpoint set: %s", bool(HF_FALLBACK_ENDPOINT))
log.info("Friendli key prefix: %s", _prefix(FRIENDLI_API_KEY))

# --- Embeddings + Vector DB ---
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},  # cosine-ready vectors
)
os.makedirs(DB_DIR, exist_ok=True)
client = PersistentClient(path=DB_DIR)
# global vectordb guarded by a lock for atomic swapping
_vectordb_lock = threading.RLock()
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k: int = 4):
    # Use MMR with a larger candidate set for better diversity
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(20, 5 * k)})

# --- Prompt ---
prompt = ChatPromptTemplate.from_template(
    "Use <context> to answer. If not sure, say 'I don’t know.'\n\nQ: {question}\n<context>\n{context}\n</context>"
)
parser = StrOutputParser()

# -------- Managed Endpoint Caller (HF primary + Friendli fallback) --------
def _payload_openai_style(content: str) -> Dict[str, Any]:
    return {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "temperature": GEN_TEMPERATURE,
        "max_tokens": GEN_MAX_TOKENS,
        "stream": False,
    }

def _payload_tgi_generate(content: str) -> Dict[str, Any]:
    return {"inputs": content, "parameters": {"max_new_tokens": GEN_MAX_TOKENS, "temperature": GEN_TEMPERATURE}}

def _extract_text_from_openai(resp_json: Dict[str, Any]) -> str:
    with suppress(Exception):
        return resp_json["choices"][0]["message"]["content"]
    return json.dumps(resp_json)  # last resort

def _extract_text_from_tgi(resp_json: Any) -> str:
    if isinstance(resp_json, dict) and "generated_text" in resp_json:
        return resp_json["generated_text"]
    if isinstance(resp_json, list) and resp_json and isinstance(resp_json[0], dict) and "generated_text" in resp_json[0]:
        return resp_json[0]["generated_text"]
    return json.dumps(resp_json)

def _http_post_with_retries(url: str, headers: Dict[str, str], json_body: Dict[str, Any]) -> httpx.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as s:
                r = s.post(url, headers=headers, json=json_body)
                # Treat 429/5xx as retryable
                if r.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"Retryable {r.status_code}", request=r.request, response=r)
                return r
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECS * attempt)
            else:
                raise
    # Should not reach here
    raise last_exc or RuntimeError("Unknown upstream error")

def call_hf_managed(formatted_prompt: str) -> str:
    if not HF_PRIMARY_ENDPOINT:
        raise RuntimeError("HF_PRIMARY_ENDPOINT is not set")
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set")
    headers = _auth_header(HF_API_TOKEN)  # sanitized, newline-free
    # Try OpenAI-compatible first
    url_chat = f"{HF_PRIMARY_ENDPOINT}/v1/chat/completions"
    try:
        r = _http_post_with_retries(url_chat, headers, _payload_openai_style(formatted_prompt))
        if r.status_code == 200:
            return _extract_text_from_openai(r.json())
        # If not found/allowed, try TGI
        if r.status_code in (404, 405):
            raise httpx.HTTPStatusError("Switching to /generate", request=r.request, response=r)
        # Non-OK
        raise RuntimeError(f"HF /v1/chat/completions error {r.status_code}: {r.text[:200]}")
    except Exception as e_first:
        # Try TGI classic
        url_gen = f"{HF_PRIMARY_ENDPOINT}/generate"
        r2 = _http_post_with_retries(url_gen, headers, _payload_tgi_generate(formatted_prompt))
        if r2.status_code == 200:
            return _extract_text_from_tgi(r2.json())
        raise RuntimeError(f"HF /generate error {r2.status_code}: {r2.text[:200]} | prior: {str(e_first)[:200]}")

def call_friendli_fallback(formatted_prompt: str) -> str:
    if not HF_FALLBACK_ENDPOINT:
        raise RuntimeError("HF_FALLBACK_ENDPOINT is not set")
    if not FRIENDLI_API_KEY:
        raise RuntimeError("FRIENDLI_API_KEY is not set")
    headers = _auth_header(FRIENDLI_API_KEY)  # sanitized, newline-free
    path = "/v1/chat/completions" if not HF_FALLBACK_ENDPOINT.endswith("/v1/chat/completions") else ""
    url = f"{HF_FALLBACK_ENDPOINT}{path}"
    r = _http_post_with_retries(url, headers, _payload_openai_style(formatted_prompt))
    if r.status_code == 200:
        return _extract_text_from_openai(r.json())
    raise RuntimeError(f"Friendli error {r.status_code}: {r.text[:200]}")

def _redact(s: str) -> str:
    # best-effort redaction for tokens if they ever appear in upstream error strings
    if not s:
        return s
    for t in (HF_API_TOKEN, FRIENDLI_API_KEY):
        if t:
            s = s.replace(t, _prefix(t))
    return s

def generate_answer(question: str, context: str) -> str:
    formatted = prompt.format(question=question, context=context)
    # Try HF primary → Friendli fallback
    try:
        return call_hf_managed(formatted)
    except Exception as e1:
        log.warning("HF primary failed: %s", _redact(str(e1)))
        try:
            return call_friendli_fallback(formatted)
        except Exception as e2:
            log.error("Friendli fallback failed: %s", _redact(str(e2)))
            # bubble up a redacted error
            raise RuntimeError(f"Primary & fallback failed. Primary: {str(e1)[:200]}; Fallback: {str(e2)[:200]}")

# -------- Dataset sync + indexing --------
def sync_pdfs() -> str:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATA_REV,
        local_dir=CORPUS_DIR,
        local_dir_use_symlinks=False,
        max_workers=4,
    )
    info = HfApi().repo_info(repo_id=DATASET_ID, repo_type="dataset", revision=DATA_REV)
    return info.sha

def list_pdfs(root: str) -> List[str]:
    out: List[str] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(r, f))
    return out

def load_docs(paths: List[str]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for p in paths:
        try:
            pages = PyPDFLoader(p).load()
            for pg in pages:
                docs.extend(splitter.split_documents([pg]))
        except Exception as e:
            log.warning("Skipping corrupt or unreadable PDF %s: %s", p, e)
    return docs

def rebuild_index(docs):
    # delete & recreate collection safely
    with suppress(Exception):
        client.delete_collection(name=COLLECTION_NAME)
    new_client = PersistentClient(path=DB_DIR)
    new_db = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=new_client)
    # batch adds
    for i in range(0, len(docs), 32):
        new_db.add_documents(docs[i : i + 32])
    return new_db

def reindex(force: bool = False) -> Dict[str, Any]:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    new_sha = sync_pdfs()
    old_sha = None
    if os.path.exists(STATE_FILE):
        with suppress(Exception):
            old_sha = json.load(open(STATE_FILE)).get("dataset_sha")
    if force or new_sha != old_sha:
        pdfs = list_pdfs(CORPUS_DIR)
        docs = load_docs(pdfs)
        global vectordb
        new_db = rebuild_index(docs)
        # atomic swap under lock
        with _vectordb_lock:
            vectordb = new_db
        with open(STATE_FILE, "w") as f:
            json.dump({"dataset_sha": new_sha, "docs": len(docs), "ts": int(time.time())}, f)
        return {"reindexed": True, "commit": new_sha, "docs": len(docs)}
    return {"reindexed": False, "commit": new_sha}

# -------- FastAPI --------
app = FastAPI(title="Career GPT RAG API", version="1.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

INDEX_STATUS = {"state": "idle", "detail": "", "last_commit": None}

def warmup():
    global INDEX_STATUS
    try:
        INDEX_STATUS.update({"state": "syncing", "detail": "starting"})
        info = reindex(force=False)
        INDEX_STATUS.update({"state": "ready", "detail": str(info), "last_commit": info.get("commit")})
        log.info("Index ready %s", info)
    except Exception as e:
        INDEX_STATUS.update({"state": "error", "detail": str(e)})
        log.error("Warmup error: %s", e)

@app.on_event("startup")
def _startup():
    threading.Thread(target=warmup, daemon=True).start()

@app.get("/health")
def health():
    cfg = {
        "hf_primary_set": bool(HF_PRIMARY_ENDPOINT),
        "hf_token_set": bool(HF_API_TOKEN),
        "fallback_set": bool(HF_FALLBACK_ENDPOINT),
        "friendli_key_set": bool(FRIENDLI_API_KEY),
        "index_status": INDEX_STATUS,
    }
    return {"status": "ok", "config": cfg}

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Ask):
    # Retrieve context
    try:
        with _vectordb_lock:  # ensure we don't swap vectordb mid-query
            docs = build_retriever(k=4).invoke(q.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)[:200]}")

    ctx = "\n\n".join([d.page_content for d in docs])
    try:
        text = generate_answer(q.question, ctx)
        return {
            "answer": text,
            "sources": [
                {"path": d.metadata.get("source"), "page": d.metadata.get("page")}
                for d in docs
            ],
        }
    except Exception as e:
        # Clean, redacted 502 without leaking newliney tokens
        msg = _redact(str(e))[:500]
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {msg}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
