#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, json, re
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from huggingface_hub import snapshot_download, HfApi
from langchain_chroma import Chroma
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --------------------- Setup & Logging ---------------------
warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

log = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Lock to prevent Chroma collection being swapped/deleted while queries are running
VDB_LOCK = threading.RLock()

# --------------------- Env ---------------------
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/tmp/chroma_db")
COLLECTION_NAME = ENV("RAG_COLLECTION", "career_gpt")
DATASET_ID = ENV("RAG_DATASET_ID", "internationalscholarsprogram/DOC")
DATA_REV = ENV("RAG_DATASET_REVISION", "main")
CORPUS_DIR = ENV("RAG_CORPUS_DIR", "/data/corpus")
STATE_FILE = "/data/.state.json"
PORT = int(ENV("PORT", "7860"))
HOST = "0.0.0.0"

# Retrieval params
TOP_K = int(ENV("RAG_TOP_K", "4"))
MAX_CONTEXT_CHARS = int(ENV("RAG_MAX_CONTEXT_CHARS", "12000"))

# --------------------- Embeddings + Vector DB ---------------------
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
os.makedirs(DB_DIR, exist_ok=True)

# IMPORTANT: keep ONE persistent client for the lifetime of the process.
# Deleting collections while old handles exist causes InvalidCollectionException.
client = PersistentClient(path=DB_DIR)

# Create (or connect to) collection
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k: int = TOP_K):
    # MMR gives diverse context
    with VDB_LOCK:
        return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

# --------------------- Dataset sync + indexing ---------------------
def sync_pdfs() -> str:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=DATA_REV,
        local_dir=CORPUS_DIR,
        local_dir_use_symlinks=False
    )
    info = HfApi().repo_info(repo_id=DATASET_ID, repo_type="dataset", revision=DATA_REV)
    return info.sha

def list_pdfs(root: str) -> List[str]:
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f.lower().endswith(".pdf")]

def load_docs(paths: List[str]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for p in paths:
        for pg in PyPDFLoader(p).load():
            docs += splitter.split_documents([pg])
    return docs

def rebuild_index(docs):
    """
    Rebuild the collection WITHOUT deleting it.

    Why:
      Deleting the collection creates a new collection_id. Any in-flight or cached
      references to the old collection_id will crash with:
        InvalidCollectionException: Collection ... does not exist.
    """
    # Ensure collection exists
    coll = client.get_or_create_collection(COLLECTION_NAME)

    # Clear existing docs without deleting the collection
    try:
        existing = coll.get(include=[])
        ids = (existing or {}).get("ids") or []
        if ids:
            coll.delete(ids=ids)
    except Exception:
        # If collection is empty or get/delete isn't supported in a particular state, ignore
        pass

    # Re-wrap the existing collection through LangChain
    new_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    for i in range(0, len(docs), 32):
        new_db.add_documents(docs[i:i+32])

    return new_db

def reindex(force: bool = False):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    new_sha = sync_pdfs()
    old_sha = json.load(open(STATE_FILE))["dataset_sha"] if os.path.exists(STATE_FILE) else None

    if force or new_sha != old_sha:
        pdfs = list_pdfs(CORPUS_DIR)
        docs = load_docs(pdfs)

        with VDB_LOCK:
            global vectordb
            vectordb = rebuild_index(docs)

        json.dump({"dataset_sha": new_sha}, open(STATE_FILE, "w"))
        return {"reindexed": True, "commit": new_sha, "docs": len(docs)}

    return {"reindexed": False, "commit": new_sha}

# --------------------- FastAPI app ---------------------
app = FastAPI(title="ISP Retrieval (RAG) API", version="2.0.0")
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
        log.info(f"Index ready {info}")
    except Exception as e:
        INDEX_STATUS.update({"state": "error", "detail": str(e)})
        log.exception("Warmup failed")

@app.on_event("startup")
def _startup():
    threading.Thread(target=warmup, daemon=True).start()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "config": {
            "dataset_id": DATASET_ID,
            "dataset_rev": DATA_REV,
            "collection": COLLECTION_NAME,
            "top_k": TOP_K,
            "index_status": INDEX_STATUS,
        }
    }

# --------------------- Schemas ---------------------
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

class AskIn(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    top_k: Optional[int] = None  # optional override

    @property
    def text(self) -> str:
        t = (self.question or self.query or "").strip()
        t = CONTROL_CHARS_RE.sub(" ", t)
        if not t:
            raise ValueError("Provide 'question' or 'query'.")
        return t

class ContextItem(BaseModel):
    rank: int
    text: str
    source: Optional[str] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = {}

class AskOut(BaseModel):
    ok: bool = True
    question: str
    context: str
    contexts: List[ContextItem]

def _doc_to_item(d, rank: int) -> ContextItem:
    meta = d.metadata or {}
    # metadata keys vary by loader; try best-effort
    source = meta.get("source") or meta.get("file_path") or meta.get("file_name")
    page = meta.get("page")
    # some loaders store page as string
    try:
        if page is not None:
            page = int(page)
    except Exception:
        page = None
    return ContextItem(
        rank=rank,
        text=d.page_content,
        source=source,
        page=page,
        metadata=meta
    )

def _build_context_text(items: List[ContextItem], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    total = 0
    for it in items:
        header = f"[{it.rank}] source={it.source or 'unknown'} page={it.page if it.page is not None else 'n/a'}\n"
        chunk = header + it.text.strip()
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts).strip()

# --------------------- /ask (retrieval only) ---------------------
@app.post("/ask", response_model=AskOut)
def ask(q: AskIn):
    if INDEX_STATUS.get("state") != "ready":
        raise HTTPException(status_code=503, detail=f"Index not ready: {INDEX_STATUS}")

    k = int(q.top_k or TOP_K)
    if k < 1 or k > 12:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 12")

    docs = build_retriever(k).invoke(q.text)
    items = [_doc_to_item(d, i+1) for i, d in enumerate(docs)]
    context_text = _build_context_text(items)

    return AskOut(
        question=q.text,
        context=context_text,
        contexts=items
    )

# Optional: manual reindex trigger (protect if needed)
@app.post("/reindex")
def reindex_now():
    info = reindex(force=True)
    INDEX_STATUS.update({"state": "ready", "detail": str(info), "last_commit": info.get("commit")})
    return {"ok": True, "info": info}

# --------------------- Entrypoint ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
