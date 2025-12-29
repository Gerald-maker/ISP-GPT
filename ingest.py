#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, json, re
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header
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

# --------------------- Env ---------------------
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/tmp/chroma_db")
COLLECTION_NAME = ENV("RAG_COLLECTION", "isp_rag")
DATASET_ID = ENV("RAG_DATASET_ID", "internationalscholarsprogram/DOC")
DATA_REV = ENV("RAG_DATASET_REVISION", "main")
CORPUS_DIR = ENV("RAG_CORPUS_DIR", "/data/corpus")
STATE_FILE = "/data/.state.json"

PORT = int(ENV("PORT", "7860"))
HOST = "0.0.0.0"

# Optional: protect reindex endpoint (set in HF Space secrets)
ADMIN_REINDEX_TOKEN = ENV("ADMIN_REINDEX_TOKEN", "").strip()

# --------------------- Embeddings + Vector DB ---------------------
# BGE recommended settings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

os.makedirs(DB_DIR, exist_ok=True)
client = PersistentClient(path=DB_DIR)
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k: int = 4):
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

# --------------------- Text cleanup ---------------------
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "")
    s = CONTROL_CHARS_RE.sub(" ", s)
    s = re.sub(r"[ \t]+$", "", s, flags=re.M)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

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
    out = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(r, f))
    return out

def load_docs(pdf_paths: List[str]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for p in pdf_paths:
        for pg in PyPDFLoader(p).load():
            pg.page_content = clean_text(pg.page_content)
            # ensure useful metadata for citations
            pg.metadata = dict(pg.metadata or {})
            pg.metadata["source_path"] = p
            pg.metadata["source"] = os.path.basename(p)
            # pg.metadata["page"] typically exists from loader
            docs += splitter.split_documents([pg])
    return docs

def rebuild_index(docs):
    # delete existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    new_client = PersistentClient(path=DB_DIR)
    new_db = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=new_client)

    for i in range(0, len(docs), 32):
        new_db.add_documents(docs[i:i+32])

    return new_db

def reindex(force: bool = False) -> Dict[str, Any]:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    new_sha = sync_pdfs()

    old_sha = None
    if os.path.exists(STATE_FILE):
        try:
            old_sha = json.load(open(STATE_FILE, "r"))["dataset_sha"]
        except Exception:
            old_sha = None

    if force or new_sha != old_sha:
        pdfs = list_pdfs(CORPUS_DIR)
        docs = load_docs(pdfs)

        global vectordb
        vectordb = rebuild_index(docs)

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        json.dump({"dataset_sha": new_sha}, open(STATE_FILE, "w"))

        return {"reindexed": True, "commit": new_sha, "chunks": len(docs), "pdfs": len(pdfs)}

    return {"reindexed": False, "commit": new_sha}

# --------------------- FastAPI app ---------------------
app = FastAPI(title="ISP Retriever API (RAG only)", version="2.0.0")

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
        log.exception("Index warmup failed")

@app.on_event("startup")
def _startup():
    threading.Thread(target=warmup, daemon=True).start()

# --------------------- Schemas ---------------------
class AskIn(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    k: Optional[int] = 4

    @property
    def text(self) -> str:
        t = (self.question or self.query or "").strip()
        t = clean_text(t)
        if not t:
            raise ValueError("Provide 'question' or 'query'.")
        return t

class SourceOut(BaseModel):
    source: str
    source_path: str
    page: Optional[int] = None
    snippet: str

class AskOut(BaseModel):
    question: str
    context: str
    sources: List[SourceOut]

# --------------------- Routes ---------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "config": {
            "dataset_id": DATASET_ID,
            "dataset_rev": DATA_REV,
            "collection": COLLECTION_NAME,
            "db_dir": DB_DIR,
            "corpus_dir": CORPUS_DIR,
            "index_status": INDEX_STATUS,
            "reindex_protected": bool(ADMIN_REINDEX_TOKEN),
        }
    }

@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    # Retrieval only (NO LLM generation here)
    try:
        k = int(payload.k or 4)
        k = max(1, min(k, 12))
        docs = build_retriever(k).invoke(payload.text)

        # Build context string
        ctx_parts = []
        sources: List[SourceOut] = []

        for d in docs:
            text = clean_text(d.page_content)
            if not text:
                continue
            ctx_parts.append(text)

            md = d.metadata or {}
            sources.append(SourceOut(
                source=str(md.get("source", "")),
                source_path=str(md.get("source_path", "")),
                page=md.get("page", None),
                snippet=(text[:300] + "…") if len(text) > 300 else text
            ))

        context = "\n\n---\n\n".join(ctx_parts).strip()

        return AskOut(
            question=payload.text,
            context=context,
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {str(e)[:500]}")

@app.post("/reindex")
def reindex_route(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
    force: Optional[bool] = True
):
    # Optional protection
    if ADMIN_REINDEX_TOKEN:
        if not x_admin_token or x_admin_token.strip() != ADMIN_REINDEX_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        info = reindex(force=bool(force))
        INDEX_STATUS.update({"state": "ready", "detail": str(info), "last_commit": info.get("commit")})
        return {"ok": True, "info": info}
    except Exception as e:
        INDEX_STATUS.update({"state": "error", "detail": str(e)})
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)[:500]}")

# --------------------- Entrypoint ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
