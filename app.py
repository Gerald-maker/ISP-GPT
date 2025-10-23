#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Career GPT RAG API — FastAPI over Chroma + Embeddings + HuggingFace Inference LLM
Optimized for Hugging Face Spaces deployment.

Enhancements in this version:
- Pull PDFs from a Hugging Face DATASET repo into /data/corpus (persistent storage)
- Auto-(re)index Chroma when the dataset commit SHA changes
- /refresh endpoint to force re-pull + reindex without redeploying

Space requirements:
- Enable Persistent storage in Space settings
- Set env (optional defaults shown below):
    RAG_DATASET_ID=internationalscholarsprogram/DOC
    RAG_DATASET_REVISION=main
    RAG_DB_DIR=/data/chroma_db
    RAG_CORPUS_DIR=/data/corpus
- Add to requirements.txt: huggingface_hub, pypdf, langchain (or your version)
"""

import os, sys, logging, warnings, json, shutil
from typing import List, Optional, Iterable, Dict, Any

# -------------------- Quiet warnings --------------------
if not sys.warnoptions:
    warnings.simplefilter("ignore")
for cat in (DeprecationWarning, UserWarning, FutureWarning):
    warnings.filterwarnings("ignore", category=cat)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("rag_api")
for _noisy in ["httpx", "chromadb", "uvicorn", "langchain", "asyncio"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# -------------------- Imports --------------------
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Vector store
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback

# LLM endpoint
try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    from langchain_community.llms import HuggingFaceEndpoint  # fallback

# Embeddings
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    FastEmbedEmbeddings,
)
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings  # optional
except ImportError:
    HFEmbeddings = None

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_core.runnables import RunnableParallel
except ImportError:
    RunnableParallel = None

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings  # modern base

# NEW: dataset + PDF loading helpers
from huggingface_hub import snapshot_download, HfApi   # <-- fixed import
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------- Config --------------------
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/data/chroma_db")               # persistent Chroma dir
EMBED_PROVIDER = ENV("RAG_EMBED_PROVIDER", "bge").lower()   # bge | fastembed | hf_local
EMBED_MODEL = ENV("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = ENV("RAG_DEVICE", "cpu")
HF_TOKEN = ENV("HUGGINGFACEHUB_API_TOKEN", "")
USE_PREFIX = ENV("RAG_BGE_PREFIX", "1") not in ("0", "false", "False")
EMBED_BATCH = int(ENV("RAG_EMBED_BATCH", "32"))
TOP_K_DEFAULT = int(ENV("RAG_TOP_K", "4"))
FETCH_K = int(ENV("RAG_FETCH_K", str(max(TOP_K_DEFAULT * 3, 12))))
LAMBDA_MMR = float(ENV("RAG_LAMBDA", "0.5"))
NUM_CTX = int(ENV("RAG_NUM_CTX", "8192"))
NUM_PREDICT = int(ENV("RAG_NUM_PREDICT", "512"))
TEMPERATURE = float(ENV("RAG_TEMPERATURE", "0.2"))
FALLBACK_MSG = ENV(
    "RAG_FALLBACK_MSG",
    "I am Career GPT for International Scholars Program and I’m still under training. "
    "I hope I’ll keep learning and improve my responses next time."
)
API_KEY = ENV("RAG_API_KEY")                                # optional bearer key for /ask
HOST = ENV("RAG_HOST", "0.0.0.0")
PORT = int(ENV("PORT", ENV("RAG_PORT", "7860")))            # Spaces $PORT first
CORS_ORIGINS = ENV("RAG_CORS_ORIGINS", "*")

# NEW: dataset sync locations
DATASET_ID   = ENV("RAG_DATASET_ID", "internationalscholarsprogram/DOC")
DATA_REV     = ENV("RAG_DATASET_REVISION", "main")          # tag/branch/sha, or "main"
CORPUS_DIR   = ENV("RAG_CORPUS_DIR", "/data/corpus")        # where PDFs are downloaded
STATE_FILE   = ENV("RAG_STATE_FILE", "/data/.state.json")   # remembers last indexed commit

# -------------------- Embeddings --------------------
def batched(iterable: Iterable, n: int):
    b = []
    for x in iterable:
        b.append(x)
        if len(b) >= n:
            yield b
            b = []
    if b:
        yield b

class BGEAdapter(Embeddings):
    """Add 'passage:' and 'query:' prefixes (BGE best practice)."""
    def __init__(self, base: Embeddings, use_prefixes: bool = True):
        self.base = base
        self.use_prefixes = use_prefixes

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.use_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if self.use_prefixes:
            text = f"query: {text}"
        return self.base.embed_query(text)

def build_embeddings(provider: str, model: str, device: str,
                     use_prefixes: bool, hf_token: str, batch_size: int) -> Embeddings:
    provider = (provider or "").lower()
    if provider in ("bge", "hf_bge", "bge_small"):
        base = HuggingFaceBgeEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return BGEAdapter(base, use_prefixes=use_prefixes)

    if provider in ("fastembed", "fe"):
        return FastEmbedEmbeddings()

    if HFEmbeddings is not None and provider in ("hf_local", "hf", "sentence_transformers", ""):
        base = HFEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        return BGEAdapter(base, use_prefixes=("bge" in model.lower() and use_prefixes))

    return FastEmbedEmbeddings()

embeddings = build_embeddings(
    provider=EMBED_PROVIDER,
    model=EMBED_MODEL,
    device=DEVICE,
    use_prefixes=USE_PREFIX,
    hf_token=HF_TOKEN,
    batch_size=EMBED_BATCH,
)

# -------------------- Vector DB handle (created now; filled later) --------------------
os.makedirs(DB_DIR, exist_ok=True)
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

def build_retriever(k: int):
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MMR}
    )

# -------------------- LLM (Hugging Face Inference) --------------------
HF_LLM_REPO = ENV("HF_LLM_REPO", "mistralai/Mistral-7B-Instruct-v0.3")

if not HF_TOKEN:
    log.warning("HUGGINGFACEHUB_API_TOKEN is not set; /ask will fail until you add the secret.")

llm = HuggingFaceEndpoint(
    repo_id=HF_LLM_REPO,
    task="text-generation",
    max_new_tokens=NUM_PREDICT,
    temperature=TEMPERATURE,
    timeout=120,
    huggingfacehub_api_token=HF_TOKEN,
)

SYSTEM_RULES = (
    "You are a careful RAG assistant for the International Scholars Program.\n"
    "Use only the information inside <context> to answer.\n"
    "If the answer is not fully supported by the context, say exactly: \"I don’t know.\""
)
prompt = ChatPromptTemplate.from_template(
    f"{SYSTEM_RULES}\n\n<question>\n{{question}}\n</question>\n\n<context>\n{{context}}\n</context>\n\n"
    "Answer concisely and include source tags like [1], [2] where relevant."
)

parser = StrOutputParser()
chain = (prompt | llm | parser)

# -------------------- Dataset sync & indexing --------------------
def _state_load() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _state_save(st: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)

def sync_pdfs(revision: str = DATA_REV) -> str:
    """
    Pull/update PDFs from the HF dataset into CORPUS_DIR and return the exact commit sha.
    Uses ETag-aware snapshot_download → only changed files are fetched.
    """
    os.makedirs(CORPUS_DIR, exist_ok=True)
    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=CORPUS_DIR,
        local_dir_use_symlinks=False,
    )
    # --- fixed: use HfApi().repo_info instead of removed get_repo_info ---
    api = HfApi()
    info = api.repo_info(repo_id=DATASET_ID, repo_type="dataset", revision=revision)
    return info.sha

def list_pdf_paths(root: str) -> List[str]:
    out: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".pdf"):
                out.append(os.path.join(r, f))
    return sorted(out)

def load_docs_from_pdfs(pdf_paths: List[str]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    docs: List[Document] = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            for c in chunks:
                c.metadata.setdefault("source", path)
            docs.extend(chunks)
        except Exception as e:
            log.error(f"Failed to parse {path}: {e}")
    return docs

def _reset_chroma_dir():
    # safest reset: delete the dir and recreate
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

def rebuild_chroma(docs: List[Document]):
    global vectordb
    _reset_chroma_dir()
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    if docs:
        vectordb.add_documents(docs)
        vectordb.persist()

def reindex_if_needed(force: bool = False, revision: str = DATA_REV) -> Dict[str, Any]:
    """
    Pull dataset → compare commit sha → rebuild index if changed or forced.
    """
    new_sha = sync_pdfs(revision)
    st = _state_load()
    old_sha = st.get("dataset_sha")

    if force or (new_sha != old_sha) or (not os.path.isdir(DB_DIR)):
        pdfs = list_pdf_paths(CORPUS_DIR)
        docs = load_docs_from_pdfs(pdfs)
        rebuild_chroma(docs)
        st["dataset_sha"] = new_sha
        _state_save(st)
        return {"reindexed": True, "commit": new_sha, "docs": len(docs)}
    return {"reindexed": False, "commit": new_sha}

# -------------------- Helpers --------------------
def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        tag = f"[{i}] ({src}" + (f", p.{page}" if page is not None else "") + ")"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(parts)

def fallback_msg() -> str:
    return FALLBACK_MSG

def normalize_unknown(answer: str) -> str:
    lowered = answer.strip().lower()
    for p in [
        "i don't know", "i do not know", "not in the context",
        "cannot find", "unsure", "no context"
    ]:
        if p in lowered:
            return fallback_msg()
    return answer

def answer_question(question: str, k: int = TOP_K_DEFAULT) -> Dict[str, Any]:
    docs = build_retriever(k).invoke(question)
    if not docs:
        return {"answer": fallback_msg(), "citations": [], "used_k": k}
    context = format_docs(docs)
    try:
        raw = chain.invoke({"question": question, "context": context})
        answer = normalize_unknown(raw)
    except Exception as e:
        log.exception("LLM error")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    cits = [{"index": i,
             "source": d.metadata.get("source", "unknown"),
             "page": d.metadata.get("page")}
            for i, d in enumerate(docs, 1)]
    return {"answer": answer, "citations": cits, "used_k": k}

# -------------------- FastAPI --------------------
app = FastAPI(title="Career GPT RAG API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",") if o.strip() ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_api_key(authorization: Optional[str] = Header(None)):
    if API_KEY:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")
    return True

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: Optional[int] = Field(None, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    citations: list
    used_k: int

# ---- Startup: sync + (re)index if dataset changed ----
try:
    info = reindex_if_needed(force=False, revision=DATA_REV)
    log.info(f"Index warmup → {info}")
except Exception as e:
    log.exception("Initial sync/index failed")

@app.get("/healthz")
def healthz():
    try:
        # Best-effort count
        count = 0
        try:
            count = vectordb._collection.count()  # type: ignore[attr-defined]
        except Exception:
            meta = vectordb.get(limit=1)
            count = len(meta.get("ids", []))
        st = _state_load()
        return {
            "status": "ok",
            "db_dir": DB_DIR,
            "docs_indexed": count,
            "embed_provider": EMBED_PROVIDER,
            "embed_model": EMBED_MODEL,
            "llm": HF_LLM_REPO,
            "hf_token_present": bool(HF_TOKEN),
            "dataset": DATASET_ID,
            "dataset_rev": DATA_REV,
            "dataset_sha_indexed": st.get("dataset_sha"),
        }
    except Exception as e:
        log.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.get("/health")
def health():
    return healthz()

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _ok: bool = Depends(require_api_key)):
    k = req.top_k or TOP_K_DEFAULT
    if not HF_TOKEN:
        raise HTTPException(status_code=503, detail="Hugging Face token not configured.")
    try:
        return AskResponse(**answer_question(req.question, k=k))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled /ask error")
        raise HTTPException(status_code=500, detail=str(e))

# ---- NEW: manual refresh endpoint ----
@app.post("/refresh")
def refresh(_ok: bool = Depends(require_api_key)):
    """
    Force re-pull dataset + rebuild index (use right after pushing new PDFs).
    """
    try:
        info = reindex_if_needed(force=True, revision=DATA_REV)
        return {"status": "ok", **info}
    except Exception as e:
        log.exception("/refresh failed")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Runner --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
