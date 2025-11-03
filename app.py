#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, json, time, asyncio, urllib.parse, re
from typing import List, Dict, Any, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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

# --------------------- Setup & Logging ---------------------
warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

log = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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

# Managed endpoints + gen params (Hugging Face TGI)
HF_PRIMARY_ENDPOINT = (ENV("HF_PRIMARY_ENDPOINT", "") or "").rstrip("/")  # e.g. https://xxx.endpoints.huggingface.cloud
HF_API_TOKEN = ENV("HF_API_TOKEN", "")
REQUEST_TIMEOUT = int(ENV("REQUEST_TIMEOUT_MS", "60000")) / 1000
GEN_MAX_TOKENS = int(ENV("GEN_MAX_TOKENS", "512"))
GEN_TEMPERATURE = float(ENV("GEN_TEMPERATURE", "0.4"))

# --------------------- Embeddings + Vector DB ---------------------
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
os.makedirs(DB_DIR, exist_ok=True)
client = PersistentClient(path=DB_DIR)
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k: int = 4):
    # MMR gives diverse context
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

# --------------------- Prompt ---------------------
prompt = ChatPromptTemplate.from_template(
    "Use <context> to answer. If not sure, say 'I don’t know.'\n\nQ: {question}\n<context>\n{context}\n</context>"
)
parser = StrOutputParser()

# --------------------- Payload builders (TGI) ---------------------
def _payload_tgi_generate(content: str) -> Dict[str, Any]:
    return {
        "inputs": content,
        "parameters": {
            "max_new_tokens": GEN_MAX_TOKENS,
            "temperature": GEN_TEMPERATURE
        }
    }

# --------------------- Response extractors (non-stream) ---------------------
def _extract_text_from_tgi(resp_json: Dict[str, Any]) -> str:
    if isinstance(resp_json, dict) and "generated_text" in resp_json:
        return resp_json["generated_text"]
    if isinstance(resp_json, list) and resp_json and "generated_text" in resp_json[0]:
        return resp_json[0]["generated_text"]
    return json.dumps(resp_json)

# --------------------- Non-stream call against TGI ---------------------
def call_tgi_generate(formatted_prompt: str) -> str:
    if not HF_PRIMARY_ENDPOINT:
        raise RuntimeError("HF_PRIMARY_ENDPOINT is not set")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    url = f"{HF_PRIMARY_ENDPOINT}/generate"
    with httpx.Client(timeout=REQUEST_TIMEOUT) as s:
        r = s.post(url, headers=headers, json=_payload_tgi_generate(formatted_prompt))
        if r.status_code == 200:
            return _extract_text_from_tgi(r.json())
        raise RuntimeError(f"TGI /generate error {r.status_code}: {r.text[:200]}")

def generate_answer(question: str, context: str) -> str:
    formatted = prompt.format(question=question, context=context)
    return call_tgi_generate(formatted)

# --------------------- TGI Streaming: /generate_stream ---------------------
async def _stream_tgi(formatted_prompt: str) -> AsyncGenerator[str, None]:
    """
    Stream tokens from a Hugging Face TGI endpoint (/generate_stream).
    Lines are JSON (sometimes prefixed with 'data: '). Typical shape:
      {"token":{"id":...,"text":"..."}, ...}
    Final line may include {"generated_text":"..."}.
    """
    if not HF_PRIMARY_ENDPOINT:
        raise RuntimeError("HF_PRIMARY_ENDPOINT is not set")

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = _payload_tgi_generate(formatted_prompt)
    url = f"{HF_PRIMARY_ENDPOINT}/generate_stream"

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw:
                    continue
                # handle optional SSE "data: " prefix
                line = raw[6:].strip() if raw.startswith("data: ") else raw.strip()
                try:
                    obj = json.loads(line)
                except Exception:
                    # ignore keepalives or non-JSON lines
                    continue

                # incremental token
                tok = obj.get("token", {})
                if isinstance(tok, dict) and "text" in tok:
                    yield tok["text"]
                    continue

                # final text (optional)
                if obj.get("generated_text"):
                    yield obj["generated_text"]

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

def list_pdfs(root):
    return [os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs if f.lower().endswith(".pdf")]

def load_docs(paths):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for p in paths:
        for pg in PyPDFLoader(p).load():
            docs += splitter.split_documents([pg])
    return docs

def rebuild_index(docs):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    new_client = PersistentClient(path=DB_DIR)
    new_db = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=new_client)
    for i in range(0, len(docs), 32):
        new_db.add_documents(docs[i:i+32])
    return new_db

def reindex(force=False):
    os.makedirs(CORPUS_DIR, exist_ok=True)
    new_sha = sync_pdfs()
    old_sha = json.load(open(STATE_FILE))["dataset_sha"] if os.path.exists(STATE_FILE) else None
    if force or new_sha != old_sha:
        pdfs = list_pdfs(CORPUS_DIR)
        docs = load_docs(pdfs)
        global vectordb
        vectordb = rebuild_index(docs)
        json.dump({"dataset_sha": new_sha}, open(STATE_FILE, "w"))
        return {"reindexed": True, "commit": new_sha, "docs": len(docs)}
    return {"reindexed": False, "commit": new_sha}

# --------------------- FastAPI app ---------------------
app = FastAPI(title="Career GPT RAG API", version="1.4.1")
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
        log.error(e)

@app.on_event("startup")
def _startup():
    threading.Thread(target=warmup, daemon=True).start()

@app.get("/health")
def health():
    cfg = {
        "hf_primary_set": bool(HF_PRIMARY_ENDPOINT),
        "hf_token_set": bool(HF_API_TOKEN),
        "index_status": INDEX_STATUS,
    }
    return {"status": "ok", "config": cfg}

# --------------------- Schemas ---------------------
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

class AskIn(BaseModel):
    # Accept either 'question' or 'query' for convenience
    question: Optional[str] = None
    query: Optional[str] = None

    @property
    def text(self) -> str:
        t = (self.question or self.query or "").strip()
        t = CONTROL_CHARS_RE.sub(" ", t)  # sanitize control chars
        if not t:
            raise ValueError("Provide 'question' or 'query'.")
        return t

class AskOut(BaseModel):
    answer: str

# --------------------- /ask (non-stream) ---------------------
@app.post("/ask", response_model=AskOut)
def ask(q: AskIn):
    # Retrieve context
    docs = build_retriever(4).invoke(q.text)
    ctx = "\n\n".join([d.page_content for d in docs])
    try:
        text = generate_answer(q.text, ctx)
        return {"answer": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {str(e)[:500]}")

# --------------------- /ask_stream (SSE, pure TGI) ---------------------
@app.post("/ask_stream")
async def ask_stream(req: AskIn):
    # 1) Build RAG context
    try:
        docs = build_retriever(4).invoke(req.text)
        ctx = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        async def err():
            yield "event: error\ndata: {\"error\":\"context_build_failed\"}\n\n"
            yield "event: done\ndata: {}\n\n"
        log.error(f"Context build failed: {e}")
        return StreamingResponse(err(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    # 2) Format prompt
    formatted = prompt.format(question=req.text, context=ctx)

    # 3) Stream tokens from TGI and forward as SSE frames
    async def sse():
        # keep-alive comment to placate proxies
        yield ": keep-alive\n\n"
        try:
            async for token in _stream_tgi(formatted):
                if not token:
                    continue
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
            yield "event: done\ndata: {}\n\n"
        except httpx.HTTPStatusError as he:
            err = {"error": "upstream_http", "status": he.response.status_code}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            err = {"error": "upstream_exception", "detail": str(e)[:200]}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# --------------------- Entrypoint ---------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
