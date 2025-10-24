#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, json, time
from typing import List, Dict, Any
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

warnings.filterwarnings("ignore")
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

log = logging.getLogger("rag_api")
logging.basicConfig(level=logging.INFO)

# --- Env ---
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/tmp/chroma_db")
COLLECTION_NAME = ENV("RAG_COLLECTION", "career_gpt")
DATASET_ID = ENV("RAG_DATASET_ID", "internationalscholarsprogram/DOC")
DATA_REV = ENV("RAG_DATASET_REVISION", "main")
CORPUS_DIR = ENV("RAG_CORPUS_DIR", "/data/corpus")
STATE_FILE = "/data/.state.json"
PORT = int(ENV("PORT", "7860"))
HOST = "0.0.0.0"

# Managed endpoints + gen params
HF_PRIMARY_ENDPOINT = (ENV("HF_PRIMARY_ENDPOINT", "") or "").rstrip("/")
HF_FALLBACK_ENDPOINT = (ENV("HF_FALLBACK_ENDPOINT", "") or "").rstrip("/")
HF_API_TOKEN = ENV("HF_API_TOKEN", "")
FRIENDLI_API_KEY = ENV("FRIENDLI_API_KEY", "")
REQUEST_TIMEOUT = int(ENV("REQUEST_TIMEOUT_MS", "60000")) / 1000
GEN_MAX_TOKENS = int(ENV("GEN_MAX_TOKENS", "512"))
GEN_TEMPERATURE = float(ENV("GEN_TEMPERATURE", "0.4"))

# --- Embeddings + Vector DB ---
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
os.makedirs(DB_DIR, exist_ok=True)
client = PersistentClient(path=DB_DIR)
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k: int = 4):
    return vectordb.as_retriever(search_type="mmr")

# --- Prompt (we’ll format it and send to the endpoints) ---
prompt = ChatPromptTemplate.from_template(
    "Use <context> to answer. If not sure, say 'I don’t know.'\n\nQ: {question}\n<context>\n{context}\n</context>"
)
parser = StrOutputParser()

# -------- Managed Endpoint Caller (HF primary + Friendli fallback) --------
def _is_hf(url: str) -> bool:
    return "endpoints.huggingface.cloud" in urllib.parse.urlparse(url).netloc

def _is_friendli(url: str) -> bool:
    return "friendli.ai" in urllib.parse.urlparse(url).netloc

def _payload_openai_style(content: str) -> Dict[str, Any]:
    # model field is ignored by some managed endpoints but safe to include
    return {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": content}],
        "temperature": GEN_TEMPERATURE,
        "max_tokens": GEN_MAX_TOKENS,
        "stream": False
    }

def _payload_tgi_generate(content: str) -> Dict[str, Any]:
    return {
        "inputs": content,
        "parameters": {"max_new_tokens": GEN_MAX_TOKENS, "temperature": GEN_TEMPERATURE}
    }

def _extract_text_from_openai(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(resp_json)  # last resort for debugging

def _extract_text_from_tgi(resp_json: Dict[str, Any]) -> str:
    # TGI /generate returns dict with "generated_text" or a list of generations
    if isinstance(resp_json, dict) and "generated_text" in resp_json:
        return resp_json["generated_text"]
    # Some runtimes return a list like [{"generated_text": "..."}]
    if isinstance(resp_json, list) and resp_json and "generated_text" in resp_json[0]:
        return resp_json[0]["generated_text"]
    return json.dumps(resp_json)

def call_hf_managed(formatted_prompt: str) -> str:
    if not HF_PRIMARY_ENDPOINT:
        raise RuntimeError("HF_PRIMARY_ENDPOINT is not set")
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
    with httpx.Client(timeout=REQUEST_TIMEOUT) as s:
        # Try OpenAI-compatible first
        url = f"{HF_PRIMARY_ENDPOINT}/v1/chat/completions"
        r = s.post(url, headers=headers, json=_payload_openai_style(formatted_prompt))
        if r.status_code == 200:
            return _extract_text_from_openai(r.json())
        # If not found or not supported, try TGI classic
        url = f"{HF_PRIMARY_ENDPOINT}/generate"
        r2 = s.post(url, headers=headers, json=_payload_tgi_generate(formatted_prompt))
        if r2.status_code == 200:
            return _extract_text_from_tgi(r2.json())
        raise RuntimeError(f"HF endpoint error {r.status_code}/{r2.status_code}: {r.text[:160]} | {r2.text[:160]}")

def call_friendli_fallback(formatted_prompt: str) -> str:
    if not HF_FALLBACK_ENDPOINT:
        raise RuntimeError("HF_FALLBACK_ENDPOINT is not set")
    headers = {"Authorization": f"Bearer {FRIENDLI_API_KEY}", "Content-Type": "application/json"}
    # most Friendli deployments are OpenAI-compatible; append path if needed
    path = "/v1/chat/completions" if not HF_FALLBACK_ENDPOINT.endswith("/v1/chat/completions") else ""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as s:
        url = f"{HF_FALLBACK_ENDPOINT}{path}"
        r = s.post(url, headers=headers, json=_payload_openai_style(formatted_prompt))
        if r.status_code == 200:
            return _extract_text_from_openai(r.json())
        raise RuntimeError(f"Friendli endpoint error {r.status_code}: {r.text[:160]}")

def generate_answer(question: str, context: str) -> str:
    # format prompt using your LangChain template
    formatted = prompt.format(question=question, context=context)
    # Try HF primary → Friendli fallback
    try:
        return call_hf_managed(formatted)
    except Exception as e1:
        log.warning(f"HF primary failed: {e1}")
        try:
            return call_friendli_fallback(formatted)
        except Exception as e2:
            log.error(f"Friendli fallback failed: {e2}")
            raise

# -------- Dataset sync + indexing --------
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

# -------- FastAPI --------
app = FastAPI(title="Career GPT RAG API", version="1.3.1")
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
    # Quick probe: checks that endpoints are configured (not calling them)
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
    docs = build_retriever(4).invoke(q.question)
    ctx = "\n\n".join([d.page_content for d in docs])
    try:
        text = generate_answer(q.question, ctx)
        return {"answer": text}
    except Exception as e:
        # return a clean 502 with upstream error details
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {str(e)[:500]}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
