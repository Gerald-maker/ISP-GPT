#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, threading, logging, warnings, sys, json, sqlite3, shutil, time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download, HfApi
from langchain_chroma import Chroma
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

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

# --- Embeddings + Vector DB ---
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
os.makedirs(DB_DIR, exist_ok=True)
client = PersistentClient(path=DB_DIR)
vectordb = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, client=client)

def build_retriever(k:int=4): 
    return vectordb.as_retriever(search_type="mmr")

# --- LLM ---
HF_TOKEN = ENV("HUGGINGFACEHUB_API_TOKEN","")
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",   # ← switched from mistralai/Mistral-7B-Instruct-v0.3
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.2
)
prompt = ChatPromptTemplate.from_template(
    "Use <context> to answer. If not sure, say 'I don’t know.'\n\nQ: {question}\n<context>\n{context}\n</context>"
)
chain = prompt | llm | StrOutputParser()

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
    return [os.path.join(r,f) for r,_,fs in os.walk(root) for f in fs if f.lower().endswith(".pdf")]

def load_docs(paths): 
    docs=[]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    for p in paths:
        for pg in PyPDFLoader(p).load():
            docs += splitter.split_documents([pg])
    return docs

def rebuild_index(docs):
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
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

# --- FastAPI ---
app = FastAPI(title="Career GPT RAG API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"status": "ok", "index_status": INDEX_STATUS}

class Ask(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Ask):
    docs = build_retriever(4).invoke(q.question)
    ctx = "\n\n".join([d.page_content for d in docs])
    try:
        return {"answer": chain.invoke({"question": q.question, "context": ctx})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
