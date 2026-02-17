from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from ingest import build_faiss_index, faiss_exists
from config import REFRESH_ON_START
from rag_chain import rag_answer
from agent import build_agent_runner

app = FastAPI(title="HF Space News RAG (HF Embeddings + GPT Generation)")

agent_runner = build_agent_runner()


class AskReq(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = None


class AgentReq(BaseModel):
    input: str
    chat_history: Optional[List[Dict[str, str]]] = None


@app.on_event("startup")
def startup():
    if REFRESH_ON_START or (not faiss_exists()):
        print("🔄 Building FAISS index (startup)...")
        build_faiss_index()
    else:
        print("✅ FAISS index already exists. Skipping ingest.")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: AskReq):
    return rag_answer(req.question, chat_history=req.chat_history or [])


@app.post("/agent")
def agent(req: AgentReq):
    out = agent_runner(req.input, chat_history=req.chat_history or [])
    return {"output": out}
