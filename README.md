# News RAG (LangChain) hosted on Hugging Face Spaces, GPT for generation

## What this Space does
- Ingests RSS feeds (BBC/NYT Tech/TechCrunch/TheVerge/Krebs)
- Builds FAISS vector index using OpenAI embeddings
- Answers queries with GPT (OpenAI) using LangChain RAG

## Endpoints
- GET /health
- POST /ask
  body: {"question": "...", "chat_history":[{"role":"user","content":"..."}, ...]}
- POST /agent
  body: {"input":"...", "chat_history":[...]}

## Required HF Space Secrets
Set these in your Space settings:
- OPENAI_API_KEY

Optional env vars:
- OPENAI_CHAT_MODEL (default: gpt-4o-mini)
- OPENAI_EMBED_MODEL (default: text-embedding-3-small)
- TOP_K (default: 6)
- USE_MMR (default: true)
- MAX_PER_SOURCE (default: 25)
- REFRESH_ON_START (default: false)

## Local run
pip install -r requirements.txt
export OPENAI_API_KEY=...
python ingest.py
uvicorn app:app --reload --port 8000
# news_agent
