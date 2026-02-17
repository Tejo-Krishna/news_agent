import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")

# GPT generator
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# HF embeddings
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEFAULT_TOP_K = int(os.getenv("TOP_K", "6"))
DEFAULT_MMR = os.getenv("USE_MMR", "true").lower() == "true"

MAX_PER_SOURCE = int(os.getenv("MAX_PER_SOURCE", "25"))
REFRESH_ON_START = os.getenv("REFRESH_ON_START", "false").lower() == "true"
