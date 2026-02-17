# ingest.py (RSS-only, fast, parallel, with SSL fix for macOS/Python 3.12)

import os
import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

import feedparser

# ✅ SSL fix (CERTIFICATE_VERIFY_FAILED) using certifi
import ssl
import certifi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from sources import SOURCES
from config import FAISS_DIR, HF_EMBED_MODEL, MAX_PER_SOURCE


# Force Python to use certifi CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

def _certifi_https_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _certifi_https_context


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_feed_xml(url: str, timeout_sec: int = 6) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (NewsRAG/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read()


def _load_rss_one(url: str, max_per_source: int) -> List[Document]:
    print(f"\n🔎 Fetching: {url}")
    try:
        xml = _fetch_feed_xml(url, timeout_sec=6)
        feed = feedparser.parse(xml)

        if getattr(feed, "bozo", False):
            print(f"⚠️ bozo_exception ({url}): {feed.bozo_exception}")

        entries = getattr(feed, "entries", []) or []
        print(f"📦 entries found: {len(entries)}")

        docs: List[Document] = []
        kept = 0

        for e in entries[:max_per_source]:
            title = _clean_html(getattr(e, "title", ""))

            summary = _clean_html(getattr(e, "summary", "")) or _clean_html(
                getattr(e, "description", "")
            )

            content = ""
            if hasattr(e, "content") and e.content:
                try:
                    content = _clean_html(e.content[0].value)
                except Exception:
                    content = ""

            text = (title + "\n\n" + (content or summary)).strip()
            if len(text) < 30:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "rss",
                        "source_feed": url,
                        "title": title,
                        "link": getattr(e, "link", ""),
                        "published": getattr(e, "published", "")
                        or getattr(e, "updated", ""),
                    },
                )
            )
            kept += 1

        print(f"✅ kept: {kept}")
        return docs

    except Exception as e:
        print(f"⏭️ Skipping {url} (error/timeout): {e}")
        return []


def fetch_rss_documents(max_per_source: int = MAX_PER_SOURCE) -> List[Document]:
    docs: List[Document] = []

    # Parallel fetch (much faster than sequential)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_load_rss_one, url, max_per_source) for url in SOURCES]
        for f in as_completed(futures):
            docs.extend(f.result())

    print(f"\n📌 TOTAL documents kept: {len(docs)}")
    return docs


def build_faiss_index() -> None:
    t0 = time.time()

    raw_docs = fetch_rss_documents()

    if not raw_docs:
        raise RuntimeError(
            "No RSS documents fetched. If SSL is fixed, check whether your network/VPN blocks RSS URLs."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    print(f"✂️ Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_DIR, exist_ok=True)
    vs.save_local(FAISS_DIR)

    print(f"✅ FAISS saved at: {FAISS_DIR}")
    print(f"   Raw docs: {len(raw_docs)} | Chunks: {len(chunks)}")
    print(f"⏱️ Total ingest time: {time.time() - t0:.1f}s")


def faiss_exists() -> bool:
    return os.path.exists(os.path.join(FAISS_DIR, "index.faiss"))


if __name__ == "__main__":
    build_faiss_index()
