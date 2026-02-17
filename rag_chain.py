from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings

from config import FAISS_DIR, OPENAI_CHAT_MODEL, HF_EMBED_MODEL, DEFAULT_TOP_K, DEFAULT_MMR


class SourceItem(BaseModel):
    title: str = Field(default="")
    link: str = Field(default="")
    published: str = Field(default="")


class RAGResponse(BaseModel):
    answer: str
    takeaways: List[str]
    confidence: str = Field(description="One of: low/medium/high")
    sources: List[SourceItem]


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


def make_retriever(vs: FAISS):
    if DEFAULT_MMR:
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": DEFAULT_TOP_K, "fetch_k": DEFAULT_TOP_K * 3})
    return vs.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})


def format_docs_with_citations(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        blocks.append(
            f"[{i}] TITLE: {meta.get('title','')}\n"
            f"    PUBLISHED: {meta.get('published','')}\n"
            f"    LINK: {meta.get('link','')}\n"
            f"    TEXT: {d.page_content}\n"
        )
    return "\n---\n".join(blocks)


def build_rag_chain():
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2, max_tokens=550)
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a news assistant.\n"
             "Use ONLY the provided context to answer.\n"
             "If the answer is not supported by context, say you don't know.\n"
             "Return valid JSON exactly matching this schema:\n"
             "{format_instructions}"),
            MessagesPlaceholder("chat_history"),
            ("human",
             "User question: {question}\n\n"
             "Context:\n{context}\n"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def rag_answer(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    vs = load_vectorstore()
    retriever = make_retriever(vs)
    chain = build_rag_chain()

    docs = retriever.invoke(question)
    context = format_docs_with_citations(docs)

    messages = []
    for m in (chat_history or []):
        if m.get("role") == "user":
            messages.append(HumanMessage(content=m.get("content", "")))
        else:
            messages.append(AIMessage(content=m.get("content", "")))

    result: RAGResponse = chain.invoke(
        {"question": question, "context": context, "chat_history": messages}
    )

    if not result.sources:
        srcs = []
        for d in docs[:5]:
            meta = d.metadata or {}
            srcs.append(SourceItem(
                title=meta.get("title", ""),
                link=meta.get("link", ""),
                published=meta.get("published", ""),
            ))
        result.sources = srcs

    return result.model_dump()

