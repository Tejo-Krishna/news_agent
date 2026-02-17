# agent.py
from __future__ import annotations

from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from rag_chain import rag_answer
from config import OPENAI_CHAT_MODEL


# -------------------------
# Tool input schemas
# -------------------------
class NewsRAGArgs(BaseModel):
    question: str = Field(..., description="User's question about the news")


class DigestArgs(BaseModel):
    topic: str = Field(..., description="Topic to create a short news digest for")


# -------------------------
# Tools
# -------------------------
@tool("news_rag", args_schema=NewsRAGArgs)
def news_rag_tool(question: str) -> Dict[str, Any]:
    """
    Answer a news question using RAG over the ingested RSS corpus.
    Returns structured JSON.
    """
    return rag_answer(question=question, chat_history=[])


@tool("daily_digest", args_schema=DigestArgs)
def daily_digest_tool(topic: str) -> str:
    """
    Create a short digest for a topic based on RAG context.
    """
    resp = rag_answer(
        question=f"Create a compact digest about '{topic}'. Summarize key stories and why they matter.",
        chat_history=[]
    )
    bullets = "\n".join([f"- {b}" for b in resp.get("takeaways", [])])
    sources = resp.get("sources", [])
    src_lines = "\n".join([f"- {s.get('title','')} | {s.get('link','')}" for s in sources[:5]])

    return f"{resp.get('answer','')}\n\nTakeaways:\n{bullets}\n\nSources:\n{src_lines}"


# -------------------------
# Agent runner
# -------------------------
def build_agent_runner():
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.2)

    tools = [news_rag_tool, daily_digest_tool]
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a news assistant.\n"
             "Use tools when needed:\n"
             "- If user asks a factual news question -> use news_rag\n"
             "- If user asks for a digest/briefing -> use daily_digest\n"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def run(input_text: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        msgs = []
        for m in (chat_history or []):
            if m.get("role") == "user":
                msgs.append(HumanMessage(content=m.get("content", "")))
            else:
                msgs.append(AIMessage(content=m.get("content", "")))

        out = (prompt | llm_with_tools).invoke({"input": input_text, "chat_history": msgs})

        # If no tool calls, return the model response
        if not getattr(out, "tool_calls", None):
            return out.content

        rendered = []
        for tc in out.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}

            if name == "news_rag":
                res = news_rag_tool.invoke(args)
                ans = res.get("answer", "")
                takeaways = "\n".join([f"- {t}" for t in res.get("takeaways", [])])
                sources = "\n".join([f"- {s.get('title','')} | {s.get('link','')}" for s in res.get("sources", [])[:5]])
                rendered.append(f"{ans}\n\nTakeaways:\n{takeaways}\n\nSources:\n{sources}")

            elif name == "daily_digest":
                rendered.append(daily_digest_tool.invoke(args))

            else:
                rendered.append(f"(Unknown tool: {name})")

        return "\n\n---\n\n".join(rendered)

    return run
