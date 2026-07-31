"""Model Context Protocol (MCP) server.

Exposes the knowledge base as MCP tools so any MCP-compatible client (Claude
Desktop, IDEs, other agents) can search it and ask grounded questions. This is the
same retrieval core the API and UI use — reused, not duplicated.

Run:
    python mcp_server.py         # stdio transport
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.agent import Agent
from src.pipeline import get_retriever

mcp = FastMCP("senior-care-kb")

_retriever = None
_agent = None


def _retr():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


def _agent_():
    global _agent
    if _agent is None:
        _agent = Agent(retriever=_retr())
    return _agent


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 4) -> list[dict]:
    """Semantic search over the senior-care knowledge base.

    Returns the most relevant passages with their source and similarity score.
    Use this to ground answers in the organization's documents.
    """
    results = _retr().retrieve(query, top_k=top_k)
    return [
        {
            "text": c.text,
            "source": c.source,
            "title": c.title,
            "score": round(score, 4),
        }
        for c, score in results
    ]


@mcp.tool()
def ask(query: str) -> dict:
    """Answer a question with the full agentic RAG pipeline (retrieve → grade →
    generate → verify). Returns the answer, whether it was verified as grounded,
    and the citations used."""
    state = _agent_().run(query)
    return {
        "answer": state.get("answer", ""),
        "grounded": bool(state.get("grounded", False)),
        "citations": state.get("citations", []),
    }


if __name__ == "__main__":
    mcp.run()
