# AGENTS.md

Machine-readable guidance for AI coding agents and MCP clients working in this repo.

## Project

Agentic RAG assistant over a senior-care operations knowledge base. Core pipeline:
`retrieve → grade → generate → verify` (LangGraph), served three ways — a Streamlit
UI (`app.py`), a FastAPI service (`api.py`), and an MCP server (`mcp_server.py`).

## Setup commands

- Install deps: `pip install -r requirements.txt`
- Build the index: `python -m scripts.build_index`
- Run tests: `pytest -q`
- Run the UI: `streamlit run app.py`
- Run the API: `uvicorn api:app --reload`
- Run the MCP server: `python mcp_server.py`

## Architecture rules

- Keep retrieval logic in `src/pipeline.py`; do not duplicate it in the servers.
- All three servers (UI, API, MCP) must reuse `src.agent.Agent` — one brain, many
  front doors.
- Every backend (embeddings, vector store, LLM, agent graph) must degrade
  gracefully when an optional dependency or API key is missing. Preserve the
  fallbacks; the demo must run with zero credentials.

## Safety constraints

- The assistant answers only from retrieved context and must not give individualized
  medical, legal, or benefits advice. Preserve the system prompt in `src/agent.py`.
- Do not commit secrets. Configuration comes from environment variables / `.env`.

## Conventions

- Python 3.10+, type hints, small pure functions, docstrings on public functions.
- Prefer standard library and the already-listed dependencies over adding new ones.
