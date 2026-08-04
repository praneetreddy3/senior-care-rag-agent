# Changelog

## [0.1.0]
### Added
- Document ingestion with word-based overlapping chunking.
- Embedding layer: sentence-transformers with an offline TF-IDF fallback.
- Vector store: FAISS inner-product search with a NumPy cosine fallback + on-disk persistence.
- Agentic pipeline (LangGraph): retrieve → grade → generate → verify, with a
  sequential fallback when LangGraph isn't installed.
- Three interfaces: FastAPI service (custom web UI at `/`), Streamlit app, MCP server.
- Evaluation pipeline: Hit@k, MRR, keyword coverage, LLM-judge faithfulness.
- Tests (pytest), CI (GitHub Actions), Dockerfile.

### Notes
- Runs with zero credentials via graceful fallbacks; add `LLM_API_KEY` to enable
  full LLM generation, grading, and verification.

## [0.2.0]
### Added
- Multi-turn conversational chat: follow-up questions are condensed against chat
  history into a standalone query before retrieval.
- Streaming responses (Server-Sent Events) via `POST /chat/stream`.
- Redesigned web UI as a proper chat interface (message thread, streaming text,
  live sources + grounded badge).
- Pluggable vector store: **ChromaDB** backend alongside FAISS (`VECTOR_BACKEND`).
- Warm, conversational tone + small-talk handling (greetings skip retrieval).
- Verba-inspired UI: sidebar shell, source cards with relevance-score badges and
  expandable passages, teal theme; voice input/output; adjustable text size.
- Optional server-side **Whisper** transcription (`/transcribe`) with sample-audio
  generator and tests.
- Expanded test suite (vector-store round-trip, streaming, condensing, small talk,
  API, Chroma, optional audio).

## [0.3.0]
### Added
- **Self-correcting RAG**: if the verify step finds an answer isn't grounded, the
  agent widens retrieval and retries (corrective loop, `MAX_RETRIES`).
- Eval now reports a **grounded rate** (agent self-verification) alongside the
  LLM-judge faithfulness; eval set grown to 20 questions.
- README: "Productionizing safely for healthcare (PHI/HIPAA)" section.

## Roadmap
- [ ] Request tracing + token/cost logging (observability).
- [ ] Swap demo corpus for a larger real dataset and re-benchmark.
- [ ] Hybrid retrieval (BM25 + dense) with a reranker.
