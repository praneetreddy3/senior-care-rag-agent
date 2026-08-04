# 🩺 Senior Care RAG Agent

![CI](https://github.com/USERNAME/senior-care-rag-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Replace `USERNAME` in the badge URL above after you push.

An **agentic retrieval-augmented generation (RAG)** assistant that answers questions
from a senior-care operations knowledge base (PACE eligibility, Medicare/Medicaid,
care coordination, medication safety, fall prevention, caregiver support). Every
answer is **grounded in source documents** and shown with **citations**, and the
agent **verifies its own answer** before returning it. It's a **multi-turn chatbot**
with **streaming** responses, so follow-up questions work and answers appear as they
generate.

The knowledge base is **domain-swappable** — drop your own PDFs or markdown into
`data/`, rebuild the index, and the same app serves a completely different domain.

> ⚠️ Demo project. The bundled corpus is general, publicly available program
> information. It is **not** medical, legal, or benefits advice.

---

## Why this architecture

The agent is a small **LangGraph** state machine, not a single prompt:

```
          ┌──────────┐   ┌────────┐   ┌──────────┐   ┌────────┐
  query → │ retrieve │ → │ grade  │ → │ generate │ → │ verify │ → answer + citations
          └──────────┘   └────────┘   └──────────┘   └────────┘
           top-k search   drop         grounded       self-check
           (embeddings)   irrelevant   answer w/       "is every claim
                          passages     citations        supported?"
```

- **retrieve** — semantic search over embedded document chunks.
- **grade** — an LLM (or keyword heuristic offline) filters out retrieved passages
  that aren't actually relevant, which reduces hallucination and noise.
- **generate** — answers strictly from the graded context, citing sources inline.
- **verify** — a self-check step flags answers that aren't fully grounded, surfaced
  to the user as a ⚠️ badge.

One core (`src/agent.py`) is served three ways: a **FastAPI** service with a custom
**streaming chat UI**, a **Streamlit UI**, and an **MCP server** — so it plugs into a
web demo, another service, or an MCP-compatible client like Claude Desktop.

**Multi-turn:** follow-up questions are condensed against the conversation history
into a standalone query before retrieval (e.g. "what about someone under 55?" →
"Is someone under 55 eligible for PACE?").

## Tech stack

| Concern            | Choice                                                    |
|--------------------|-----------------------------------------------------------|
| Embeddings         | `sentence-transformers` (MiniLM) · TF-IDF offline fallback |
| Vector store       | Pluggable: FAISS (default) · **ChromaDB** · NumPy fallback  |
| Agent orchestration| LangGraph (`retrieve → grade → generate → verify`)         |
| LLM                | Pluggable: **Anthropic Claude**, OpenAI, or Groq (free) — one interface |
| Serving            | FastAPI (streaming chat UI) · Streamlit · MCP (`FastMCP`)   |
| Chat               | Multi-turn (history-aware query condensing) + SSE streaming |
| Evaluation         | Hit@k, MRR, keyword coverage, LLM-judge faithfulness       |
| Tests              | pytest                                                     |

**Runs with zero credentials.** Without an API key the agent uses an extractive
fallback and keyword grading, so the demo always works; add a key to enable full
LLM generation, grading, and verification.

## Quickstart

```bash
git clone <your-repo-url> && cd senior-care-rag-agent
pip install -r requirements.txt

cp .env.example .env          # optional: add a free Groq key to enable the LLM
python -m scripts.build_index # embed the corpus → storage/

uvicorn api:app --reload      # custom web UI at http://localhost:8000  (API docs at /docs)
# or
streamlit run app.py          # Streamlit UI at http://localhost:8501
# or
python mcp_server.py          # MCP server over stdio
```

Get a free LLM key at <https://console.groq.com/keys> and put it in `.env` as
`LLM_API_KEY`.

**Switching LLM provider** is config-only (no code change). In `.env`:

```ini
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-latest
```

The default is Groq (free). OpenAI works too — see `.env.example`.

**Switching vector store** is also config-only:

```ini
VECTOR_BACKEND=chroma   # default is "faiss"
```
Then rebuild the index (`python -m scripts.build_index`).

## Voice & audio

Two independent paths:

- **Browser (live mic):** the web UI uses the Web Speech API for voice input and
  reads answers aloud — no setup, works in Chrome/Edge.
- **Server-side (optional):** an open-source **Whisper** endpoint transcribes
  uploaded audio files, which also makes the voice path testable:

  ```bash
  pip install -r requirements-audio.txt      # faster-whisper + gTTS
  python scripts/make_sample_audio.py        # writes sample clips to tests/audio/
  python -m pytest tests/test_audio.py -q     # transcribes them and checks the text
  # POST an audio file to the API:
  # curl -F file=@tests/audio/q1_pace_eligibility.mp3 http://localhost:8000/transcribe
  ```

  The Whisper model downloads on first use. This extra is not required for the core
  app or deployment.

## Evaluation

```bash
python -m eval.run_eval --k 4 --json eval/results.json
```

Measured on the 20-question eval set (`eval/eval_set.json`) with the LLM enabled:

| Metric               | Score |
|----------------------|-------|
| Hit@4 (retrieval)    | 0.950 |
| MRR                  | 0.783 |
| Answer keyword cov.  | 0.983 |
| Grounded rate (self-verification) | 1.000 |
| Faithfulness (independent LLM judge) | 0.950 |

Without an `LLM_API_KEY` the app still runs end-to-end (extractive fallback +
keyword grading), just without the generation/faithfulness metrics — see
`eval/results.json` for the raw run.

## Deploy the live demo (free)

**Streamlit Community Cloud:**
1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → New app → pick the repo → entrypoint `app.py`.
3. In **Advanced settings → Secrets**, add:
   ```toml
   LLM_API_KEY = "your_groq_key"
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   ```
4. Deploy — you get a public URL to put on your resume.

## Use your own documents

```bash
# drop any .md / .txt / .pdf files into data/, then:
python -m scripts.build_index
```
No code changes are needed to point the assistant at a new domain.

## Project layout

```
src/            core: ingest, embeddings, vectorstore, llm, agent, pipeline
data/           knowledge base (swappable)
eval/           eval set + metrics runner
tests/          pytest smoke + unit tests
app.py          Streamlit UI (deploy target)
api.py          FastAPI REST service
mcp_server.py   MCP server (search_knowledge_base, ask)
AGENTS.md       guidance for AI coding agents / MCP clients
SKILL.md        MCP skill definition
```

## Productionizing safely for healthcare (PHI / HIPAA)

This demo uses only public, non-sensitive program information. If it handled real
Protected Health Information (PHI), here is how I would harden it — the constraints
matter as much as the features in healthcare:

- **Model access under a BAA.** Never send PHI to an LLM without a signed Business
  Associate Agreement. Use a BAA-covered endpoint (Azure OpenAI, Anthropic/Bedrock)
  or a self-hosted open model — not a consumer API. The pluggable LLM layer here
  makes swapping to a compliant endpoint a config change.
- **Encryption everywhere.** TLS in transit; encryption at rest for the vector store
  and documents; managed secrets (no keys in code or `.env` in prod).
- **Access control + audit logging.** AuthN/AuthZ per user, row-/document-level
  authorization so retrieval only returns data the user may see, and immutable audit
  logs of who asked what and which sources were returned.
- **Data minimization + PHI redaction in logs.** Scrub PHI from prompts/responses
  before logging; short retention windows; configurable deletion (right to erasure).
- **Tenant isolation.** Separate indexes/namespaces per facility or participant so
  data never leaks across tenants.
- **Grounding + human-in-the-loop.** The grade → verify → self-correct loop and
  mandatory citations reduce hallucination; the assistant declines individualized
  medical/legal advice and defers to a clinician. Clinical decisions stay with people.
- **Safety evaluation before release.** Gate deploys on the faithfulness/grounded
  metrics in `eval/`, and monitor them continuously in production.
- **Network isolation.** Private endpoints/VPC, no public egress from the data plane.

## License

MIT — see [LICENSE](LICENSE).
