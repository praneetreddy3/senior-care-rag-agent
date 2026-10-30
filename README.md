# Senior Care RAG Agent

**Agentic Retrieval-Augmented Generation for PACE Program Policy Q&A**

![CI](https://github.com/praneetreddy3/senior-care-rag-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project implements an **agentic RAG assistant** that answers questions about
PACE (Program of All-Inclusive Care for the Elderly) — eligibility and enrollment,
regulatory and licensure requirements, quality assurance, participant rights, and
how PACE interacts with Medicare and Medicaid — grounded entirely in real CMS and
state policy source documents. Rather than a single prompt-and-generate call, the
agent runs a **retrieve → grade → generate → verify** loop that filters irrelevant
context before answering, cites every claim, and self-checks its own answer for
grounding before returning it, with a corrective retry when that check fails.

**Key Achievement:** 92.6% retrieval hit rate and 97.8% answer-groundedness on a
54-question evaluation set benchmarked against the actual source PDFs, with an
independent LLM-judge faithfulness score of 77.8%.

**Technologies:** LangGraph • Sentence-Transformers / TF-IDF Embeddings • FAISS /
ChromaDB • Groq / Anthropic Claude / OpenAI / Ollama (pluggable LLM layer) •
FastAPI (SSE streaming) • Streamlit • MCP Server • n8n Automation

## Try it

**Live demo:** <https://senior-care-rag-agent.streamlit.app/> (Streamlit Community Cloud — if the app has been idle, click "wake up" and give it a minute)

Ask something like *"Who is eligible for PACE?"* — every answer cites the source PDF it came from.

> Demo project built on real public CMS/state PACE program documents (see `data/`).
> Not medical, legal, or benefits advice.

---

## System Architecture

The agent is a small **LangGraph** state machine, not a single prompt:

```
          ┌──────────┐   ┌────────┐   ┌──────────┐   ┌────────┐
  query → │ retrieve │ → │ grade  │ → │ generate │ → │ verify │ → answer + citations
          └──────────┘   └────────┘   └──────────┘   └────────┘
           top-k search   drop         grounded       self-check
           (embeddings)   irrelevant   answer w/       "is every claim
                          passages     citations        supported?"
                                                            │
                                                    fails → widen retrieval,
                                                             retry once
```

- **retrieve** — semantic search over embedded document chunks (FAISS or ChromaDB).
- **grade** — a single batched LLM call (or keyword heuristic offline) filters out
  retrieved passages that aren't actually relevant, reducing hallucination and noise.
- **generate** — answers strictly from the graded context, citing sources inline.
- **verify** — a self-check step flags answers that aren't fully grounded, surfaced
  to the user as a warning badge; a failed check triggers one corrective retry with
  a widened retrieval window.

One core (`src/agent.py`) is served three ways: a **FastAPI** service with a custom
streaming chat UI, a **Streamlit** UI, and an **MCP server** — so it plugs into a
web demo, another service, or an MCP-compatible client like Claude Desktop.

**Multi-turn:** follow-up questions are condensed against the conversation history
into a standalone query before retrieval (e.g. "what about someone under 55?" →
"Is someone under 55 eligible for PACE?").

## Corpus

Real public PACE source documents, ingested as PDFs via `pypdf`:

| Document | Description |
|----------|-------------|
| `cms_pace_manual.pdf` | 191-page CMS PACE regulatory manual (chapters on licensure, IDT, QAPI, disenrollment, audits) |
| `macpac_pace_stakeholder_interviews_2025.pdf` | 20-slide MACPAC stakeholder-interview deck on PACE quality, enrollment, and payment |
| `ny_pace_expansion_options_2022.pdf` | 18-slide NY State Medicaid discussion of PACE expansion models and licensure law |

## Tech stack

| Concern             | Choice                                                       |
|----------------------|---------------------------------------------------------------|
| Embeddings           | `sentence-transformers` (MiniLM) · TF-IDF offline fallback   |
| Vector store         | Pluggable: FAISS (default) · ChromaDB · NumPy fallback        |
| Agent orchestration  | LangGraph (`retrieve → grade → generate → verify`)            |
| LLM                  | Pluggable: Anthropic Claude, OpenAI, Groq, or local Ollama — one interface |
| Serving              | FastAPI (streaming chat UI) · Streamlit · MCP (`FastMCP`)      |
| Chat                 | Multi-turn (history-aware query condensing) + SSE streaming   |
| Evaluation           | Hit@k, MRR, keyword coverage, grounded rate, LLM-judge faithfulness |
| Regression monitoring| n8n workflow polling a `/eval-check` endpoint on a schedule    |
| Tests                | pytest, isolated from live LLM calls by default                |

**Runs with zero credentials.** Without an API key the agent uses an extractive
fallback and keyword grading, so the demo always works; add a key to enable full
LLM generation, grading, and verification.

## Quickstart

```bash
git clone https://github.com/praneetreddy3/senior-care-rag-agent.git
cd senior-care-rag-agent
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

**Switching LLM provider** is config-only. In `.env`:

```ini
# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-latest
```

The default is Groq (free) with `openai/gpt-oss-20b`. The 54-question evaluation
below was run locally with Llama 3.1 8B through Ollama. OpenAI and Ollama work too — see
`.env.example`.

**Switching vector store** is also config-only:

```ini
VECTOR_BACKEND=chroma   # default is "faiss"
```
Then rebuild the index (`python -m scripts.build_index`).

**Switching embedding backend** is also config-only:

```ini
EMBEDDING_BACKEND=tfidf   # default is "st" (sentence-transformers)
```
`tfidf` is pure Python with no model download and a much smaller memory
footprint than `sentence-transformers`/`torch` — useful on memory-constrained
hosts, or if you'd rather not download an embedding model at all. Trades some
retrieval quality for that.

## Deployment

The live demo runs on **Streamlit Community Cloud**, deployed straight from this repo:

- Entry point is `app.py`; dependencies come from `requirements.txt` (CPU-only torch to keep the install small).
- On first start the app embeds the PDFs in `data/` and caches the index, so no build artifacts are committed.
- `LLM_API_KEY` is stored in the app's **Secrets** (never committed). Without it the app falls back to extractive answers.

To self-host instead, the `Dockerfile` builds the index into the image and serves Streamlit on port 8501
(or the FastAPI service: `docker run -p 8000:8000 senior-care-rag-agent uvicorn api:app --host 0.0.0.0 --port 8000`).

## Voice & audio

Two independent paths:

- **Browser (live mic):** the web UI uses the Web Speech API for voice input and
  reads answers aloud — no setup, works in Chrome/Edge.
- **Server-side (optional):** an open-source Whisper endpoint transcribes uploaded
  audio files, which also makes the voice path testable:

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

Full 54-question eval set (`eval/eval_set.json`), written against the real CMS/state
PACE documents above, run with an LLM provider enabled (grading, generation, and
self-verification all live, plus an independent LLM-judge faithfulness check):

| Metric                                 | Score |
|-----------------------------------------|-------|
| Hit@4 (retrieval)                       | 0.926 |
| MRR                                     | 0.861 |
| Answer keyword coverage                 | 0.704 |
| Grounded rate (agent self-verification) | 0.981 |
| Faithfulness (independent LLM judge)    | 0.778 |

Without an LLM key the app still runs end-to-end on an extractive fallback and
keyword grading — a genuine no-credentials lower bound on retrieval quality, but
without the grounded-rate or faithfulness metrics, which need an LLM judge.

## Automated regression monitoring (n8n)

`GET /eval-check` runs a fast random sample of the eval set and returns pass/fail
against configurable quality thresholds (`EVAL_*_THRESHOLD` in `.env`) — a
lightweight version of the full eval, cheap enough to run on a schedule instead of
by hand.

An n8n workflow (`automation/n8n-eval-monitor.json`) polls that endpoint every 6
hours and pushes a push notification via [ntfy.sh](https://ntfy.sh) (no account
needed) when retrieval, grounding, or faithfulness regresses below threshold:

```
Schedule (6h) → GET /eval-check → pass? ──no──→ push alert with failure detail
                                    │
                                   yes → no-op
```

Both branches were tested live end to end — including deliberately forcing a
threshold failure and confirming the push notification fires with the specific
metric that regressed. See `automation/README.md` for setup.

## Project layout

```
src/            core: ingest, embeddings, vectorstore, llm, agent, pipeline
data/           corpus: real CMS/state PACE PDFs (regulatory manual, stakeholder
                interviews, NY expansion policy discussion)
eval/           eval set (54 questions), metrics runner, fast regression checker
automation/     n8n workflow definition + setup guide
tests/          pytest smoke + unit tests
scripts/        index builder, sample-audio generator
frontend/       custom streaming chat UI served by api.py
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
  metrics in `eval/`, and monitor them continuously in production via the `/eval-check`
  regression endpoint.
- **Network isolation.** Private endpoints/VPC, no public egress from the data plane.

## License

MIT — see [LICENSE](LICENSE).
