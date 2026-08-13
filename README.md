# 🩺 Senior Care RAG Agent

![CI](https://github.com/praneetreddy3/senior-care-rag-agent/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An **agentic retrieval-augmented generation (RAG)** assistant that answers questions
about the PACE program (Program of All-Inclusive Care for the Elderly) — eligibility
and enrollment, regulatory and licensure requirements, quality assurance, participant
rights, and how PACE interacts with Medicare and Medicaid — grounded in real CMS and
state policy source documents. Every answer is **grounded in source documents** and
shown with **citations**, and the agent **verifies its own answer** before returning
it. It's a **multi-turn chatbot** with **streaming** responses, so follow-up
questions work and answers appear as they generate.

The knowledge base is **corpus-swappable**: retrieval, embeddings, and the vector
store don't know anything about senior care — drop your own PDFs or markdown into
`data/`, rebuild the index, and they'll index a new document set as-is. The one part
that's tuned to this domain on purpose is `src/agent.py`'s system prompt and
small-talk filter (e.g. "never give individualized medical/legal advice," topic
keywords like "medicare"/"medicaid"), since a real deployment should have guardrails
specific to what it's actually answering questions about. Repointing the whole app at
a different domain means swapping the corpus **and** rewriting that prompt.

> ⚠️ Demo project built on real public CMS/state PACE program documents (see
> `data/`). It is **not** medical, legal, or benefits advice.

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

Measured on the 54-question eval set (`eval/eval_set.json`), written against the
real CMS/state PACE documents in `data/` (a 191-page CMS regulatory manual, a
20-slide MACPAC stakeholder-interview deck, and an 18-slide NY State Medicaid
policy discussion):

| Metric               | Score (LLM-enabled) |
|----------------------|-------|
| Hit@4 (retrieval)    | 0.926 |
| MRR                  | 0.861 |
| Answer keyword cov.  | 0.704 |
| Grounded rate (self-verification) | 0.981 |
| Faithfulness (independent LLM judge) | 0.778 |

Run above is the full 54-question set with `LLM_API_KEY` set (real LLM
grading/generation/verification, which also unlocks the faithfulness metric).
Without a key the app still runs end-to-end on an extractive fallback + keyword
grading — a genuine no-credentials lower bound, not a mocked number, but
retrieval-only (no grounded rate / faithfulness).

## Automated regression monitoring (n8n)

`GET /eval-check` runs a fast random sample of the eval set and returns
`pass`/`fail` against quality thresholds (`EVAL_*_THRESHOLD` in `.env`) — a
lightweight version of the full eval, cheap enough to run on a schedule. An
n8n workflow (`automation/n8n-eval-monitor.json`) polls it every 6 hours and
pushes an alert (via [ntfy.sh](https://ntfy.sh), no account needed) if
retrieval, grounding, or faithfulness regresses. See `automation/README.md`
for the 5-minute setup.

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
No code changes are needed to index a new document set — retrieval works on
anything in `data/`. If you're pointing this at a genuinely different *domain*
(not just more PACE documents), also update the system prompt and small-talk
guardrails in `src/agent.py` so the assistant's tone and safety rules match
what it's actually answering questions about.

## Project layout

```
src/            core: ingest, embeddings, vectorstore, llm, agent, pipeline
data/           corpus: real CMS/state PACE PDFs (regulatory manual, stakeholder
                 interviews, NY expansion policy discussion) — swappable
eval/           eval set (54 questions) + metrics runner
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
  metrics in `eval/`, and monitor them continuously in production.
- **Network isolation.** Private endpoints/VPC, no public egress from the data plane.

## License

MIT — see [LICENSE](LICENSE).
