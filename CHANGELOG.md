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

## [0.4.0]
### Changed
- Replaced the hand-written markdown corpus with real public PACE source
  documents (CMS PACE manual, MACPAC stakeholder interview slides, NY State
  Medicaid PACE expansion discussion) — real PDF ingestion via `pypdf`.
- Rewrote `eval/eval_set.json` (15 questions) against the real document content;
  fixed a test that pinned to an old markdown filename.
- README: clarified that retrieval/embeddings/vector store are corpus-agnostic,
  but the system prompt and small-talk guardrails in `src/agent.py` are
  intentionally tuned to this domain (not a fully generic swap).
- Grew `eval/eval_set.json` to 54 questions covering all three source documents
  in depth (regulatory chapters, participant rights, licensure, quality
  assurance, rate-setting).
- Renamed source PDFs to descriptive filenames (`cms_pace_manual.pdf`,
  `macpac_pace_stakeholder_interviews_2025.pdf`,
  `ny_pace_expansion_options_2022.pdf`).
- Fixed a real inconsistency: the chat system prompt, small-talk fallback text,
  frontend "About" copy, and both apps' example questions still referenced
  topics (fall prevention, medication reconciliation, caregiver support) that
  the real PACE source documents don't cover — a live demo would have returned
  "I don't have that information" for its own suggested questions. Replaced
  with topics the corpus actually answers (eligibility, licensure, quality,
  participant rights, Medicare Advantage comparison).
- Widened the small-talk detection guard (`_TOPICAL` regex) to cover the new
  corpus's real topics so genuine questions are never misrouted as greetings.
- README: full line-by-line accuracy pass — fixed a self-contradiction ("no
  code changes needed for a new domain" vs. the corpus-swappable note above
  it), refreshed the eval table and question count, added Makefile/scripts/
  frontend to the project layout, and updated the intro to describe the real
  PACE-specific scope instead of the retired generic senior-care topic list.
- `src/llm.py`: added exponential-backoff retry on rate-limit (429) errors for
  both providers. A 54-question eval run makes many sequential LLM calls
  (grade per passage, generate, verify, condense) and can trip a free-tier
  per-minute token cap even when the daily budget is fine — retrying lets a
  long eval run finish instead of crashing.

## [0.5.0]
### Added
- `GET /eval-check`: fast sampled regression check (pass/fail against
  configurable thresholds) — `eval/health_check.py`, backed by the same
  metrics logic as `eval/run_eval.py` but cheap enough to run on a schedule.
- n8n workflow (`automation/n8n-eval-monitor.json`): polls `/eval-check` every
  6 hours and pushes an alert via ntfy.sh on regression. See
  `automation/README.md` for setup.
- `EVAL_GROUNDED_THRESHOLD` / `EVAL_FAITHFULNESS_THRESHOLD` /
  `EVAL_HIT_RATE_THRESHOLD` config.

## [0.5.1]
### Fixed
- `grade()` was calling the LLM once **per retrieved passage** (up to 8 calls
  on a self-correction retry), plus generate + verify — 6-10 LLM calls per
  question. On a real 54-question eval run against Groq's free tier, this
  reliably tripped the per-minute token cap. Rewrote grading to ask about all
  retrieved passages in a single batched prompt (one call, parses a `N: yes`/
  `N: no` line per passage) — cuts grading to 1 call regardless of `top_k`,
  ~3 LLM calls per question total instead of 6-10.
- OpenAI/Anthropic clients now set `max_retries=0` so only our own
  `_with_retry` backoff runs. Previously the SDK's internal retry ran first,
  then ours ran again on top of it — waits were compounding instead of
  additive, which is why eval runs felt like they hung on rate limits.
- Root cause of the rate-limit trouble: `pytest` was silently making real LLM
  calls whenever a local `.env` had `LLM_API_KEY` set, since nothing forced
  the test suite into the app's own zero-credentials mode. Running the full
  suite (15+ tests, several touching the agent) before an eval run meant the
  eval started with an already-depleted free-tier budget. `conftest.py` now
  clears `LLM_API_KEY` for all test runs by default (opt back in with
  `RUN_LIVE_LLM_TESTS=1`), so `pytest` never touches the network — confirmed
  4-minute run time drops to ~30s and eval runs get the full token budget.
- `grade()`'s batched call was still sending each retrieved passage's full
  ~800-char chunk (up to 8 of them on a self-correction retry) — around 1900
  tokens in one call, right at the ceiling of the free-tier per-minute cap.
  Truncated passages to 250 chars for the relevance check only (judging
  topical relevance needs far less text than actually answering from it);
  `generate`/`verify` still see full context. Cuts the grading call's token
  footprint roughly 3x.
- `_with_retry` now parses the exact wait time Groq's error message reports
  ("...try again in 16.56s") and sleeps that long, instead of guessing with
  blind exponential backoff — retries succeed on the first attempt after
  waiting instead of repeatedly retrying too early.
- `_with_retry` previously only caught 429 rate-limit errors. A `503 "over
  capacity"` from Groq (a provider-side outage, unrelated to our own token
  usage) crashed the run instead of retrying. Widened the retryable set to
  429/5xx/connection/timeout errors — still deliberately excludes 4xx errors
  like bad requests or bad API keys, which should fail fast and visibly
  instead of silently retrying a real bug 8 times.

## [0.5.2]
### Fixed
- **Real correctness bug, not just a rate-limit issue.** `grade()`, `verify()`,
  and `eval/run_eval.py::faithfulness()` all parsed yes/no verdicts with
  `text.startswith("y")`. Hosted frontier models reliably reply with the bare
  word, so this worked in practice — but running against a local model via
  Ollama, replies like "Based on the passage, yes, this is supported" don't
  start with "y" and were silently read as "no". Result: retrieval stayed
  healthy (Hit@4 0.8) while grounded_rate and faithfulness both collapsed to
  near-zero — a misleading regression that had nothing to do with answer
  quality. Added `llm.parse_yes_no()`: searches for a whole-word "yes"/"no"
  in the first ~15 words instead of requiring an exact prefix match. Wired
  into all three call sites.

## [0.5.3]
### Changed
- Full 54-question eval run against local Ollama (`llama3.1:8b`), LLM-enabled
  end to end (grading, generation, self-verification, independent faithfulness
  judge): Hit@4 0.926, MRR 0.861, keyword coverage 0.704, grounded rate 0.981,
  faithfulness 0.778. README eval table updated with these real numbers,
  replacing the earlier TF-IDF-only baseline.

## Roadmap
- [ ] Request tracing + token/cost logging (observability).
- [ ] Swap demo corpus for a larger real dataset and re-benchmark.
- [ ] Hybrid retrieval (BM25 + dense) with a reranker.
