---
name: senior-care-kb-search
description: >-
  Search and answer questions about the PACE program (eligibility and
  enrollment, regulatory/licensure requirements, quality assurance, participant
  rights, Medicare/Medicaid interaction) grounded in real CMS and state policy
  documents. Use when a question can be answered from those source documents
  and answers must be grounded with citations.
---

# PACE Knowledge Base Skill

This skill grounds answers in real PACE program policy/regulatory documents using
an agentic RAG pipeline.

## When to use

- A user asks a regulatory or policy question about the PACE program.
- An answer must cite source documents rather than rely on model memory.

## How to use

The skill is backed by an MCP server (`mcp_server.py`) exposing two tools:

1. `search_knowledge_base(query, top_k=4)` — returns the most relevant passages
   with source + similarity score. Use this to gather evidence.
2. `ask(query)` — runs the full pipeline (retrieve → grade → generate → verify)
   and returns an answer, a `grounded` flag, and citations.

Prefer `ask` for end-user answers; use `search_knowledge_base` when you want to
inspect or re-rank the evidence yourself.

## Guardrails

- Answer only from retrieved context; if the context lacks the answer, say so.
- Never provide individualized medical, legal, or benefits advice — defer to a
  qualified professional.
- Always surface citations so the user can verify.

## Extending

Drop new `.md`, `.txt`, or `.pdf` files into `data/` and run
`python -m scripts.build_index` to re-index new documents — no code changes needed.
Repointing the assistant at a genuinely different domain also means updating the
system prompt and guardrails in `src/agent.py` (see `AGENTS.md`).
