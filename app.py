"""Streamlit UI — the live, deployable demo.

Deployed on Hugging Face Spaces (Docker SDK, port 7860) with LLM_API_KEY stored as
a Space secret. Works without a key too (extractive fallback).
"""
from __future__ import annotations

import streamlit as st

from src.agent import Agent
from src.config import settings

st.set_page_config(page_title="Senior Care RAG Agent", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Building knowledge index…")
def load_agent() -> Agent:
    return Agent()


st.title("🩺 Senior Care RAG Agent")
st.caption(
    "Agentic retrieval-augmented generation over a senior-care operations knowledge "
    "base. Answers are grounded in source documents and every claim shows its citation."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Pipeline:** retrieve → grade → generate → verify (LangGraph)\n"
        "- **Retrieval:** sentence-transformers embeddings + FAISS\n"
        "- **LLM:** any OpenAI-compatible endpoint (Groq by default)\n"
        "- **Swap the domain:** drop your own PDFs/markdown in `data/` and rebuild."
    )
    mode = "🟢 LLM enabled" if settings.llm_enabled else "🟡 Extractive fallback (no API key)"
    st.info(f"Mode: {mode}")
    st.markdown(
        "⚠️ Demo only. General program information — not medical, legal, or "
        "benefits advice."
    )

examples = [
    "Who is eligible for PACE?",
    "Is PACE the same as a Medicare Advantage plan?",
    "What does the interdisciplinary team (IDT) do?",
    "How often does CMS audit PACE organizations?",
]

st.write("**Try an example:**")
cols = st.columns(len(examples))
clicked = None
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        clicked = ex

query = st.text_input("Ask a question:", value=clicked or "")

if query:
    agent = load_agent()
    with st.spinner("Thinking…"):
        state = agent.run(query)

    grounded = state.get("grounded", False)
    badge = "✅ Grounded" if grounded else "⚠️ Not fully grounded — verify sources"
    st.subheader("Answer")
    st.markdown(state.get("answer", ""))
    st.caption(badge)

    citations = state.get("citations", [])
    if citations:
        st.subheader("Sources")
        for c in citations:
            st.markdown(
                f"**[{c['n']}]** {c['title']}  \n"
                f"`{c['source']}` · similarity {c['score']}"
            )

    with st.expander("Agent trace"):
        for step in state.get("notes", []):
            st.write("•", step)
