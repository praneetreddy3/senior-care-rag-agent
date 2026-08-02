"""Smoke + unit tests. Run with: pytest -q

Uses the TF-IDF embedding backend so they run fast and offline (no model
download, no API key).
"""
import os

os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")

import numpy as np
import pytest

from src.ingest import Chunk, load_chunks, split_text
from src.pipeline import build_index
from src.vectorstore import VectorStore
from src import llm
from src.agent import Agent, condense_question, smalltalk_reply


def test_split_text_overlaps():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(chunks)


def test_corpus_loads():
    chunks = load_chunks()
    assert len(chunks) > 0
    assert all(c.text and c.source for c in chunks)


def test_retrieval_finds_pace_eligibility():
    retriever = build_index(save=False)
    results = retriever.retrieve("What age is required to enroll in PACE?", top_k=4)
    sources = [c.source for c, _ in results]
    assert "pace_eligibility_enrollment.md" in sources


def test_agent_returns_answer_and_citations():
    agent = Agent(retriever=build_index(save=False))
    state = agent.run("Who is eligible for PACE?")
    assert state.get("answer")
    assert state.get("citations")
    assert isinstance(state.get("notes"), list)


def test_vectorstore_roundtrip(tmp_path):
    """Saving then loading a store returns the same chunks and results."""
    chunks = [
        Chunk(id="a::0", text="grab bars prevent falls", source="a.md", title="A", chunk_index=0),
        Chunk(id="b::0", text="pace eligibility age 55", source="b.md", title="B", chunk_index=0),
    ]
    embs = np.eye(2, dtype="float32")  # two orthogonal unit vectors
    store = VectorStore(embs, chunks)
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    hits = loaded.search(np.array([1.0, 0.0], dtype="float32"), top_k=1)
    assert hits[0][0].source == "a.md"


def test_smalltalk_detection():
    """Greetings get a warm reply; real questions do not."""
    assert smalltalk_reply("hi") is not None
    assert smalltalk_reply("how are you") is not None
    assert smalltalk_reply("thanks") is not None
    # a real question must NOT be treated as small talk
    assert smalltalk_reply("who is eligible for PACE?") is None
    assert smalltalk_reply("hi who is eligible for pace") is None


def test_agent_run_smalltalk_no_sources():
    agent = Agent(retriever=build_index(save=False))
    state = agent.run("hello")
    assert state["answer"]
    assert state["citations"] == []
    assert state["grounded"] is True


def test_condense_question_behavior():
    """No history → query returned unchanged (LLM-independent). With history and no
    LLM configured → also unchanged. (With an LLM key set, condensing rewrites it.)"""
    q = "what about someone under 55?"
    assert condense_question(q, []) == q
    if not llm.settings.llm_enabled:
        assert condense_question(q, [{"role": "user", "content": "PACE eligibility?"}]) == q


def test_agent_stream_event_sequence():
    agent = Agent(retriever=build_index(save=False))
    events = list(agent.stream("How can I prevent falls at home?"))
    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    assert "token" in types
    done = events[-1]
    assert done["answer"] and isinstance(done["citations"], list)


def test_chroma_store_optional(tmp_path):
    """ChromaDB backend works when installed; skipped otherwise."""
    pytest.importorskip("chromadb")
    from src.vectorstore import ChromaVectorStore

    chunks = [
        Chunk(id="a::0", text="grab bars prevent falls", source="a.md", title="A", chunk_index=0),
        Chunk(id="b::0", text="pace eligibility age 55", source="b.md", title="B", chunk_index=0),
    ]
    embs = np.eye(2, dtype="float32")
    store = ChromaVectorStore.build(embs, chunks, directory=tmp_path)
    hits = store.search(np.array([0.0, 1.0], dtype="float32"), top_k=1)
    assert hits[0][0].source == "b.md"
