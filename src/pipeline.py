"""Build and load the retrieval pipeline (embedder + vector store).

Handles persistence of both the vectors and — for the TF-IDF fallback — the
fitted vectorizer, so query-time encoding matches index-time encoding.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

from .config import STORAGE_DIR, settings
from .embeddings import BaseEmbedder, TfidfEmbedder, get_embedder
from .ingest import Chunk, load_chunks
from .vectorstore import ChromaVectorStore, VectorStore

_META = "embedder_meta.json"
_PKL = "embedder.pkl"


class Retriever:
    """Couples an embedder with a vector store for query-time search.

    `store` can be a FAISS/NumPy VectorStore or a ChromaVectorStore — both expose
    the same `search(query_vec, top_k)` method, so this class doesn't care which.
    """

    def __init__(self, embedder: BaseEmbedder, store) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[Chunk, float]]:
        top_k = top_k or settings.top_k
        qvec = self.embedder.encode([query])[0]
        return self.store.search(qvec, top_k)


def build_index(save: bool = True) -> Retriever:
    """Load corpus, embed it, and build (optionally persist) the vector store."""
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("No documents found in data/. Add .md/.txt/.pdf files.")
    corpus = [c.text for c in chunks]
    embedder = get_embedder(corpus)
    embeddings = embedder.encode(corpus)

    if settings.vector_backend == "chroma":
        store = ChromaVectorStore.build(embeddings, chunks)
    else:
        store = VectorStore(embeddings, chunks)

    if save:
        if isinstance(store, VectorStore):
            store.save()
        _save_embedder(embedder)  # needed for query-time encoding (esp. TF-IDF)
    return Retriever(embedder, store)


def load_index() -> Retriever:
    """Load a previously built index from disk."""
    if settings.vector_backend == "chroma":
        store = ChromaVectorStore.load()
    else:
        store = VectorStore.load()
    embedder = _load_embedder()
    return Retriever(embedder, store)


def _index_exists() -> bool:
    if settings.vector_backend == "chroma":
        return ChromaVectorStore.exists()
    return VectorStore.exists()


def get_retriever() -> Retriever:
    """Load the index if present, otherwise build it."""
    if _index_exists():
        try:
            return load_index()
        except Exception:  # noqa: BLE001 - rebuild if the cache is stale/corrupt
            pass
    return build_index()


# ---- embedder persistence --------------------------------------------------
def _save_embedder(embedder: BaseEmbedder, directory: Path | None = None) -> None:
    directory = directory or STORAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    meta = {"backend": settings.embedding_backend, "model": settings.embedding_model}
    if isinstance(embedder, TfidfEmbedder):
        meta["backend"] = "tfidf"
        with open(directory / _PKL, "wb") as f:
            pickle.dump(embedder, f)
    with open(directory / _META, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def _load_embedder(directory: Path | None = None) -> BaseEmbedder:
    directory = directory or STORAGE_DIR
    with open(directory / _META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("backend") == "tfidf":
        with open(directory / _PKL, "rb") as f:
            return pickle.load(f)
    # Dense model: rebuild it explicitly. If it can't load (e.g. a torch/ST version
    # issue), RAISE rather than silently returning a different-dimensioned embedder —
    # that would cause a vector-dimension mismatch at query time. Raising lets
    # get_retriever() rebuild a self-consistent index instead.
    from .embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(meta.get("model", settings.embedding_model))
