"""Vector store with a FAISS backend and a pure-NumPy fallback.

Vectors are L2-normalized, so inner-product search == cosine similarity.
The store persists to disk (embeddings + chunk metadata) so we don't re-embed
the corpus on every startup.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import STORAGE_DIR
from .ingest import Chunk

# ChromaDB's telemetry logs harmless "Failed to send telemetry event" errors even
# when disabled (a known bug); quiet it so the console stays clean.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: List[Chunk]):
        self.embeddings = embeddings.astype("float32")
        self.chunks = chunks
        self._faiss_index = self._try_build_faiss(self.embeddings)

    @staticmethod
    def _try_build_faiss(embeddings: np.ndarray):
        try:
            import faiss

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            return index
        except Exception:  # noqa: BLE001 - fall back to numpy search
            return None

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        query_vec = query_vec.astype("float32").reshape(1, -1)
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(query_vec, top_k)
            pairs = zip(idxs[0], scores[0])
        else:
            sims = (self.embeddings @ query_vec.T).ravel()
            order = np.argsort(-sims)[:top_k]
            pairs = ((i, sims[i]) for i in order)
        return [
            (self.chunks[i], float(score))
            for i, score in pairs
            if 0 <= i < len(self.chunks)
        ]

    # ---- persistence -------------------------------------------------------
    def save(self, directory: Path | None = None) -> None:
        directory = directory or STORAGE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorStore":
        directory = directory or STORAGE_DIR
        embeddings = np.load(directory / "embeddings.npy")
        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            chunks = [Chunk(**d) for d in json.load(f)]
        return cls(embeddings, chunks)

    @staticmethod
    def exists(directory: Path | None = None) -> bool:
        directory = directory or STORAGE_DIR
        return (directory / "embeddings.npy").exists() and (
            directory / "chunks.json"
        ).exists()


class ChromaVectorStore:
    """ChromaDB-backed store with the same `search` interface as VectorStore, so
    the rest of the app is unchanged. We supply our own embeddings (from the
    configured embedder) and let Chroma handle persistence and nearest-neighbour
    search. Enable with VECTOR_BACKEND=chroma.
    """

    COLLECTION = "senior_care_kb"

    def __init__(self, collection, chunks_by_id: dict):
        self.collection = collection
        self._chunks = chunks_by_id

    @staticmethod
    def _dir(directory: Path | None = None) -> Path:
        return (directory or STORAGE_DIR) / "chroma"

    @classmethod
    def build(
        cls, embeddings: np.ndarray, chunks: List[Chunk], directory: Path | None = None
    ) -> "ChromaVectorStore":
        import chromadb

        persist = cls._dir(directory)
        persist.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(cls.COLLECTION)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        col = client.create_collection(
            cls.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=[c.id for c in chunks],
            embeddings=[e.tolist() for e in embeddings.astype("float32")],
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "title": c.title, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )
        return cls(col, {c.id: c for c in chunks})

    @classmethod
    def load(cls, directory: Path | None = None) -> "ChromaVectorStore":
        import chromadb

        client = chromadb.PersistentClient(
            path=str(cls._dir(directory)),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(cls.COLLECTION)
        got = col.get(include=["documents", "metadatas"])
        chunks = {}
        for cid, doc, md in zip(got["ids"], got["documents"], got["metadatas"]):
            chunks[cid] = Chunk(
                id=cid,
                text=doc,
                source=md["source"],
                title=md["title"],
                chunk_index=md.get("chunk_index", 0),
            )
        return cls(col, chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        res = self.collection.query(
            query_embeddings=[query_vec.astype("float32").tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: List[Tuple[Chunk, float]] = []
        for cid, doc, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = self._chunks.get(cid) or Chunk(
                id=cid,
                text=doc,
                source=md["source"],
                title=md["title"],
                chunk_index=md.get("chunk_index", 0),
            )
            out.append((chunk, float(1.0 - dist)))  # cosine distance -> similarity
        return out

    @classmethod
    def exists(cls, directory: Path | None = None) -> bool:
        return (cls._dir(directory) / "chroma.sqlite3").exists()
