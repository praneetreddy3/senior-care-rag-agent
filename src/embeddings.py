"""Embedding backends.

Production path: sentence-transformers (dense semantic embeddings).
Fallback path: scikit-learn TF-IDF, so the project runs fully offline / without
downloading a model. The backend is chosen by EMBEDDING_BACKEND, and we fall back
automatically if sentence-transformers can't be imported or loaded.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import settings


class BaseEmbedder:
    dim: int

    def encode(self, texts: List[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype="float32")


class TfidfEmbedder(BaseEmbedder):
    """Offline fallback: L2-normalized TF-IDF vectors (cosine == dot product)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self._fitted = False
        self.dim = 0

    def fit(self, corpus: List[str]) -> None:
        self.vectorizer.fit(corpus)
        self.dim = len(self.vectorizer.get_feature_names_out())
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Fit lazily on first batch if not fit explicitly.
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


def get_embedder(corpus: List[str] | None = None) -> BaseEmbedder:
    """Return an embedder, honoring config and falling back gracefully."""
    backend = settings.embedding_backend
    if backend == "st":
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - deliberate graceful fallback
            print(f"[embeddings] sentence-transformers unavailable ({exc}); "
                  f"falling back to TF-IDF.")
    emb = TfidfEmbedder()
    if corpus:
        emb.fit(corpus)
    return emb
