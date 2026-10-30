"""Central configuration, loaded from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STORAGE_DIR = ROOT / "storage"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # LLM provider: "openai" (covers Groq/OpenAI/Azure — anything OpenAI-compatible)
    # or "anthropic" (Claude via the native Anthropic SDK).
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower()
    )
    # Groq is OpenAI-compatible and free to start.
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    )
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 1024))

    # Embeddings
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "st").lower()
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Vector store backend: "faiss" (library, default) or "chroma" (ChromaDB)
    vector_backend: str = field(
        default_factory=lambda: os.getenv("VECTOR_BACKEND", "faiss").lower()
    )

    # Retrieval / chunking
    top_k: int = field(default_factory=lambda: _int("TOP_K", 4))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 120))

    # Regression-gate thresholds used by GET /eval-check (see eval/health_check.py).
    # If a metric drops below its threshold, the endpoint reports pass=false so an
    # automation tool (e.g. n8n) can alert on it.
    eval_grounded_threshold: float = field(
        default_factory=lambda: float(os.getenv("EVAL_GROUNDED_THRESHOLD", "0.8"))
    )
    eval_faithfulness_threshold: float = field(
        default_factory=lambda: float(os.getenv("EVAL_FAITHFULNESS_THRESHOLD", "0.75"))
    )
    eval_hit_rate_threshold: float = field(
        default_factory=lambda: float(os.getenv("EVAL_HIT_RATE_THRESHOLD", "0.6"))
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()
