"""Load documents from data/ and split them into overlapping chunks."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from .config import DATA_DIR, settings


@dataclass
class Chunk:
    id: str
    text: str
    source: str          # filename
    title: str           # human-friendly document title
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def _read_document(path: Path) -> str:
    """Read a .md/.txt file, or a .pdf if pypdf is installed."""
    if path.suffix.lower() in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install pypdf to ingest PDF files.") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _title_from(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").title()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple word-based sliding window splitter.

    Word-based (rather than character-based) keeps chunks semantically cleaner
    and avoids cutting words in half. Overlap preserves context across chunk
    boundaries so retrieval doesn't miss answers that straddle a split.
    """
    words = text.split()
    if not words:
        return []
    approx_words = max(1, chunk_size // 5)          # ~5 chars per word
    step = max(1, approx_words - overlap // 5)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + approx_words]
        if window:
            chunks.append(" ".join(window))
        if start + approx_words >= len(words):
            break
    return chunks


def load_chunks(data_dir: Path | None = None) -> List[Chunk]:
    data_dir = data_dir or DATA_DIR
    paths = sorted(
        p for p in data_dir.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}
    )
    chunks: List[Chunk] = []
    for path in paths:
        text = _read_document(path)
        title = _title_from(path, text)
        for i, piece in enumerate(
            split_text(text, settings.chunk_size, settings.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    id=f"{path.stem}::{i}",
                    text=piece,
                    source=path.name,
                    title=title,
                    chunk_index=i,
                )
            )
    return chunks
