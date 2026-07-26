"""Build (or rebuild) the vector index from documents in data/.

Usage:
    python -m scripts.build_index
"""
from src.pipeline import build_index


def main() -> None:
    retriever = build_index(save=True)
    n = len(retriever.store.chunks)
    print(f"Index built and saved to storage/ with {n} chunks.")


if __name__ == "__main__":
    main()
