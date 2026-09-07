"""In-memory semantic retrieval over local text documents."""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the embedding model once, on the first embedding request."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def _embed_texts(texts: Sequence[str]) -> NDArray[np.float32]:
    """Return normalized float32 vectors suitable for FAISS cosine search."""
    embeddings = _get_model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def load_and_chunk_docs(directory_path: str | Path) -> list[str]:
    """Read UTF-8 .txt files directly in a directory, in filename order.

    Split each document into windows of up to 500 whitespace-separated words
    with a 50-word overlap. Empty files are skipped, and windows never cross
    file boundaries. Filesystem and decoding errors propagate to the caller.
    """
    chunk_size = 500
    overlap = 50
    chunks: list[str] = []

    for file_path in sorted(Path(directory_path).iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            continue

        words = file_path.read_text(encoding="utf-8").split()
        for start in range(0, len(words), chunk_size - overlap):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break

    return chunks


def build_index(chunks: Sequence[str]) -> faiss.IndexFlatIP:
    """Embed nonempty chunks and return an in-memory inner-product index.

    Normalized vectors make inner-product rankings equivalent to cosine
    similarity. Keep the chunks in their original order for search_index.
    The model applies its own token limit when embedding long chunks.

    Raises:
        ValueError: If no chunks are supplied or a chunk is blank.
    """
    if not chunks or any(not chunk.strip() for chunk in chunks):
        raise ValueError("Provide at least one chunk, with no blank chunks.")

    embeddings = _embed_texts(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: Sequence[str],
    top_k: int = 3,
) -> list[str]:
    """Return up to top_k original chunk texts, ranked by similarity.

    Chunks must match the index's original insertion order. An empty index or
    nonpositive top_k returns no results. A blank query or mismatched chunk
    count raises ValueError.
    """
    if index.ntotal != len(chunks):
        raise ValueError("Chunk count must match the number of indexed vectors.")
    if not query.strip():
        raise ValueError("Query must not be blank.")
    if top_k <= 0 or index.ntotal == 0:
        return []

    query_embedding = _embed_texts([query])
    if query_embedding.shape[1] != index.d:
        raise ValueError("Index dimension does not match the embedding model.")

    _, indices = index.search(query_embedding, min(top_k, index.ntotal))
    return [chunks[int(position)] for position in indices[0] if position >= 0]
