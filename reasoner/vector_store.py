"""Grounded chunk retrieval with a low-memory Render fallback."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model: Any | None = None

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


def _lightweight_enabled() -> bool:
    return os.environ.get(
        "RFP_LIGHTWEIGHT_RETRIEVAL", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _terms(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def get_embedding_model() -> Any:
    """Load the normal semantic model only for local runs."""
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def load_and_chunk_docs(sample_data_dir: Path) -> list[str]:
    """Load text and markdown documents into evidence chunks."""
    chunks: list[str] = []

    for pattern in ("*.txt", "*.md"):
        for file_path in sample_data_dir.glob(pattern):
            if file_path.is_file():
                content = file_path.read_text(encoding="utf-8")
                paragraphs = [
                    paragraph.strip()
                    for paragraph in content.split("\n\n")
                    if paragraph.strip()
                ]
                chunks.extend(
                    paragraphs
                    or ([content.strip()] if content.strip() else [])
                )

    if not chunks:
        raise RuntimeError("Knowledge base is empty.")

    return chunks


def build_index(chunks: list[str]) -> Any:
    """Use lexical retrieval on Render Free, semantic FAISS locally."""
    if _lightweight_enabled():
        return None

    import faiss

    model = get_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def search_index(
    query: str,
    index: Any,
    chunks: list[str],
    top_k: int = 3,
) -> list[str]:
    """Retrieve grounded chunks using the configured retrieval path."""
    if index is None:
        query_terms = _terms(query)

        ranked = sorted(
            enumerate(chunks),
            key=lambda item: (
                len(query_terms & _terms(item[1])),
                -item[0],
            ),
            reverse=True,
        )

        return [chunk for _, chunk in ranked[:top_k]]

    model = get_embedding_model()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    _, indices = index.search(query_embedding, top_k)

    return [
        chunks[index]
        for index in indices[0]
        if 0 <= index < len(chunks)
    ]