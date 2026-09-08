"""Vector store management and semantic chunk retrieval using FAISS and SentenceTransformers."""

from __future__ import annotations

from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# Lightweight embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    """Singleton pattern to load the embedding model once."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_and_chunk_docs(sample_data_dir: Path) -> list[str]:
    """Load all .txt and .md files from sample-data and split them into semantic chunks."""
    chunks: list[str] = []
    
    # Ingest both text and markdown documentation files
    file_patterns = ["*.txt", "*.md"]
    for pattern in file_patterns:
        for file_path in sample_data_dir.glob(pattern):
            if file_path.is_file():
                content = file_path.read_text(encoding="utf-8")
                # Split content into paragraph blocks for granular retrieval
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                if paragraphs:
                    chunks.extend(paragraphs)
                elif content.strip():
                    chunks.append(content.strip())
                    
    # Fallback if no files match
    if not chunks:
        chunks.append("Default fallback context: No reference documentation found.")
        
    return chunks


def build_index(chunks: list[str]) -> faiss.IndexFlatIP:
    """Build a normalized FAISS inner-product index from document chunks."""
    model = get_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search_index(query: str, index: faiss.IndexFlatIP, chunks: list[str], top_k: int = 3) -> list[str]:
    """Retrieve the top-k most relevant text chunks for a given query string."""
    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding, top_k)
    
    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
            
    return results