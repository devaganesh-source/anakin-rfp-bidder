"""Print the exact top-three capability chunks used for the two verification RFPs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")

from crawler import RFPCrawler
from reasoner.vector_store import build_index, load_and_chunk_docs, search_index


PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = PROJECT_ROOT / "sample-data"
INPUTS = {
    "A": "http://127.0.0.1:8080/mock_rfp.html",
    "B": "http://127.0.0.1:8080/mock_rfp_b.html",
}


def chunk_provenance() -> dict[str, list[dict[str, object]]]:
    """Mirror vector_store chunking and retain file and paragraph positions."""
    provenance: dict[str, list[dict[str, object]]] = {}
    global_index = 0
    for pattern in ("*.txt", "*.md"):
        for file_path in SAMPLE_DATA.glob(pattern):
            content = file_path.read_text(encoding="utf-8")
            paragraphs = [part.strip() for part in content.split("\n\n") if part.strip()]
            file_chunks = paragraphs or ([content.strip()] if content.strip() else [])
            for file_chunk_index, chunk in enumerate(file_chunks):
                provenance.setdefault(chunk, []).append(
                    {
                        "document": file_path.name,
                        "document_chunk_index": file_chunk_index,
                        "global_chunk_index": global_index,
                    }
                )
                global_index += 1
    return provenance


async def main() -> None:
    """Replay the production query construction and top-three FAISS lookup."""
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = load_and_chunk_docs(SAMPLE_DATA)
    index = build_index(chunks)
    provenance = chunk_provenance()
    output: dict[str, object] = {}
    selected_case = os.environ.get("INSPECT_CASE", "").strip().upper()
    inputs = INPUTS.items() if not selected_case else (
        (selected_case, INPUTS[selected_case]),
    )
    for label, url in inputs:
        sections = await RFPCrawler(url).crawl()
        section_output: list[dict[str, object]] = []
        for section, requirement in sections.items():
            contexts = search_index(f"{section}\n{requirement}", index, chunks)
            section_output.append(
                {
                    "section": section,
                    "requirement": requirement,
                    "retrieved_context": [
                        {
                            "rank": rank,
                            **provenance[chunk][0],
                            "chunk": chunk,
                        }
                        for rank, chunk in enumerate(contexts, start=1)
                    ],
                }
            )
        output[label] = section_output
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
