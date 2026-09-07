"""Exercise the Read and Reason loop with a simulated RFP and live Groq calls."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from reasoner.groq_reasoner import process_all_sections
from reasoner.vector_store import build_index, load_and_chunk_docs


async def main() -> None:
    """Index local capabilities, generate section answers, and print JSON."""
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    chunks = load_and_chunk_docs(project_root / "sample-data")
    faiss_index = build_index(chunks)

    rfp_sections = {
        "Security": "What encryption standard do you use?",
        "Support": "Do you offer 24/7 support?",
        "Pricing": "What is your pricing model?",
    }

    # Indexing finishes first; the wrapper runs Groq section calls concurrently.
    # This read/reason-only check has no submission action or approval lock.
    answers = await process_all_sections(rfp_sections, faiss_index, chunks)
    print(json.dumps(answers, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
